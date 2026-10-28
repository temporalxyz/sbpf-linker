//! Lowers 512-bit vector intrinsics to sBPF vector ops (see `SPEC.md`).
//!
//! Intrinsics reach the linker as calls to undefined `__svm_*` markers whose
//! `i32` values are opaque SSA tokens. On the whole program after LTO, this
//! pass allocates `v0`–`v29` to the tokens and replaces every marker call with
//! side-effecting inline asm that emits the encoded `CALL_IMM src=2`
//! instruction as one `.quad`. Programs without markers are left untouched.

pub mod ops;
pub mod regalloc;

use std::{
    collections::HashMap,
    ffi::{CStr, c_char},
    fmt, ptr,
};

use bpf_linker::llvm_sys::{
    LLVMInlineAsmDialect, LLVMOpcode,
    analysis::{LLVMVerifierFailureAction, LLVMVerifyModule},
    bit_reader::LLVMParseBitcodeInContext2,
    bit_writer::LLVMWriteBitcodeToMemoryBuffer,
    core::*,
    error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage},
    prelude::*,
    target::{LLVMABISizeOfType, LLVMGetModuleDataLayout},
    transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions,
        LLVMRunPassesOnFunction,
    },
};

use ops::{Kind, LD, LDN, MARKER_PREFIX, MOV, OpInfo, ST, STN, XOR};
use regalloc::{
    ALL_REGS, AllocError, Allocation, Inst, InstKind, Move, Phi, Reg, RegSet,
    Run, Tok,
};

/// Errors in the input program, one message per problem.
#[derive(Debug)]
pub struct VectorError(pub Vec<String>);

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, message) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "vector pass: {message}")?;
        }
        Ok(())
    }
}

impl std::error::Error for VectorError {}

/// Cheap pre-check on raw linker inputs: whether any marker name occurs.
pub fn mentions_markers(bytes: &[u8]) -> bool {
    bytes.windows(MARKER_PREFIX.len()).any(|w| w == MARKER_PREFIX.as_bytes())
}

/// Runs the pass on LLVM bitcode and returns the lowered bitcode.
pub fn lower_bitcode(bitcode: &[u8]) -> Result<Vec<u8>, VectorError> {
    unsafe {
        let context = LLVMContextCreate();
        let buffer = LLVMCreateMemoryBufferWithMemoryRange(
            bitcode.as_ptr().cast(),
            bitcode.len(),
            c"svm-vector".as_ptr(),
            0,
        );
        let mut module = ptr::null_mut();
        let failed = LLVMParseBitcodeInContext2(context, buffer, &mut module);
        LLVMDisposeMemoryBuffer(buffer);
        let result = if failed != 0 {
            Err(VectorError(vec!["failed to parse linked bitcode".into()]))
        } else {
            let result = lower_module(module).map(|_| {
                let out = LLVMWriteBitcodeToMemoryBuffer(module);
                let bytes = std::slice::from_raw_parts(
                    LLVMGetBufferStart(out).cast::<u8>(),
                    LLVMGetBufferSize(out),
                )
                .to_vec();
                LLVMDisposeMemoryBuffer(out);
                bytes
            });
            LLVMDisposeModule(module);
            result
        };
        LLVMContextDispose(context);
        result
    }
}

/// Runs the pass on `module`. Returns whether any marker was lowered.
///
/// # Safety
/// `module` must be a valid module.
pub unsafe fn lower_module(
    module: LLVMModuleRef,
) -> Result<bool, VectorError> {
    unsafe { Pass::new(module).run() }
}

enum Callee {
    /// A function defined in the module.
    Defined(LLVMValueRef),
    /// An indirect call: may use any register.
    Unknown,
}

/// What an allocator instruction stands for in the IR.
enum Site {
    Op {
        call: LLVMValueRef,
        op: &'static OpInfo,
        mode: u8,
        /// Operand numbers of the scalar and the mask. They are read when
        /// the op is emitted: an earlier op may have replaced the value.
        scalar: Option<u32>,
        mask: Option<u32>,
        imm8: u8,
    },
    Select(LLVMValueRef),
    Call(LLVMValueRef, Callee),
}

impl Site {
    /// The IR instruction that code for this site is inserted before.
    fn anchor(&self) -> LLVMValueRef {
        match *self {
            Site::Op { call, .. }
            | Site::Select(call)
            | Site::Call(call, _) => call,
        }
    }
}

/// Vector uses, scalar (`r1`), mask (`r2`) and immediates of a marker call.
type MarkerOperands =
    (Vec<Option<Tok>>, Option<LLVMValueRef>, Option<LLVMValueRef>, [u8; 2]);

/// A load or store that can join a run: `base + offset`, `count` = n of
/// the `ldn`/`stn` it is a register of, 0 for a plain `ld`/`st`.
struct RunKey {
    store: bool,
    base: LLVMValueRef,
    offset: i64,
    count: u8,
}

/// Numbers the runs among the allocator instructions of one block: loads
/// (stores) next to each other, from one base at offsets 64 apart.
fn assign_runs(insts: &mut [Inst], keys: &[Option<RunKey>], next: &mut u32) {
    let mut i = 0;
    while i < insts.len() {
        let Some(first) = &keys[i] else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        while j < insts.len() && j - i < usize::from(regalloc::NUM_REGS) {
            match (&keys[j - 1], &keys[j]) {
                (Some(prev), Some(key))
                    if key.store == first.store
                        && key.base == first.base
                        && key.count == first.count
                        && key.offset == prev.offset + 64 =>
                {
                    j += 1
                }
                _ => break,
            }
        }
        if j - i >= 2 {
            for (k, inst) in insts[i..j].iter_mut().enumerate() {
                inst.run =
                    Some(Run { id: *next, k: k as u8, store: first.store });
            }
            *next += 1;
        }
        i = j;
    }
}

unsafe fn may_access_memory(inst: LLVMValueRef) -> bool {
    use LLVMOpcode::*;
    matches!(
        unsafe { LLVMGetInstructionOpcode(inst) },
        LLVMLoad
            | LLVMStore
            | LLVMAtomicRMW
            | LLVMAtomicCmpXchg
            | LLVMFence
            | LLVMCall
            | LLVMInvoke
            | LLVMCallBr
            | LLVMVAArg
    )
}

/// One function with vector tokens, in allocator form.
struct Function {
    func: regalloc::Func,
    /// IR sites per block, indexed by `Inst::origin`.
    sites: Vec<Vec<Site>>,
    blocks: Vec<LLVMBasicBlockRef>,
    /// IR value of each token.
    tokens: Vec<LLVMValueRef>,
    /// Number of predecessors of each block.
    preds: Vec<usize>,
}

struct Pass {
    module: LLVMModuleRef,
    context: LLVMContextRef,
    markers: HashMap<LLVMValueRef, (&'static OpInfo, u8)>,
    errors: Vec<String>,
}

unsafe fn value_name<'a>(value: LLVMValueRef) -> &'a str {
    let mut len = 0;
    let name = unsafe { LLVMGetValueName2(value, &mut len) };
    if name.is_null() {
        return "";
    }
    let bytes = unsafe { std::slice::from_raw_parts(name.cast::<u8>(), len) };
    std::str::from_utf8(bytes).unwrap_or("<non-utf8>")
}

unsafe fn function_name(function: LLVMValueRef) -> String {
    format!("{:#}", rustc_demangle::demangle(unsafe { value_name(function) }))
}

unsafe fn instructions(
    block: LLVMBasicBlockRef,
) -> impl Iterator<Item = LLVMValueRef> {
    let first = unsafe { LLVMGetFirstInstruction(block) };
    std::iter::successors((!first.is_null()).then_some(first), |&i| {
        let next = unsafe { LLVMGetNextInstruction(i) };
        (!next.is_null()).then_some(next)
    })
}

unsafe fn blocks(
    function: LLVMValueRef,
) -> impl Iterator<Item = LLVMBasicBlockRef> {
    let first = unsafe { LLVMGetFirstBasicBlock(function) };
    std::iter::successors((!first.is_null()).then_some(first), |&b| {
        let next = unsafe { LLVMGetNextBasicBlock(b) };
        (!next.is_null()).then_some(next)
    })
}

unsafe fn functions(
    module: LLVMModuleRef,
) -> impl Iterator<Item = LLVMValueRef> {
    let first = unsafe { LLVMGetFirstFunction(module) };
    std::iter::successors((!first.is_null()).then_some(first), |&f| {
        let next = unsafe { LLVMGetNextFunction(f) };
        (!next.is_null()).then_some(next)
    })
}

unsafe fn users(value: LLVMValueRef) -> Vec<LLVMValueRef> {
    let mut users = Vec::new();
    let mut u = unsafe { LLVMGetFirstUse(value) };
    while !u.is_null() {
        users.push(unsafe { LLVMGetUser(u) });
        u = unsafe { LLVMGetNextUse(u) };
    }
    users
}

/// Whether the index (operand 1) of a two-table permute is used only by
/// `call` and the first table (operand 0) is used again. The permute ties the
/// first table; tying the index instead (`perm2i`) then saves a copy.
unsafe fn index_dies_first(call: LLVMValueRef) -> bool {
    unsafe {
        users(LLVMGetOperand(call, 1)).len() == 1
            && users(LLVMGetOperand(call, 0)).len() > 1
    }
}

unsafe fn is_undef(value: LLVMValueRef) -> bool {
    unsafe { LLVMIsUndef(value) != 0 || LLVMIsPoison(value) != 0 }
}

unsafe fn parent_function(inst: LLVMValueRef) -> LLVMValueRef {
    unsafe { LLVMGetBasicBlockParent(LLVMGetInstructionParent(inst)) }
}

/// Describes where a non-token value used as a vector comes from.
unsafe fn describe_origin(value: LLVMValueRef) -> &'static str {
    unsafe {
        if !LLVMIsAArgument(value).is_null() {
            "a function argument"
        } else if !LLVMIsALoadInst(value).is_null() {
            "a load from memory"
        } else if !LLVMIsACallInst(value).is_null() {
            "the result of a non-intrinsic call"
        } else if !LLVMIsAConstant(value).is_null() {
            "a constant"
        } else {
            "a non-vector computation"
        }
    }
}

impl Pass {
    unsafe fn new(module: LLVMModuleRef) -> Self {
        let context = unsafe { LLVMGetModuleContext(module) };
        Self { module, context, markers: HashMap::new(), errors: Vec::new() }
    }

    fn error(&mut self, function: Option<LLVMValueRef>, message: String) {
        match function {
            Some(f) => {
                let name = unsafe { function_name(f) };
                self.errors.push(format!("in `{name}`: {message}"))
            }
            None => self.errors.push(message),
        }
    }

    fn check(&mut self) -> Result<(), VectorError> {
        if self.errors.is_empty() {
            return Ok(());
        }
        // Report each distinct problem once, with how often it occurs.
        let mut counts: Vec<(String, usize)> = Vec::new();
        for error in self.errors.drain(..) {
            match counts.iter_mut().find(|(e, _)| *e == error) {
                Some((_, n)) => *n += 1,
                None => counts.push((error, 1)),
            }
        }
        Err(VectorError(
            counts
                .into_iter()
                .map(
                    |(e, n)| {
                        if n > 1 { format!("{e} ({n} times)") } else { e }
                    },
                )
                .collect(),
        ))
    }

    unsafe fn run(mut self) -> Result<bool, VectorError> {
        unsafe {
            let mut with_markers = Vec::new();
            for f in functions(self.module) {
                let name = value_name(f);
                if !name.starts_with(MARKER_PREFIX)
                    || LLVMIsDeclaration(f) == 0
                {
                    continue;
                }
                let Some(op) = ops::by_marker(name) else {
                    if LLVMGetFirstUse(f).is_null() {
                        continue;
                    }
                    let message = match ops::unmasked(name) {
                        Some(op) => format!(
                            "`{name}`: `{}` has no such mask mode",
                            op.name
                        ),
                        None => {
                            format!("unknown vector intrinsic marker `{name}`")
                        }
                    };
                    self.error(None, message);
                    continue;
                };
                self.markers.insert(f, op);
                for user in users(f) {
                    if LLVMIsACallInst(user).is_null()
                        || LLVMGetCalledValue(user) != f
                    {
                        self.error(
                            None,
                            format!(
                                "`{name}` is used other than by a direct call"
                            ),
                        );
                    } else if !with_markers.contains(&parent_function(user)) {
                        with_markers.push(parent_function(user));
                    }
                }
            }
            self.check()?;
            if with_markers.is_empty() {
                return Ok(false);
            }

            // Phi copies go on edges, so split critical ones first.
            let options = LLVMCreatePassBuilderOptions();
            for &f in &with_markers {
                let err = LLVMRunPassesOnFunction(
                    f,
                    c"unreachableblockelim,break-crit-edges".as_ptr(),
                    ptr::null_mut(),
                    options,
                );
                if !err.is_null() {
                    let message = LLVMGetErrorMessage(err);
                    let text =
                        CStr::from_ptr(message).to_string_lossy().into_owned();
                    LLVMDisposeErrorMessage(message);
                    self.error(
                        Some(f),
                        format!("running LLVM passes failed: {text}"),
                    );
                }
            }
            LLVMDisposePassBuilderOptions(options);
            self.check()?;

            let defined: Vec<LLVMValueRef> = functions(self.module)
                .filter(|&f| LLVMIsDeclaration(f) == 0)
                .collect();
            let mut analyzed = HashMap::new();
            let mut callees = HashMap::new();
            for &f in &defined {
                let (function, calls) =
                    self.analyze(f, with_markers.contains(&f));
                callees.insert(f, calls);
                if let Some(function) = function {
                    analyzed.insert(f, function);
                }
            }
            self.check()?;

            // Allocate callees first: a vector value live across a call must
            // avoid every register the callee (transitively) writes.
            let mut clobbers: HashMap<LLVMValueRef, RegSet> = HashMap::new();
            let mut allocations = Vec::new();
            for f in post_order(&defined, &callees) {
                let resolve = |callee: &Callee| match callee {
                    Callee::Defined(g) => {
                        clobbers.get(g).copied().unwrap_or(ALL_REGS)
                    }
                    Callee::Unknown => ALL_REGS,
                };
                let mut clobbered: RegSet = callees[&f]
                    .iter()
                    .fold(0, |set, callee| set | resolve(callee));
                if let Some(mut function) = analyzed.remove(&f) {
                    for (block, sites) in
                        function.func.blocks.iter_mut().zip(&function.sites)
                    {
                        for inst in &mut block.insts {
                            if let Site::Call(_, callee) = &sites[inst.origin]
                            {
                                inst.kind = InstKind::Call {
                                    clobbers: resolve(callee),
                                };
                            }
                        }
                    }
                    match regalloc::allocate(&mut function.func) {
                        Ok(allocation) => {
                            clobbered |= allocation.written;
                            allocations.push((f, function, allocation));
                        }
                        Err(err) => self.allocation_error(f, &function, err),
                    }
                }
                clobbers.insert(f, clobbered);
            }
            self.check()?;

            let builder = LLVMCreateBuilderInContext(self.context);
            for (f, function, allocation) in &allocations {
                self.emit(builder, *f, function, allocation);
            }
            LLVMDisposeBuilder(builder);
            self.check()?;
            let markers: Vec<LLVMValueRef> =
                self.markers.keys().copied().collect();
            for marker in markers {
                if LLVMGetFirstUse(marker).is_null() {
                    LLVMDeleteFunction(marker);
                } else {
                    let name = value_name(marker);
                    self.error(
                        None,
                        format!("internal error: `{name}` remains"),
                    );
                }
            }

            let mut message = ptr::null_mut();
            if LLVMVerifyModule(
                self.module,
                LLVMVerifierFailureAction::LLVMReturnStatusAction,
                &mut message,
            ) != 0
            {
                let text =
                    CStr::from_ptr(message).to_string_lossy().into_owned();
                self.error(
                    None,
                    format!(
                        "internal error, invalid IR after lowering: {text}"
                    ),
                );
            }
            LLVMDisposeMessage(message);
            self.check()?;
            Ok(true)
        }
    }

    fn allocation_error(
        &mut self,
        f: LLVMValueRef,
        function: &Function,
        err: AllocError,
    ) {
        let (AllocError::TooManyLive { block, inst, .. }
        | AllocError::ClobberedByCalls { block, inst }) = err;
        let site = |i: usize| {
            &function.sites[block][function.func.blocks[block].insts[i].origin]
        };
        let at = match inst.map(site) {
            Some(Site::Op { op, .. }) => format!("at op `{}`", op.name),
            Some(Site::Select(_)) => "at a select between vectors".into(),
            _ => "at a loop or branch merge".into(),
        };
        let hint = match err {
            AllocError::TooManyLive { .. } => {
                "; keep fewer vectors alive at once (only values whose remaining uses \
                 are in the same basic block can be spilled)"
            }
            AllocError::ClobberedByCalls { .. } => {
                "; inline the callee or do not keep vectors alive across the call"
            }
        };
        self.error(Some(f), format!("{err} {at}{hint}"));
    }

    /// Finds tokens, checks they don't escape, and builds the allocator's
    /// view of `f`. Also returns the calls `f` makes, for the call graph.
    unsafe fn analyze(
        &mut self,
        f: LLVMValueRef,
        has_markers: bool,
    ) -> (Option<Function>, Vec<Callee>) {
        unsafe {
            let mut calls = Vec::new();
            let mut token_of: HashMap<LLVMValueRef, Tok> = HashMap::new();
            let mut tokens = Vec::new();
            let mut new_token =
                |v: LLVMValueRef, tokens: &mut Vec<LLVMValueRef>| {
                    token_of.entry(v).or_insert_with(|| {
                        tokens.push(v);
                        (tokens.len() - 1) as Tok
                    });
                };
            for block in blocks(f) {
                for inst in instructions(block) {
                    if LLVMIsACallInst(inst).is_null() {
                        continue;
                    }
                    let callee = LLVMGetCalledValue(inst);
                    if let Some((op, _)) = self.markers.get(&callee) {
                        if op.kind.defines_vector() {
                            new_token(inst, &mut tokens);
                        }
                    } else if !LLVMIsAInlineAsm(callee).is_null() {
                    } else if LLVMIsAFunction(callee).is_null() {
                        calls.push(Callee::Unknown);
                    } else if LLVMIsDeclaration(callee) == 0 {
                        calls.push(Callee::Defined(callee));
                    }
                }
            }
            if !has_markers {
                return (None, calls);
            }

            // Tokens flow through phis and selects. Every other use escapes.
            let mut next = 0;
            while next < tokens.len() {
                let token = tokens[next];
                next += 1;
                for user in users(token) {
                    let select_value = !LLVMIsASelectInst(user).is_null()
                        && LLVMGetOperand(user, 0) != token;
                    if !LLVMIsAPHINode(user).is_null() || select_value {
                        new_token(user, &mut tokens);
                    } else if let Some(problem) = self.escape(user, token) {
                        self.error(
                            Some(f),
                            format!("a vector value {problem}"),
                        );
                    }
                }
            }
            let token_of = token_of;
            let operand =
                |value: LLVMValueRef| -> Result<Option<Tok>, &'static str> {
                    match token_of.get(&value) {
                        Some(&t) => Ok(Some(t)),
                        None if is_undef(value) => Ok(None),
                        None => Err(describe_origin(value)),
                    }
                };

            let bbs: Vec<LLVMBasicBlockRef> = blocks(f).collect();
            let index: HashMap<LLVMBasicBlockRef, usize> =
                bbs.iter().enumerate().map(|(i, &b)| (b, i)).collect();
            let mut func = regalloc::Func {
                blocks: Vec::with_capacity(bbs.len()),
                num_tokens: tokens.len(),
            };
            let mut sites = Vec::with_capacity(bbs.len());
            let mut preds = vec![0; bbs.len()];
            let mut next_run = 0;
            for &bb in &bbs {
                let mut block = regalloc::Block::default();
                let mut block_sites = Vec::new();
                let term = LLVMGetBasicBlockTerminator(bb);
                for i in 0..LLVMGetNumSuccessors(term) {
                    let succ = index[&LLVMGetSuccessor(term, i)];
                    if !block.succs.contains(&succ) {
                        block.succs.push(succ);
                        preds[succ] += 1;
                    }
                }
                // Whether an instruction that may access memory precedes
                // `inst` since the last allocator instruction.
                let mut barrier = false;
                let mut run_keys = Vec::new();
                for inst in instructions(bb) {
                    if let Some(&def) = token_of.get(&inst)
                        && !LLVMIsAPHINode(inst).is_null()
                    {
                        let mut incoming = Vec::new();
                        for i in 0..LLVMCountIncoming(inst) {
                            let pred = index[&LLVMGetIncomingBlock(inst, i)];
                            match operand(LLVMGetIncomingValue(inst, i)) {
                                Ok(v) => incoming.push((pred, v)),
                                Err(origin) => self.error(
                                    Some(f),
                                    format!("a phi merges a vector value with {origin}"),
                                ),
                            }
                        }
                        block.phis.push(Phi { def, incoming });
                        continue;
                    }
                    if let Some(&def) = token_of.get(&inst)
                        && !LLVMIsASelectInst(inst).is_null()
                    {
                        let mut uses = Vec::new();
                        for k in 1..3 {
                            match operand(LLVMGetOperand(inst, k)) {
                                Ok(v) => uses.push(v),
                                Err(origin) => self.error(
                                    Some(f),
                                    format!("a select chooses between a vector value and {origin}"),
                                ),
                            }
                        }
                        block.insts.push(Inst {
                            kind: InstKind::Select,
                            def: Some(def),
                            uses,
                            origin: block_sites.len(),
                            run: None,
                        });
                        block_sites.push(Site::Select(inst));
                        run_keys.push(None);
                        barrier = false;
                        continue;
                    }
                    if LLVMIsACallInst(inst).is_null() {
                        barrier |= may_access_memory(inst);
                        continue;
                    }
                    let callee = LLVMGetCalledValue(inst);
                    if let Some(&(op, mode)) = self.markers.get(&callee) {
                        let def = token_of.get(&inst).copied();
                        if let Some((mut uses, scalar, mask, imms)) =
                            self.marker_operands(f, inst, op, mode, &operand)
                        {
                            let (op, imms) = match ops::perm2i_of(op, imms[0])
                            {
                                Some((perm2i, imm8))
                                    if mode != 1 && index_dies_first(inst) =>
                                {
                                    uses.swap(0, 1);
                                    (perm2i, [imm8, imms[1]])
                                }
                                _ => (op, imms),
                            };
                            run_keys.push(
                                (!barrier)
                                    .then(|| {
                                        self.run_key(op, mode, scalar, imms)
                                    })
                                    .flatten(),
                            );
                            block.insts.push(Inst {
                                kind: InstKind::Op { tied: op.tied(mode) },
                                def,
                                uses,
                                origin: block_sites.len(),
                                run: None,
                            });
                            let vectors = op.vector_args(mode) as u32;
                            block_sites.push(Site::Op {
                                call: inst,
                                op,
                                mode,
                                scalar: scalar.map(|_| vectors),
                                mask: mask.map(|_| {
                                    vectors + u32::from(scalar.is_some())
                                }),
                                imm8: imms[0],
                            });
                            barrier = false;
                        }
                    } else if LLVMIsAInlineAsm(callee).is_null()
                        && (LLVMIsAFunction(callee).is_null()
                            || LLVMIsDeclaration(callee) == 0)
                    {
                        let site = if LLVMIsAFunction(callee).is_null() {
                            Callee::Unknown
                        } else {
                            Callee::Defined(callee)
                        };
                        block.insts.push(Inst {
                            kind: InstKind::Call { clobbers: ALL_REGS },
                            def: None,
                            uses: Vec::new(),
                            origin: block_sites.len(),
                            run: None,
                        });
                        block_sites.push(Site::Call(inst, site));
                        run_keys.push(None);
                        barrier = false;
                    } else {
                        barrier = true;
                    }
                }
                assign_runs(&mut block.insts, &run_keys, &mut next_run);
                func.blocks.push(block);
                sites.push(block_sites);
            }
            (Some(Function { func, sites, blocks: bbs, tokens, preds }), calls)
        }
    }

    /// Splits a marker call's arguments into vector uses, the scalar, the
    /// mask and the immediates, reporting malformed ones.
    unsafe fn marker_operands(
        &mut self,
        f: LLVMValueRef,
        call: LLVMValueRef,
        op: &OpInfo,
        mode: u8,
        operand: &dyn Fn(LLVMValueRef) -> Result<Option<Tok>, &'static str>,
    ) -> Option<MarkerOperands> {
        unsafe {
            let kind = op.kind;
            let marker = value_name(LLVMGetCalledValue(call));
            let nargs = LLVMGetNumArgOperands(call) as usize;
            let vectors = op.vector_args(mode);
            let expected = vectors
                + usize::from(kind.has_scalar())
                + usize::from(op.reads_mask(mode))
                + kind.imm_args();
            if nargs != expected {
                self.error(
                    Some(f),
                    format!("`{marker}` takes {expected} arguments, but is called with {nargs}"),
                );
                return None;
            }
            let mut ok = true;
            let mut uses = Vec::new();
            for k in 0..vectors {
                match operand(LLVMGetOperand(call, k as u32)) {
                    Ok(v) => uses.push(v),
                    Err(origin) => {
                        self.error(
                            Some(f),
                            format!(
                                "vector operand {k} of `{marker}` is {origin}, not a vector value; \
                                 vectors cannot be loaded from memory or passed between functions \
                                 (use the load intrinsics, or #[inline(always)])"
                            ),
                        );
                        ok = false;
                    }
                }
            }
            let mut next = vectors as u32;
            let mut scalar_arg = |present: bool| {
                present.then(|| {
                    next += 1;
                    LLVMGetOperand(call, next - 1)
                })
            };
            let scalar = scalar_arg(kind.has_scalar());
            let mask = scalar_arg(op.reads_mask(mode));
            let mut imms = [0; 2];
            for (k, imm) in imms.iter_mut().enumerate().take(kind.imm_args()) {
                let value = LLVMGetOperand(call, next + k as u32);
                if LLVMIsAConstantInt(value).is_null() {
                    self.error(
                        Some(f),
                        format!("the immediate of `{marker}` is not a compile-time constant"),
                    );
                    ok = false;
                } else {
                    *imm = LLVMConstIntGetZExtValue(value).min(256) as u16;
                }
            }
            if ok {
                let valid = match kind {
                    // register k of n
                    Kind::Ldn | Kind::Stn => {
                        (2..=30).contains(&imms[1]) && imms[0] < imms[1]
                    }
                    _ => op.imm.accepts(u64::from(imms[0])),
                };
                if !valid {
                    let value = if kind.imm_args() == 2 {
                        format!("{} of {}", imms[0], imms[1])
                    } else {
                        format!("{:#x}", imms[0])
                    };
                    self.error(
                        Some(f),
                        format!(
                            "immediate {value} is out of range for `{}`",
                            op.name
                        ),
                    );
                    ok = false;
                }
            }
            ok.then_some((uses, scalar, mask, imms.map(|imm| imm as u8)))
        }
    }

    /// What makes a marker call a candidate member of a load or store run.
    unsafe fn run_key(
        &self,
        op: &OpInfo,
        mode: u8,
        scalar: Option<LLVMValueRef>,
        imms: [u8; 2],
    ) -> Option<RunKey> {
        let address = scalar?;
        match op.kind {
            Kind::Ldn | Kind::Stn => Some(RunKey {
                store: op.kind == Kind::Stn,
                base: address,
                offset: i64::from(imms[0]) * 64,
                count: imms[1],
            }),
            // Only loads (stores) from one allocated object: fused, they
            // need the whole span in one region.
            _ if mode == 0 && (op.code == LD || op.code == ST) => {
                let (base, offset) = unsafe { self.inbounds_offset(address)? };
                Some(RunKey { store: op.code == ST, base, offset, count: 0 })
            }
            _ => None,
        }
    }

    /// `address` as `base + offset` through inbounds GEPs with constant
    /// indices.
    unsafe fn inbounds_offset(
        &self,
        mut address: LLVMValueRef,
    ) -> Option<(LLVMValueRef, i64)> {
        unsafe {
            let mut offset = 0i64;
            loop {
                let opcode = if !LLVMIsAInstruction(address).is_null() {
                    LLVMGetInstructionOpcode(address)
                } else if !LLVMIsAConstantExpr(address).is_null() {
                    LLVMGetConstOpcode(address)
                } else {
                    break;
                };
                if opcode != LLVMOpcode::LLVMGetElementPtr
                    || LLVMIsInBounds(address) == 0
                {
                    break;
                }
                let Some((base, step)) = self.constant_step(address) else {
                    break;
                };
                offset = offset.checked_add(step)?;
                address = base;
            }
            Some((address, offset))
        }
    }

    /// Returns why `user` lets the token escape, if it does.
    unsafe fn escape(
        &self,
        user: LLVMValueRef,
        token: LLVMValueRef,
    ) -> Option<String> {
        unsafe {
            if !LLVMIsACallInst(user).is_null() {
                let callee = LLVMGetCalledValue(user);
                if let Some(&(op, mode)) = self.markers.get(&callee) {
                    let args = LLVMGetNumArgOperands(user) as usize;
                    let vectors = op.vector_args(mode);
                    let vector_arg = (0..args.min(vectors))
                        .any(|k| LLVMGetOperand(user, k as u32) == token);
                    let other_arg = (vectors..args)
                        .any(|k| LLVMGetOperand(user, k as u32) == token);
                    return (!vector_arg || other_arg).then(|| {
                        format!(
                            "is used as a scalar argument of `{}`",
                            value_name(callee)
                        )
                    });
                }
                let name = if LLVMIsAFunction(callee).is_null() {
                    "an indirect call".to_string()
                } else {
                    format!("`{}`", function_name(callee))
                };
                return Some(format!(
                    "is passed to {name}; vectors cannot cross function boundaries \
                     (mark the function #[inline(always)])"
                ));
            }
            let what = match LLVMGetInstructionOpcode(user) {
                LLVMOpcode::LLVMStore => "is stored to memory (e.g. kept in an array, a struct \
                                          behind a reference, or a `static`)"
                    .to_string(),
                LLVMOpcode::LLVMRet => "is returned from the function; vectors cannot cross \
                                        function boundaries (mark it #[inline(always)])"
                    .to_string(),
                _ => {
                    let printed = LLVMPrintValueToString(user);
                    let text = CStr::from_ptr(printed).to_string_lossy().trim().to_string();
                    LLVMDisposeMessage(printed);
                    format!("is used by a non-vector instruction `{text}`")
                }
            };
            Some(what)
        }
    }

    unsafe fn emit(
        &mut self,
        builder: LLVMBuilderRef,
        f: LLVMValueRef,
        function: &Function,
        allocation: &Allocation,
    ) {
        unsafe {
            let reg = |t: Option<Tok>| {
                t.map_or(0, |t| allocation.reg[t as usize].unwrap_or(0))
            };
            // 64-byte stack slots for spilled values.
            let slot_ty =
                LLVMArrayType2(LLVMInt8TypeInContext(self.context), 64);
            LLVMPositionBuilderBefore(
                builder,
                LLVMGetFirstInstruction(function.blocks[0]),
            );
            let slots: Vec<LLVMValueRef> = (0..allocation.slots)
                .map(|_| {
                    let slot = LLVMBuildAlloca(
                        builder,
                        slot_ty,
                        c"svm.spill".as_ptr(),
                    );
                    LLVMSetAlignment(slot, 64);
                    slot
                })
                .collect();
            let op_info = |name| ops::by_marker(name).unwrap().0;
            let (ld, st) = (op_info("__svm_ld"), op_info("__svm_st"));
            let (ldn, stn) = (op_info("__svm_ldn"), op_info("__svm_stn"));
            for (b, (block, sites)) in
                function.func.blocks.iter().zip(&function.sites).enumerate()
            {
                let fused = fuse_runs(&block.insts, &allocation.reg);
                for (i, inst) in block.insts.iter().enumerate() {
                    let site = &sites[inst.origin];
                    match (inst.kind, site) {
                        (InstKind::Spill { slot }, _) => {
                            LLVMPositionBuilderBefore(builder, site.anchor());
                            let word = ops::encode(
                                st.code,
                                0,
                                0,
                                reg(inst.uses[0]),
                                0,
                            );
                            self.emit_op(
                                builder,
                                st,
                                word,
                                Some(slots[slot as usize]),
                                None,
                                site.anchor(),
                            );
                        }
                        (InstKind::Reload { slot }, _) => {
                            LLVMPositionBuilderBefore(builder, site.anchor());
                            let word =
                                ops::encode(ld.code, 0, reg(inst.def), 0, 0);
                            self.emit_op(
                                builder,
                                ld,
                                word,
                                Some(slots[slot as usize]),
                                None,
                                site.anchor(),
                            );
                        }
                        (
                            _,
                            &Site::Op { call, op, mode, scalar, mask, imm8 },
                        ) => {
                            LLVMPositionBuilderBefore(builder, call);
                            if let Some((dst, src)) = allocation.pre_copy[b][i]
                            {
                                self.emit_words(builder, &[mov(dst, src)]);
                            }
                            let scalar =
                                scalar.map(|k| LLVMGetOperand(call, k));
                            let mask = mask.map(|k| LLVMGetOperand(call, k));
                            let out = reg(inst.def);
                            let first = usize::from(op.tied(mode));
                            let in0 =
                                inst.uses.get(first).map_or(0, |&u| reg(u));
                            let in1 = inst
                                .uses
                                .get(first + 1)
                                .map_or(0, |&u| reg(u));
                            // A run of consecutive registers is one `ldn` or
                            // `stn` at its first member; the registers of an
                            // unfused `ldn` or `stn` are single `ld`s / `st`s.
                            let member =
                                matches!(op.kind, Kind::Ldn | Kind::Stn)
                                    .then_some(imm8);
                            let (op, code, imm8, mut scalar) = match fused[i] {
                                0 => continue,
                                1 if op.kind == Kind::Ldn => {
                                    (ld, LD, 0, scalar)
                                }
                                1 if op.kind == Kind::Stn => {
                                    (st, ST, 0, scalar)
                                }
                                1 => (op, op.code, imm8, scalar),
                                n if inst.def.is_some() => {
                                    (ldn, LDN, n, scalar)
                                }
                                n => (stn, STN, n, scalar),
                            };
                            if let (Some(p), Some(k @ 1..)) = (scalar, member)
                            {
                                // register k of an `ldn` / `stn`: p + 64k
                                let mut index = [LLVMConstInt(
                                    LLVMInt64TypeInContext(self.context),
                                    u64::from(k) * 64,
                                    0,
                                )];
                                scalar = Some(LLVMBuildInBoundsGEP2(
                                    builder,
                                    LLVMInt8TypeInContext(self.context),
                                    p,
                                    index.as_mut_ptr(),
                                    1,
                                    c"".as_ptr(),
                                ));
                            }
                            let (scalar, off) = match scalar {
                                Some(p) if op.is_memory() => {
                                    let (base, off) = self.split_address(p);
                                    (Some(base), off)
                                }
                                _ => (scalar, 0),
                            };
                            let word = ops::with_mode(
                                ops::with_off(
                                    ops::encode(code, imm8, out, in0, in1),
                                    off,
                                ),
                                mode,
                            );
                            let result = self.emit_op(
                                builder, op, word, scalar, mask, call,
                            );
                            if op.kind.defines_scalar() {
                                LLVMReplaceAllUsesWith(call, result);
                            }
                        }
                        (_, &Site::Select(select)) => {
                            LLVMPositionBuilderBefore(builder, select);
                            let (a, b) = (inst.uses[0], inst.uses[1]);
                            self.emit_select(
                                builder,
                                select,
                                reg(inst.def),
                                a.map(|_| reg(a)),
                                b.map(|_| reg(b)),
                            );
                        }
                        (_, Site::Call(..)) => {}
                    }
                }
            }
            for (pred, succ, moves) in &allocation.edge_moves {
                let at = if function.preds[*succ] == 1 {
                    instructions(function.blocks[*succ])
                        .find(|&i| LLVMIsAPHINode(i).is_null())
                        .unwrap()
                } else if function.func.blocks[*pred].succs.len() == 1 {
                    LLVMGetBasicBlockTerminator(function.blocks[*pred])
                } else {
                    self.error(
                        Some(f),
                        "internal error: phi copies on a critical edge".into(),
                    );
                    continue;
                };
                LLVMPositionBuilderBefore(builder, at);
                let words: Vec<u64> = moves
                    .iter()
                    .flat_map(|m| match *m {
                        Move::Mov { dst, src } => vec![mov(dst, src)],
                        Move::Swap(a, b) => vec![
                            ops::encode(XOR, 0, a, a, b),
                            ops::encode(XOR, 0, b, a, b),
                            ops::encode(XOR, 0, a, a, b),
                        ],
                    })
                    .collect();
                self.emit_words(builder, &words);
            }
            let i32_ty = LLVMInt32TypeInContext(self.context);
            for &token in &function.tokens {
                LLVMReplaceAllUsesWith(token, LLVMGetPoison(i32_ty));
            }
            for &token in &function.tokens {
                LLVMInstructionEraseFromParent(token);
            }
            for sites in &function.sites {
                for site in sites {
                    if let Site::Op { call, op, .. } = *site
                        && !op.kind.defines_vector()
                    {
                        LLVMInstructionEraseFromParent(call);
                    }
                }
            }
        }
    }

    /// Emits `asm sideeffect` with the given text and constraints.
    unsafe fn emit_asm(
        &self,
        builder: LLVMBuilderRef,
        text: &str,
        constraints: &str,
        ret: LLVMTypeRef,
        args: &[LLVMValueRef],
    ) -> LLVMValueRef {
        unsafe {
            let mut arg_types: Vec<LLVMTypeRef> =
                args.iter().map(|&a| LLVMTypeOf(a)).collect();
            let fn_ty = LLVMFunctionType(
                ret,
                arg_types.as_mut_ptr(),
                args.len() as u32,
                0,
            );
            let asm = LLVMGetInlineAsm(
                fn_ty,
                text.as_ptr().cast::<c_char>(),
                text.len(),
                constraints.as_ptr().cast::<c_char>(),
                constraints.len(),
                1,
                0,
                LLVMInlineAsmDialect::LLVMInlineAsmDialectATT,
                0,
            );
            let mut args = args.to_vec();
            LLVMBuildCall2(
                builder,
                fn_ty,
                asm,
                args.as_mut_ptr(),
                args.len() as u32,
                c"".as_ptr(),
            )
        }
    }

    /// Emits ops that touch no scalar register. `r0` is declared clobbered
    /// because sbpf-linker's bytecode optimizer models every `call` as
    /// defining `r0`.
    unsafe fn emit_words(&self, builder: LLVMBuilderRef, words: &[u64]) {
        let text = quads(words);
        let void = unsafe { LLVMVoidTypeInContext(self.context) };
        unsafe { self.emit_asm(builder, &text, "~{r0}", void, &[]) };
    }

    unsafe fn emit_op(
        &self,
        builder: LLVMBuilderRef,
        op: &OpInfo,
        word: u64,
        scalar: Option<LLVMValueRef>,
        mask: Option<LLVMValueRef>,
        call: LLVMValueRef,
    ) -> LLVMValueRef {
        unsafe {
            let text = quads(&[word]);
            let mut constraints = Vec::new();
            let ret = if op.kind.defines_scalar() {
                constraints.push("={r0}");
                LLVMTypeOf(call)
            } else {
                LLVMVoidTypeInContext(self.context)
            };
            if scalar.is_some() {
                constraints.push("{r1}");
            }
            if mask.is_some() {
                constraints.push("{r2}");
            }
            if !op.kind.defines_scalar() {
                constraints.push("~{r0}");
            }
            if op.is_memory() {
                constraints.push("~{memory}");
            }
            let args: Vec<LLVMValueRef> =
                scalar.into_iter().chain(mask).collect();
            self.emit_asm(builder, &text, &constraints.join(","), ret, &args)
        }
    }

    /// Splits a memory op's address into a base and a constant displacement
    /// for the op's 16-bit `off` field, looking through constant GEPs, adds
    /// and int/pointer casts. Sharing the base lets LLVM keep one pointer in
    /// `r1` instead of computing every address.
    unsafe fn split_address(
        &self,
        mut addr: LLVMValueRef,
    ) -> (LLVMValueRef, i16) {
        unsafe {
            let mut off = 0i64;
            while let Some((base, step)) = self.constant_step(addr) {
                match off.checked_add(step) {
                    Some(total) if i16::try_from(total).is_ok() => {
                        off = total;
                        addr = base;
                    }
                    _ => break,
                }
            }
            (addr, off as i16)
        }
    }

    /// `value` as `base + constant`, for one GEP, add or cast.
    unsafe fn constant_step(
        &self,
        value: LLVMValueRef,
    ) -> Option<(LLVMValueRef, i64)> {
        unsafe {
            let opcode = if !LLVMIsAInstruction(value).is_null() {
                LLVMGetInstructionOpcode(value)
            } else if !LLVMIsAConstantExpr(value).is_null() {
                LLVMGetConstOpcode(value)
            } else {
                return None;
            };
            let constant = |v: LLVMValueRef| {
                (!LLVMIsAConstantInt(v).is_null())
                    .then(|| LLVMConstIntGetSExtValue(v))
            };
            let operand = |k| LLVMGetOperand(value, k);
            match opcode {
                LLVMOpcode::LLVMGetElementPtr
                    if LLVMGetNumOperands(value) == 2 =>
                {
                    let size = LLVMABISizeOfType(
                        LLVMGetModuleDataLayout(self.module),
                        LLVMGetGEPSourceElementType(value),
                    );
                    let step = constant(operand(1))?
                        .checked_mul(i64::try_from(size).ok()?)?;
                    Some((operand(0), step))
                }
                LLVMOpcode::LLVMAdd => {
                    if let Some(c) = constant(operand(1)) {
                        Some((operand(0), c))
                    } else {
                        Some((operand(1), constant(operand(0))?))
                    }
                }
                LLVMOpcode::LLVMIntToPtr | LLVMOpcode::LLVMPtrToInt => {
                    Some((operand(0), 0))
                }
                _ => None,
            }
        }
    }

    /// `out ← cond ? a : b` as a conditional move inside one asm block.
    unsafe fn emit_select(
        &self,
        builder: LLVMBuilderRef,
        select: LLVMValueRef,
        out: Reg,
        a: Option<Reg>,
        b: Option<Reg>,
    ) {
        unsafe {
            let text = match (a, b) {
                (None, None) => return,
                (Some(r), None) | (None, Some(r)) => {
                    if r != out {
                        self.emit_words(builder, &[mov(out, r)]);
                    }
                    return;
                }
                (Some(a), Some(b)) if a == b => {
                    if a != out {
                        self.emit_words(builder, &[mov(out, a)]);
                    }
                    return;
                }
                (Some(a), Some(b)) if a == out => {
                    format!("if $0 != 0 goto +1\n{}", quads(&[mov(out, b)]))
                }
                (Some(a), Some(b)) if b == out => {
                    format!("if $0 == 0 goto +1\n{}", quads(&[mov(out, a)]))
                }
                (Some(a), Some(b)) => format!(
                    "{}\nif $0 != 0 goto +1\n{}",
                    quads(&[mov(out, a)]),
                    quads(&[mov(out, b)])
                ),
            };
            let i64_ty = LLVMInt64TypeInContext(self.context);
            let cond = LLVMBuildZExt(
                builder,
                LLVMGetOperand(select, 0),
                i64_ty,
                c"".as_ptr(),
            );
            let void = LLVMVoidTypeInContext(self.context);
            self.emit_asm(builder, &text, "r,~{r0}", void, &[cond]);
        }
    }
}

/// Per allocator instruction, how many run members it emits: 2 or more
/// for the first member of consecutive registers (one `ldn` / `stn`), 0 for
/// the others, 1 for everything else.
fn fuse_runs(insts: &[Inst], reg: &[Option<Reg>]) -> Vec<u8> {
    let run_reg = |inst: &Inst| {
        let run = inst.run?;
        let t = if run.store { (*inst.uses.first()?)? } else { inst.def? };
        Some((run, reg[t as usize]?))
    };
    let mut fused = vec![1; insts.len()];
    let mut i = 0;
    while i < insts.len() {
        let mut j = i + 1;
        if let Some((first, base)) = run_reg(&insts[i]) {
            while j < insts.len()
                && run_reg(&insts[j]).is_some_and(|(run, r)| {
                    let n = (j - i) as u8;
                    run.id == first.id && run.k == first.k + n && r == base + n
                })
            {
                j += 1;
            }
        }
        if j - i >= 2 {
            fused[i] = (j - i) as u8;
            fused[i + 1..j].fill(0);
        }
        i = j;
    }
    fused
}

fn mov(dst: Reg, src: Reg) -> u64 {
    ops::encode(MOV, 0, dst, src, 0)
}

fn quads(words: &[u64]) -> String {
    words
        .iter()
        .map(|w| format!(".quad {w:#018x}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Defined functions, callees before callers (cycles broken arbitrarily).
fn post_order(
    defined: &[LLVMValueRef],
    callees: &HashMap<LLVMValueRef, Vec<Callee>>,
) -> Vec<LLVMValueRef> {
    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &root in defined {
        if !seen.insert(root) {
            continue;
        }
        let mut stack = vec![(root, 0)];
        while let Some((f, next)) = stack.pop() {
            let calls = &callees[&f];
            if let Some(callee) = calls.get(next) {
                stack.push((f, next + 1));
                if let Callee::Defined(g) = *callee
                    && callees.contains_key(&g)
                    && seen.insert(g)
                {
                    stack.push((g, 0));
                }
            } else {
                order.push(f);
            }
        }
    }
    order
}

/// Parses textual IR, for tests.
#[cfg(test)]
#[allow(deprecated)]
pub(crate) unsafe fn parse_ir(
    context: LLVMContextRef,
    ir: &str,
) -> LLVMModuleRef {
    unsafe {
        let ir = std::ffi::CString::new(ir).unwrap();
        let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            ir.as_ptr(),
            ir.as_bytes().len(),
            c"test.ll".as_ptr(),
        );
        let mut module = ptr::null_mut();
        let mut message = ptr::null_mut();
        let failed = bpf_linker::llvm_sys::ir_reader::LLVMParseIRInContext(
            context,
            buffer,
            &mut module,
            &mut message,
        );
        assert_eq!(failed, 0, "{}", CStr::from_ptr(message).to_string_lossy());
        module
    }
}

#[cfg(test)]
mod tests;
