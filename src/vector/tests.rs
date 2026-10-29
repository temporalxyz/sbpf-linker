//! IR-level tests: parse textual IR, run the pass, and decode the emitted
//! `.quad` words. Every successful allocation is also checked by the
//! allocator's own register simulation.

use std::ffi::CStr;

use bpf_linker::llvm_sys::core::*;

use super::{Transport, lower_module, lower_module_with, parse_ir};

const DECLS: &str = r#"
target triple = "bpfel"
declare i32 @__svm_ld(ptr)
declare void @__svm_st(i32, ptr)
declare i32 @__svm_zero()
declare i32 @__svm_splat_d(i64)
declare i32 @__svm_add_d(i32, i32)
declare i32 @__svm_sub_d(i32, i32)
declare i32 @__svm_dpbusd(i32, i32, i32)
declare i32 @__svm_ternlog(i32, i32, i32, i32)
declare i32 @__svm_shli_d(i32, i32)
declare i64 @__svm_hsum_d(i32)
declare i64 @__svm_extract_q(i32, i32)
declare i32 @__svm_insert_q(i32, i64, i32)
declare i32 @__svm_clmul(i32, i32, i32)
declare i32 @__svm_add_d_mask(i32, i32, i32, i64)
declare i32 @__svm_add_d_maskz(i32, i32, i64)
declare i32 @__svm_dpbusd_mask(i32, i32, i32, i64)
declare i32 @__svm_blendm(i32, i32, i64, i32)
declare i64 @__svm_cmp(i32, i32, i32)
declare i64 @__svm_mpopcnt(i64)
declare i32 @__svm_ld_m_maskz(ptr, i64, i32)
declare void @__svm_st_m_mask(i32, ptr, i64, i32)
declare i32 @__svm_gather_qq_mask(i32, i32, ptr, i64, i32)
declare void @__svm_scatter_qq(i32, i32, ptr, i32)
declare i32 @__svm_sha256rnds2(i32, i32, i32)
declare i32 @__svm_ldn(ptr, i32, i32)
declare void @__svm_stn(i32, ptr, i32, i32)
declare void @sink(i32)
"#;

/// A decoded vector op: `(op, imm8, out, in0, in1)`.
type Word = (u8, u8, u8, u8, u8);

struct Lowered {
    ir: String,
    /// Decoded ops in the order they appear in the printed module.
    words: Vec<Word>,
}

/// Lowers under `src=2` and checks that the `sol_avx512` transports emit
/// the same ops as descriptors (selects become `blendm` there instead).
fn lower(body: &str) -> Result<Lowered, String> {
    let lowered = lower_as(body, Transport::Src2);
    if let Ok(l) = &lowered
        && !body.contains(" select ")
    {
        let expected: Vec<[u8; 16]> =
            words(&l.ir).into_iter().map(super::ops::descriptor).collect();
        for transport in [Transport::Call, Transport::Asm] {
            let other = lower_as(body, transport).unwrap();
            assert!(!other.ir.contains(".quad"), "{}", other.ir);
            assert_eq!(descriptors(&other.ir), expected, "{transport:?}");
        }
    }
    lowered
}

/// Every emitted `.quad` word, in order.
fn words(ir: &str) -> Vec<u64> {
    ir.split(".quad 0x")
        .skip(1)
        .map(|s| u64::from_str_radix(&s[..16], 16).unwrap())
        .collect()
}

/// The descriptor of every `sol_avx512` call, in order.
fn descriptors(ir: &str) -> Vec<[u8; 16]> {
    let mut globals = std::collections::HashMap::new();
    for line in ir.lines() {
        let Some((name, rest)) =
            line.split_once(" = private unnamed_addr constant [16 x i8] ")
        else {
            continue;
        };
        assert!(rest.contains("section \".rodata.avx512\""), "{line}");
        if rest.starts_with("zeroinitializer") {
            globals.insert(name.trim().to_owned(), [0; 16]);
        } else if let Some(rest) = rest.strip_prefix("c\"") {
            let text = rest.split('"').next().unwrap().as_bytes();
            let mut bytes = Vec::new();
            let mut i = 0;
            while i < text.len() {
                if text[i] == b'\\' {
                    let hex =
                        std::str::from_utf8(&text[i + 1..i + 3]).unwrap();
                    bytes.push(u8::from_str_radix(hex, 16).unwrap());
                    i += 3;
                } else {
                    bytes.push(text[i]);
                    i += 1;
                }
            }
            globals.insert(
                name.trim().to_owned(),
                <[u8; 16]>::try_from(bytes).unwrap(),
            );
        }
    }
    ir.lines()
        .filter(|line| line.contains("sol_avx512"))
        .filter_map(|line| line.split_once("(ptr @"))
        .map(|(_, rest)| {
            let name = rest.split([',', ')']).next().unwrap();
            globals[&format!("@{name}")]
        })
        .collect()
}

fn lower_as(body: &str, transport: Transport) -> Result<Lowered, String> {
    unsafe {
        let context = LLVMContextCreate();
        let module = parse_ir(context, &format!("{DECLS}\n{body}"));
        let result = lower_module_with(module, transport)
            .map_err(|e| e.to_string())
            .map(|_| {
                let printed = LLVMPrintModuleToString(module);
                let ir =
                    CStr::from_ptr(printed).to_string_lossy().into_owned();
                LLVMDisposeMessage(printed);
                let words = decode(&ir);
                Lowered { ir, words }
            });
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
        result
    }
}

fn decode(ir: &str) -> Vec<Word> {
    ir.split(".quad 0x")
        .skip(1)
        .map(|s| {
            let word = u64::from_str_radix(&s[..16], 16).unwrap();
            assert_eq!(word & 0xf0ff, 0x2085, "CALL_IMM src=2");
            let imm = (word >> 32) as u32;
            assert_eq!(imm & 0x8000, 0);
            (
                (imm >> 24) as u8,
                (imm >> 16) as u8,
                (imm >> 10 & 31) as u8,
                (imm >> 5 & 31) as u8,
                (imm & 31) as u8,
            )
        })
        .collect()
}

/// The `off` field of every emitted op, in order.
fn offsets(ir: &str) -> Vec<i16> {
    ir.split(".quad 0x")
        .skip(1)
        .map(|s| u64::from_str_radix(&s[..16], 16).unwrap() as u32 >> 16)
        .map(|off| off as u16 as i16)
        .collect()
}

/// The mask mode (`dst`) of every emitted op, in order.
fn modes(ir: &str) -> Vec<u8> {
    ir.split(".quad 0x")
        .skip(1)
        .map(|s| (u64::from_str_radix(&s[..16], 16).unwrap() >> 8 & 0xf) as u8)
        .collect()
}

fn ops(words: &[Word]) -> Vec<u8> {
    words.iter().map(|w| w.0).collect()
}

#[test]
fn no_markers_is_a_no_op() {
    unsafe {
        let context = LLVMContextCreate();
        let module =
            parse_ir(context, "define i64 @f(i64 %x) {\n  ret i64 %x\n}\n");
        assert!(!lower_module(module).unwrap());
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
    }
}

#[test]
fn straight_line() {
    let l = lower(
        r#"
define void @f(ptr %p, ptr %q) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %q)
  %c = call i32 @__svm_add_d(i32 %a, i32 %b)
  call void @__svm_st(i32 %c, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert!(!l.ir.contains("__svm_"), "{}", l.ir);
    assert_eq!(
        l.words,
        [
            (0x00, 0, 0, 0, 0),
            (0x00, 0, 1, 0, 0),
            (0x12, 0, 0, 0, 1),
            (0x01, 0, 0, 0, 0)
        ]
    );
    // ld/st take the address in r1 and clobber memory; add touches no
    // scalar register except the declared r0 clobber.
    assert!(l.ir.contains(r#""{r1},~{r0},~{memory}"(ptr %p)"#), "{}", l.ir);
    assert!(l.ir.contains(r#""{r1},~{r0},~{memory}"(ptr %q)"#), "{}", l.ir);
    assert!(
        l.ir.contains(
            r#"asm sideeffect ".quad 0x1200000100002085", "~{r0}"()"#
        ),
        "{}",
        l.ir
    );
}

#[test]
fn constant_displacements_fold_into_off() {
    let l = lower(
        r#"
@g = external global [256 x i8]

define void @f(ptr %p, i64 %x) {
  %p64 = getelementptr inbounds i8, ptr %p, i64 64
  %a = call i32 @__svm_ld(ptr %p64)
  %q = getelementptr inbounds i32, ptr %p, i64 -4
  %q2 = getelementptr i8, ptr %q, i64 1
  %b = call i32 @__svm_ld(ptr %q2)
  %i = ptrtoint ptr %p to i64
  %j = add i64 %i, 32767
  %r = inttoptr i64 %j to ptr
  %c = call i32 @__svm_ld(ptr %r)
  %far = getelementptr i8, ptr %p, i64 32768
  %d = call i32 @__svm_ld(ptr %far)
  %e = call i32 @__svm_ld(ptr getelementptr (i8, ptr @g, i64 128))
  %y = add i64 %x, 8
  %s = call i32 @__svm_splat_d(i64 %y)
  %t = call i32 @__svm_add_d(i32 %a, i32 %b)
  %u = call i32 @__svm_add_d(i32 %c, i32 %d)
  %v = call i32 @__svm_add_d(i32 %e, i32 %s)
  %w = call i32 @__svm_add_d(i32 %t, i32 %u)
  %z = call i32 @__svm_add_d(i32 %w, i32 %v)
  call void @__svm_st(i32 %z, ptr %p64)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(&offsets(&l.ir)[..6], [64, -15, 32767, 0, 128, 0]);
    assert_eq!(offsets(&l.ir)[11], 64);
    assert_eq!(l.ir.matches("(ptr %p)").count(), 4, "{}", l.ir);
    // out of i16 range: the address is passed as is
    assert!(l.ir.contains("(ptr %far)"), "{}", l.ir);
    assert!(l.ir.contains("(ptr @g)"), "{}", l.ir);
    // only memory ops fold: splat gets the added value
    assert!(l.ir.contains("(i64 %y)"), "{}", l.ir);
}

#[test]
fn loop_carried_accumulator() {
    let l = lower(
        r#"
define i64 @f(ptr %p, i64 %n) {
entry:
  %z = call i32 @__svm_zero()
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi i32 [ %z, %entry ], [ %acc1, %loop ]
  %x = call i32 @__svm_ld(ptr %p)
  %acc1 = call i32 @__svm_dpbusd(i32 %acc, i32 %x, i32 %x)
  %i1 = add i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %s = call i64 @__svm_hsum_d(i32 %acc1)
  ret i64 %s
}
"#,
    )
    .unwrap();
    // No moves: the phi, the accumulator and its update share v0.
    assert_eq!(ops(&l.words), [0x08, 0x00, 0x50, 0x70]);
    assert_eq!(l.words[0].2, 0);
    assert_eq!(l.words[2], (0x50, 0, 0, 1, 1));
    assert_eq!(l.words[3], (0x70, 0, 0, 0, 0));
    assert!(l.ir.contains(
        r#"call i64 asm sideeffect ".quad 0x7000000000002085", "={r0}"()"#
    ));
}

#[test]
fn loop_carried_non_accumulating_op() {
    // acc = acc + x: the add is not tied, but phi affinity still avoids moves.
    let l = lower(
        r#"
define i64 @f(ptr %p, i64 %n) {
entry:
  %z = call i32 @__svm_zero()
  %k = call i32 @__svm_splat_d(i64 7)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi i32 [ %z, %entry ], [ %acc1, %loop ]
  %x = call i32 @__svm_ld(ptr %p)
  %y = call i32 @__svm_add_d(i32 %x, i32 %k)
  %acc1 = call i32 @__svm_add_d(i32 %y, i32 %acc)
  %i1 = add i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  %s = call i64 @__svm_hsum_d(i32 %acc1)
  ret i64 %s
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x08, 0x06, 0x00, 0x12, 0x12, 0x70]);
    let acc = l.words[0].2;
    assert_eq!(l.words[4].2, acc, "{:?}", l.words);
    assert_ne!(l.words[3].2, acc);
}

#[test]
fn nested_loops() {
    let l = lower(
        r#"
define i64 @f(ptr %p, i64 %n) {
entry:
  %z = call i32 @__svm_zero()
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i1, %outer.latch ]
  %acc = phi i32 [ %z, %entry ], [ %acc2, %outer.latch ]
  %iz = call i32 @__svm_zero()
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j1, %inner ]
  %in = phi i32 [ %iz, %outer ], [ %in1, %inner ]
  %x = call i32 @__svm_ld(ptr %p)
  %in1 = call i32 @__svm_dpbusd(i32 %in, i32 %x, i32 %x)
  %j1 = add i64 %j, 1
  %cj = icmp ult i64 %j1, %n
  br i1 %cj, label %inner, label %outer.latch
outer.latch:
  %acc2 = call i32 @__svm_add_d(i32 %acc, i32 %in1)
  %i1 = add i64 %i, 1
  %ci = icmp ult i64 %i1, %n
  br i1 %ci, label %outer, label %exit
exit:
  %s = call i64 @__svm_hsum_d(i32 %acc2)
  ret i64 %s
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x08, 0x08, 0x00, 0x50, 0x12, 0x70]);
    let (acc, inner) = (l.words[0].2, l.words[1].2);
    assert_ne!(acc, inner);
    assert_eq!(l.words[3].2, inner);
    assert_eq!(l.words[4], (0x12, 0, acc, acc, inner));
}

#[test]
fn if_else_merge() {
    let l = lower(
        r#"
define void @f(ptr %p, i1 %c) {
entry:
  %a = call i32 @__svm_ld(ptr %p)
  br i1 %c, label %then, label %else
then:
  %t = call i32 @__svm_add_d(i32 %a, i32 %a)
  br label %merge
else:
  %e = call i32 @__svm_sub_d(i32 %a, i32 %a)
  br label %merge
merge:
  %m = phi i32 [ %t, %then ], [ %e, %else ]
  %r = call i32 @__svm_add_d(i32 %m, i32 %a)
  call void @__svm_st(i32 %r, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    // Both arms write the phi's register directly; no moves needed.
    assert_eq!(ops(&l.words), [0x00, 0x12, 0x16, 0x12, 0x01]);
    assert_eq!(l.words[1].2, l.words[2].2);
    assert_eq!(
        l.words[3],
        (0x12, 0, l.words[3].2, l.words[1].2, l.words[0].2)
    );
}

#[test]
fn phi_on_critical_edge_gets_split() {
    // %entry -> %merge is critical, and the phi needs a copy on it since %a
    // stays live in %merge.
    let l = lower(
        r#"
define void @f(ptr %p, i1 %c) {
entry:
  %a = call i32 @__svm_ld(ptr %p)
  br i1 %c, label %then, label %merge
then:
  %t = call i32 @__svm_add_d(i32 %a, i32 %a)
  br label %merge
merge:
  %m = phi i32 [ %t, %then ], [ %a, %entry ]
  %r = call i32 @__svm_add_d(i32 %m, i32 %a)
  call void @__svm_st(i32 %r, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x00, 0x09, 0x12, 0x12, 0x01], "{}", l.ir);
    assert!(l.ir.contains("entry.merge_crit_edge:"), "{}", l.ir);
    // The copy writes the phi's register, which the `then` arm defines.
    assert_eq!(l.words[1].2, l.words[2].2);
    assert_eq!(l.words[1].3, l.words[0].2);
}

#[test]
fn accumulator_live_after_is_copied() {
    let l = lower(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %d = call i32 @__svm_dpbusd(i32 %a, i32 %b, i32 %b)
  call void @__svm_st(i32 %a, ptr %p)
  call void @__svm_st(i32 %d, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x00, 0x00, 0x09, 0x50, 0x01, 0x01]);
    let (a, b) = (l.words[0].2, l.words[1].2);
    let d = l.words[3].2;
    assert_eq!(l.words[2], (0x09, 0, d, a, 0));
    assert_eq!(l.words[3], (0x50, 0, d, b, b));
    assert!(d != a && d != b);
    assert_eq!(l.words[4].3, a);
    assert_eq!(l.words[5].3, d);
}

#[test]
fn accumulator_dying_is_reused() {
    let l = lower(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %c = call i32 @__svm_ld(ptr %p)
  %t = call i32 @__svm_ternlog(i32 %b, i32 %a, i32 %c, i32 202)
  call void @__svm_st(i32 %t, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(l.words[3], (0x2c, 202, 1, 0, 2));
}

#[test]
fn immediates_and_scalars() {
    let l = lower(
        r#"
define i64 @f(i64 %x) {
  %a = call i32 @__svm_splat_d(i64 %x)
  %b = call i32 @__svm_shli_d(i32 %a, i32 5)
  %c = call i32 @__svm_insert_q(i32 %b, i64 %x, i32 3)
  %d = call i32 @__svm_clmul(i32 %c, i32 %b, i32 17)
  %e = call i64 @__svm_extract_q(i32 %d, i32 7)
  %r = add i64 %e, %x
  ret i64 %r
}
"#,
    )
    .unwrap();
    assert_eq!(
        l.words,
        [
            (0x06, 0, 0, 0, 0),
            (0x31, 5, 0, 0, 0),
            (0x09, 0, 1, 0, 0),
            (0x74, 3, 1, 0, 0),
            (0x78, 0x11, 0, 1, 0),
            (0x72, 7, 0, 0, 0)
        ]
    );
    // splat and insert read r1 but not memory; extract defines r0.
    assert!(l.ir.contains(r#""{r1},~{r0}"(i64 %x)"#), "{}", l.ir);
    assert!(l.ir.contains(
        r#"= call i64 asm sideeffect ".quad 0x7207000000002085", "={r0}"()"#
    ));
    assert!(!l.ir.contains("~{memory}"));
}

#[test]
fn select_becomes_conditional_move() {
    let l = lower(
        r#"
define void @f(ptr %p, i1 %c) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %s = select i1 %c, i32 %a, i32 %b
  call void @__svm_st(i32 %s, ptr %p)
  call void @__svm_st(i32 %a, ptr %p)
  call void @__svm_st(i32 %b, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x00, 0x00, 0x09, 0x09, 0x01, 0x01, 0x01]);
    assert_eq!(l.words[2], (0x09, 0, 2, 0, 0));
    assert_eq!(l.words[3], (0x09, 0, 2, 1, 0));
    assert!(l.ir.contains("if $0 != 0 goto +1"), "{}", l.ir);
}

#[test]
fn values_live_across_calls_avoid_callee_registers() {
    let l = lower(
        r#"
define internal void @helper(ptr %p) noinline {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_add_d(i32 %a, i32 %a)
  call void @__svm_st(i32 %b, ptr %p)
  ret void
}

define void @f(ptr %p) {
  %x = call i32 @__svm_ld(ptr %p)
  call void @helper(ptr %p)
  call void @__svm_st(i32 %x, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    // helper uses v0; %x must not live in it.
    assert_eq!(
        l.words[..3],
        [(0x00, 0, 0, 0, 0), (0x12, 0, 0, 0, 0), (0x01, 0, 0, 0, 0)]
    );
    assert_eq!(l.words[3], (0x00, 0, 1, 0, 0));
}

#[test]
fn values_live_across_memory_intrinsics_avoid_libcall_registers() {
    // Codegen lowers the intrinsics to calls to the program's `memcpy` and
    // `memset`, which use v0.
    let l = lower(
        r#"
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define ptr @memcpy(ptr %d, ptr %s, i64 %n) {
  %a = call i32 @__svm_ld(ptr %s)
  call void @__svm_st(i32 %a, ptr %d)
  ret ptr %d
}

define ptr @memset(ptr %d, i32 %c, i64 %n) {
  %a = call i32 @__svm_zero()
  call void @__svm_st(i32 %a, ptr %d)
  ret ptr %d
}

define internal void @clear(ptr %p, i64 %n) noinline {
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 %n, i1 false)
  ret void
}

define void @f(ptr %p, i64 %n) {
  %x = call i32 @__svm_ld(ptr %p)
  call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %p, i64 %n, i1 false)
  call void @__svm_st(i32 %x, ptr %p)
  %y = call i32 @__svm_ld(ptr %p)
  call void @clear(ptr %p, i64 %n)
  call void @__svm_st(i32 %y, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(l.words[..4].iter().map(|w| w.2).collect::<Vec<_>>(), [0; 4]);
    assert_eq!(l.words[4], (0x00, 0, 1, 0, 0));
    assert_eq!(l.words[5], (0x01, 0, 0, 1, 0));
    assert_eq!(l.words[6], (0x00, 0, 1, 0, 0));
    assert_eq!(l.words[7], (0x01, 0, 0, 1, 0));
    // Later optimization must not add calls to them.
    assert!(l.ir.contains(r#""no-builtin-memcpy""#), "{}", l.ir);
    assert!(l.ir.contains(r#""no-builtin-memset""#), "{}", l.ir);
    assert!(!l.ir.contains("no-builtin-memmove"), "{}", l.ir);
}

#[test]
fn thirty_live_values_fit() {
    let mut body = String::from("define void @f(ptr %p) {\n");
    for i in 0..30 {
        body += &format!("  %v{i} = call i32 @__svm_ld(ptr %p)\n");
    }
    for i in 0..30 {
        body += &format!("  call void @__svm_st(i32 %v{i}, ptr %p)\n");
    }
    body += "  ret void\n}\n";
    let l = lower(&body).unwrap();
    let mut outs: Vec<u8> = l.words[..30].iter().map(|w| w.2).collect();
    outs.sort();
    assert_eq!(outs, (0..30).collect::<Vec<_>>());
}

fn error(body: &str) -> String {
    match lower(body) {
        Ok(l) => panic!("expected an error, got:\n{}", l.ir),
        Err(e) => e,
    }
}

#[test]
fn straight_line_overflow_spills() {
    let mut body = String::from("define void @f(ptr %p) {\n");
    for i in 0..32 {
        body += &format!("  %v{i} = call i32 @__svm_ld(ptr %p)\n");
    }
    for i in 0..32 {
        body += &format!("  call void @__svm_st(i32 %v{i}, ptr %p)\n");
    }
    body += "  ret void\n}\n";
    let l = lower(&body).unwrap();
    // Two values are stored to stack slots and loaded back before use.
    assert_eq!(
        l.ir.matches("alloca [64 x i8], align 64").count(),
        2,
        "{}",
        l.ir
    );
    assert_eq!(ops(&l.words).iter().filter(|&&op| op == 0x00).count(), 34);
    assert_eq!(ops(&l.words).iter().filter(|&&op| op == 0x01).count(), 34);
    assert!(
        l.ir.contains(r#""{r1},~{r0},~{memory}"(ptr %svm.spill)"#),
        "{}",
        l.ir
    );
}

#[test]
fn error_too_many_live() {
    // 31 values live around a loop cannot be spilled.
    let mut body = String::from("define void @f(ptr %p, i64 %n) {\nentry:\n");
    for i in 0..31 {
        body += &format!("  %v{i} = call i32 @__svm_ld(ptr %p)\n");
    }
    body += "  br label %loop\nloop:\n  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]\n";
    body += "  %i1 = add i64 %i, 1\n  %c = icmp ult i64 %i1, %n\n";
    body += "  br i1 %c, label %loop, label %exit\nexit:\n";
    for i in 0..31 {
        body += &format!("  call void @__svm_st(i32 %v{i}, ptr %p)\n");
    }
    body += "  ret void\n}\n";
    let e = error(&body);
    assert!(
        e.contains("in `f`: 31 vector values are live at the same time"),
        "{e}"
    );
    assert!(e.contains("at op `ld`"), "{e}");
}

#[test]
fn error_escapes() {
    let e = error(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  store i32 %a, ptr %p
  ret void
}
"#,
    );
    assert!(e.contains("a vector value is stored to memory"), "{e}");

    let e = error(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  call void @sink(i32 %a)
  ret void
}
"#,
    );
    assert!(e.contains("is passed to `sink`"), "{e}");

    let e = error(
        r#"
define i32 @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  ret i32 %a
}
"#,
    );
    assert!(e.contains("is returned from the function"), "{e}");

    let e = error(
        r#"
define i32 @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = add i32 %a, 1
  ret i32 %b
}
"#,
    );
    assert!(
        e.contains("is used by a non-vector instruction `%b = add i32 %a, 1`"),
        "{e}"
    );
}

#[test]
fn error_non_vector_operands() {
    let e = error(
        r#"
define void @f(ptr %p) {
  %a = load i32, ptr %p
  %b = call i32 @__svm_add_d(i32 %a, i32 %a)
  call void @__svm_st(i32 %b, ptr %p)
  ret void
}
"#,
    );
    assert!(
        e.contains("vector operand 0 of `__svm_add_d` is a load from memory"),
        "{e}"
    );

    let e = error(
        r#"
define void @f(ptr %p, i32 %v) {
  call void @__svm_st(i32 %v, ptr %p)
  ret void
}
"#,
    );
    assert!(e.contains("is a function argument"), "{e}");

    let e = error(
        r#"
define void @f(ptr %p, i1 %c) {
entry:
  %a = call i32 @__svm_ld(ptr %p)
  br i1 %c, label %x, label %y
x:
  br label %y
y:
  %m = phi i32 [ %a, %x ], [ 5, %entry ]
  call void @__svm_st(i32 %m, ptr %p)
  ret void
}
"#,
    );
    assert!(e.contains("a phi merges a vector value with a constant"), "{e}");
}

#[test]
fn error_immediates() {
    let e = error(
        r#"
define void @f(ptr %p, i32 %n) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_shli_d(i32 %a, i32 %n)
  call void @__svm_st(i32 %b, ptr %p)
  ret void
}
"#,
    );
    assert!(
        e.contains(
            "the immediate of `__svm_shli_d` is not a compile-time constant"
        ),
        "{e}"
    );

    let e = error(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_shli_d(i32 %a, i32 32)
  %c = call i32 @__svm_clmul(i32 %b, i32 %b, i32 2)
  call void @__svm_st(i32 %c, ptr %p)
  ret void
}
"#,
    );
    assert!(e.contains("immediate 0x20 is out of range for `shli.d`"), "{e}");
    assert!(e.contains("immediate 0x2 is out of range for `clmul`"), "{e}");
}

#[test]
fn error_live_across_indirect_call() {
    let e = error(
        r#"
define void @f(ptr %p, ptr %fp) {
  %a = call i32 @__svm_ld(ptr %p)
  call void %fp()
  call void @__svm_st(i32 %a, ptr %p)
  ret void
}
"#,
    );
    assert!(e.contains("live across calls"), "{e}");
}

#[test]
fn error_unknown_marker() {
    let e = error(
        r#"
declare i32 @__svm_frobnicate(i32)
define void @f(ptr %p) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_frobnicate(i32 %a)
  ret void
}
"#,
    );
    assert!(
        e.contains("unknown vector intrinsic marker `__svm_frobnicate`"),
        "{e}"
    );
}

#[test]
fn masked_ops_pass_the_mask_in_r2() {
    let l = lower(
        r#"
define void @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %c = call i32 @__svm_add_d_maskz(i32 %a, i32 %b, i64 %k)
  %d = call i32 @__svm_add_d_mask(i32 %c, i32 %a, i32 %b, i64 %k)
  %e = call i32 @__svm_dpbusd_mask(i32 %d, i32 %a, i32 %b, i64 %k)
  call void @__svm_st(i32 %e, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    // mode 2 writes a new register, mode 1 merges into the source (tied)
    assert_eq!(
        l.words[2..5],
        [(0x12, 0, 2, 0, 1), (0x12, 0, 2, 0, 1), (0x50, 0, 2, 0, 1)]
    );
    assert_eq!(modes(&l.ir), [0, 0, 2, 1, 1, 0]);
    assert!(
        l.ir.contains(r#".quad 0x1200080100002285", "{r2},~{r0}"(i64 %k)"#),
        "{}",
        l.ir
    );
}

#[test]
fn merge_source_live_after_is_copied() {
    let l = lower(
        r#"
define void @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %c = call i32 @__svm_add_d_mask(i32 %a, i32 %a, i32 %b, i64 %k)
  call void @__svm_st(i32 %c, ptr %p)
  call void @__svm_st(i32 %a, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(l.words[2], (0x09, 0, 2, 0, 0));
    assert_eq!(l.words[3], (0x12, 0, 2, 0, 1));
}

#[test]
fn mask_data_and_scalar_results() {
    let l = lower(
        r#"
define i64 @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_blendm(i32 %a, i32 %a, i64 %k, i32 2)
  %m = call i64 @__svm_cmp(i32 %a, i32 %b, i32 37)
  %n = call i64 @__svm_mpopcnt(i64 %m)
  %c = call i32 @__svm_sha256rnds2(i32 %b, i32 %a, i32 %a)
  call void @__svm_st(i32 %c, ptr %p)
  ret i64 %n
}
"#,
    )
    .unwrap();
    assert_eq!(
        l.words,
        [
            (0x00, 0, 0, 0, 0),
            (0x79, 2, 1, 0, 0),
            (0x46, 37, 0, 0, 1),
            (0xf6, 0, 0, 0, 0),
            (0xe1, 0, 1, 0, 0),
            (0x01, 0, 0, 1, 0)
        ]
    );
    assert_eq!(modes(&l.ir), [0; 6]);
    assert!(l.ir.contains(r#""{r2},~{r0}"(i64 %k)"#), "{}", l.ir);
    assert!(
        l.ir.contains(r#"= call i64 asm sideeffect ".quad 0x4625000100002085", "={r0}"()"#),
        "{}",
        l.ir
    );
    assert!(
        l.ir.contains(r#"= call i64 asm sideeffect ".quad 0xf600000000002085", "={r0},{r2}"(i64 %1)"#),
        "{}",
        l.ir
    );
}

#[test]
fn masked_memory_gather_scatter() {
    let l = lower(
        r#"
define void @f(ptr %p, i64 %k) {
  %q = getelementptr inbounds i8, ptr %p, i64 128
  %a = call i32 @__svm_ld_m_maskz(ptr %q, i64 %k, i32 3)
  %b = call i32 @__svm_gather_qq_mask(i32 %a, i32 %a, ptr %p, i64 %k, i32 3)
  call void @__svm_scatter_qq(i32 %a, i32 %b, ptr %q, i32 0)
  call void @__svm_st_m_mask(i32 %b, ptr %p, i64 %k, i32 1)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(
        l.words,
        [
            (0x0a, 3, 0, 0, 0),
            (0x09, 0, 1, 0, 0),
            (0xe8, 3, 1, 0, 0),
            (0xec, 0, 0, 0, 1),
            (0x0b, 1, 0, 1, 0)
        ]
    );
    assert_eq!(modes(&l.ir), [2, 0, 1, 0, 1]);
    assert_eq!(offsets(&l.ir), [128, 0, 0, 128, 0]);
    assert!(
        l.ir.contains(r#""{r1},{r2},~{r0},~{memory}"(ptr %p, i64 %k)"#),
        "{}",
        l.ir
    );
}

#[test]
fn narrow_and_broadcast_memory_dword_gather_scatter() {
    let l = lower(
        r#"
declare i32 @__svm_ld_x(ptr)
declare void @__svm_st_y(i32, ptr)
declare i32 @__svm_ldbc_x_maskz(ptr, i64)
declare i32 @__svm_gather_dd(i32, ptr, i32)
declare void @__svm_scatter_qd_mask(i32, i32, ptr, i64, i32)

define void @f(ptr %p, i64 %k) {
  %q = getelementptr inbounds i8, ptr %p, i64 16
  %a = call i32 @__svm_ld_x(ptr %q)
  %b = call i32 @__svm_ldbc_x_maskz(ptr %p, i64 %k)
  %c = call i32 @__svm_gather_dd(i32 %a, ptr %p, i32 2)
  call void @__svm_st_y(i32 %c, ptr %q)
  call void @__svm_scatter_qd_mask(i32 %a, i32 %b, ptr %q, i64 %k, i32 1)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0xf0, 0xf4, 0xe9, 0x0f, 0xef]);
    assert_eq!(modes(&l.ir), [0, 2, 0, 0, 1]);
    assert_eq!(offsets(&l.ir), [16, 0, 0, 16, 16]);
    assert_eq!(l.ir.matches("~{memory}").count(), 5, "{}", l.ir);
}

#[test]
fn error_masked_marker_arguments() {
    let e = error(
        r#"
define void @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_add_d_mask(i32 %a, i32 %a, i64 %k)
  ret void
}
"#,
    );
    assert!(
        e.contains(
            "`__svm_add_d_mask` takes 4 arguments, but is called with 3"
        ),
        "{e}"
    );
    // modes the op does not have
    let e = error(
        r#"
declare i32 @__svm_zero_mask(i32, i64)
declare i32 @__svm_ld_m(ptr, i32)
define void @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %c = call i32 @__svm_zero_mask(i32 %a, i64 %k)
  %d = call i32 @__svm_ld_m(ptr %p, i32 0)
  ret void
}
"#,
    );
    assert!(
        e.contains("`__svm_zero_mask`: `zero` has no such mask mode"),
        "{e}"
    );
    assert!(e.contains("`__svm_ld_m`: `ld.m` has no such mask mode"), "{e}");
}

/// `n` loads from `%p` at offsets 64 apart, then their sum stored.
fn load_run(n: usize, between: &str) -> String {
    let mut body = String::from("define void @f(ptr %p, ptr %q) {\n");
    for k in 0..n {
        body += &format!(
            "  %p{k} = getelementptr inbounds i8, ptr %p, i64 {}\n  %v{k} = call i32 @__svm_ld(ptr %p{k})\n{}",
            64 * k,
            between.replace('#', &k.to_string())
        );
    }
    body += "  %s0 = call i32 @__svm_add_d(i32 %v0, i32 %v0)\n";
    for k in 1..n {
        body += &format!(
            "  %s{k} = call i32 @__svm_add_d(i32 %s{}, i32 %v{k})\n",
            k - 1
        );
    }
    body += &format!(
        "  call void @__svm_st(i32 %s{}, ptr %q)\n  ret void\n}}\n",
        n - 1
    );
    body
}

#[test]
fn load_runs_fuse_to_ldn() {
    let l = lower(&load_run(4, "")).unwrap();
    assert_eq!(l.words[0], (0x0c, 4, 0, 0, 0));
    assert_eq!(ops(&l.words)[1..5], [0x12; 4]);
    assert_eq!(l.ir.matches("(ptr %p)").count(), 1, "{}", l.ir);

    // a store in between could change what the later loads read
    let between = "  store i8 0, ptr %q\n";
    let l = lower(&load_run(4, between)).unwrap();
    assert_eq!(ops(&l.words)[..4], [0x00; 4]);
    assert_eq!(offsets(&l.ir)[..4], [0, 64, 128, 192]);

    // scalar arithmetic in between is fine
    let l = lower(&load_run(3, "  %x# = add i64 0, 1\n")).unwrap();
    assert_eq!(l.words[0], (0x0c, 3, 0, 0, 0));

    // the 30 registers are enough for one ldn
    let l = lower(&load_run(30, "")).unwrap();
    assert_eq!(l.words[0], (0x0c, 30, 0, 0, 0));
}

#[test]
fn load_runs_fuse_without_inbounds() {
    let l = lower(
        r#"
define void @f(ptr %p) {
  %q = getelementptr i8, ptr %p, i64 64
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %q)
  %c = call i32 @__svm_add_d(i32 %a, i32 %b)
  call void @__svm_st(i32 %c, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x0c, 0x12, 0x01]);
}

#[test]
fn store_runs_fuse_to_stn() {
    let l = lower(
        r#"
define void @f(ptr %p, ptr %q) {
  %a = call i32 @__svm_ld(ptr %q)
  %b = call i32 @__svm_add_d(i32 %a, i32 %a)
  %c = call i32 @__svm_add_d(i32 %b, i32 %a)
  %d = call i32 @__svm_add_d(i32 %c, i32 %b)
  %p1 = getelementptr inbounds i8, ptr %p, i64 64
  %p2 = getelementptr inbounds i8, ptr %p, i64 128
  %p3 = getelementptr inbounds i8, ptr %p, i64 192
  call void @__svm_st(i32 %a, ptr %p1)
  call void @__svm_st(i32 %b, ptr %p2)
  call void @__svm_st(i32 %c, ptr %p3)
  ret void
}
"#,
    )
    .unwrap();
    let stn = l.words.iter().find(|w| w.0 == 0x0d).copied();
    assert_eq!(stn.map(|w| w.1), Some(3), "{:?}", l.words);
    assert_eq!(*offsets(&l.ir).last().unwrap(), 64);
}

#[test]
fn explicit_ldn_stn() {
    let l = lower(
        r#"
define void @f(ptr %p, ptr %q) {
  %a = call i32 @__svm_ldn(ptr %p, i32 0, i32 3)
  %b = call i32 @__svm_ldn(ptr %p, i32 1, i32 3)
  %c = call i32 @__svm_ldn(ptr %p, i32 2, i32 3)
  %x = call i32 @__svm_add_d(i32 %a, i32 %b)
  %y = call i32 @__svm_add_d(i32 %b, i32 %c)
  call void @__svm_stn(i32 %x, ptr %q, i32 0, i32 2)
  call void @__svm_stn(i32 %y, ptr %q, i32 1, i32 2)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(
        l.words,
        [
            (0x0c, 3, 0, 0, 0),
            (0x12, 0, 0, 0, 1),
            (0x12, 0, 1, 1, 2),
            (0x0d, 2, 0, 0, 0)
        ]
    );

    // registers that cannot be consecutive: single loads at p + 64k
    let l = lower(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ldn(ptr %p, i32 0, i32 2)
  call void @__svm_st(i32 %a, ptr %p)
  %b = call i32 @__svm_ldn(ptr %p, i32 1, i32 2)
  call void @__svm_st(i32 %b, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(ops(&l.words), [0x00, 0x01, 0x00, 0x01]);
    assert_eq!(offsets(&l.ir), [0, 0, 64, 0]);
}

#[test]
fn error_ldn_counts() {
    let e = error(
        r#"
define void @f(ptr %p) {
  %a = call i32 @__svm_ldn(ptr %p, i32 0, i32 31)
  %b = call i32 @__svm_ldn(ptr %p, i32 2, i32 2)
  ret void
}
"#,
    );
    assert!(e.contains("immediate 0 of 31 is out of range for `ldn`"), "{e}");
    assert!(e.contains("immediate 2 of 2 is out of range for `ldn`"), "{e}");
}

#[test]
fn two_table_permute_ties_the_dying_operand() {
    let decls = r#"
declare i32 @__svm_perm2t(i32, i32, i32, i32)
declare i32 @__svm_perm2t_mask(i32, i32, i32, i64, i32)
declare i32 @__svm_perm2_w_maskz(i32, i32, i32, i64)
declare i32 @__svm_perm2i_mask(i32, i32, i32, i64, i32)
"#;
    // the table stays live, the index dies: perm2i with out = index
    let l = lower(&format!(
        r#"{decls}
define void @f(ptr %p) {{
  %t = call i32 @__svm_ld(ptr %p)
  %i = call i32 @__svm_ld(ptr %p)
  %u = call i32 @__svm_ld(ptr %p)
  %r = call i32 @__svm_perm2t(i32 %t, i32 %i, i32 %u, i32 3)
  call void @__svm_st(i32 %r, ptr %p)
  call void @__svm_st(i32 %t, ptr %p)
  ret void
}}
"#
    ))
    .unwrap();
    let (t, i, u) = (l.words[0].2, l.words[1].2, l.words[2].2);
    assert_eq!(l.words[3], (0x5f, 3, i, t, u));
    // the index stays live: perm2t with out = table
    let l = lower(&format!(
        r#"{decls}
define void @f(ptr %p) {{
  %t = call i32 @__svm_ld(ptr %p)
  %i = call i32 @__svm_ld(ptr %p)
  %u = call i32 @__svm_ld(ptr %p)
  %r = call i32 @__svm_perm2t(i32 %t, i32 %i, i32 %u, i32 2)
  call void @__svm_st(i32 %r, ptr %p)
  call void @__svm_st(i32 %i, ptr %p)
  ret void
}}
"#
    ))
    .unwrap();
    let (t, i, u) = (l.words[0].2, l.words[1].2, l.words[2].2);
    assert_eq!(l.words[3], (0x85, 2, t, i, u));
    // zero masking may switch (perm2.w becomes perm2i w), merge keeps its source
    let l = lower(&format!(
        r#"{decls}
define void @f(ptr %p, i64 %k) {{
  %t = call i32 @__svm_ld(ptr %p)
  %i = call i32 @__svm_ld(ptr %p)
  %u = call i32 @__svm_ld(ptr %p)
  %r = call i32 @__svm_perm2_w_maskz(i32 %t, i32 %i, i32 %u, i64 %k)
  %s = call i32 @__svm_perm2t_mask(i32 %t, i32 %r, i32 %u, i64 %k, i32 0)
  %v = call i32 @__svm_perm2i_mask(i32 %s, i32 %t, i32 %u, i64 %k, i32 1)
  call void @__svm_st(i32 %v, ptr %p)
  call void @__svm_st(i32 %t, ptr %p)
  ret void
}}
"#
    ))
    .unwrap();
    let (t, u) = (l.words[0].2, l.words[2].2);
    assert_eq!(ops(&l.words)[3..], [0x5f, 0x09, 0x85, 0x5f, 0x01, 0x01]);
    assert_eq!(l.words[3].1, 1);
    assert_eq!((l.words[3].3, l.words[3].4), (t, u));
    assert_eq!(modes(&l.ir)[3..7], [2, 0, 1, 1]);
    assert_eq!(l.words[6], (0x5f, 1, l.words[5].2, t, u));
}

#[test]
fn data_movement_markers() {
    let l = lower(
        r#"
declare i32 @__svm_compress_maskz(i32, i64, i32)
declare i32 @__svm_bcast_w_mask(i32, i32, i64, i32)
declare i64 @__svm_extract_b(i32, i32)
declare i32 @__svm_insert_w(i32, i64, i32)
declare i32 @__svm_movr_q(i64)
declare i32 @__svm_narrow_q_b(i32)
define i64 @f(ptr %p, i64 %x, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_compress_maskz(i32 %a, i64 %k, i32 2)
  %c = call i32 @__svm_bcast_w_mask(i32 %a, i32 %b, i64 %k, i32 9)
  %d = call i32 @__svm_insert_w(i32 %c, i64 %x, i32 31)
  %e = call i32 @__svm_movr_q(i64 %x)
  %f = call i32 @__svm_narrow_q_b(i32 %e)
  call void @__svm_st(i32 %d, ptr %p)
  call void @__svm_st(i32 %f, ptr %p)
  %r = call i64 @__svm_extract_b(i32 %a, i32 63)
  ret i64 %r
}
"#,
    )
    .unwrap();
    let ops = ops(&l.words);
    for op in [0x92, 0x7c, 0xac, 0xae, 0x97, 0xa9] {
        assert!(ops.contains(&op), "{op:#x} {:?}", l.words);
    }
    let modes = modes(&l.ir);
    let mode = |op| modes[ops.iter().position(|o| *o == op).unwrap()];
    assert_eq!((mode(0x92), mode(0x7c), mode(0xac)), (2, 1, 0));
    let insert = l.words[ops.iter().position(|o| *o == 0xac).unwrap()];
    assert_eq!(insert.1, 31);
    assert!(l.ir.contains(r#""{r1},~{r0}"(i64 %x)"#), "{}", l.ir);
}

#[test]
fn arithmetic_masks_and_saturating_accumulators() {
    let l = lower(
        r#"
declare i32 @__svm_mul52lo_maskz(i32, i32, i64)
declare i32 @__svm_abs_q_mask(i32, i32, i64)
declare i32 @__svm_dpbusds_mask(i32, i32, i32, i64)
declare i32 @__svm_sad(i32, i32)
define void @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %c = call i32 @__svm_mul52lo_maskz(i32 %a, i32 %b, i64 %k)
  %d = call i32 @__svm_abs_q_mask(i32 %c, i32 %a, i64 %k)
  %e = call i32 @__svm_dpbusds_mask(i32 %d, i32 %a, i32 %b, i64 %k)
  %f = call i32 @__svm_sad(i32 %e, i32 %b)
  call void @__svm_st(i32 %f, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    assert_eq!(
        l.words[2..6],
        [
            (0xaf, 0, 2, 0, 1),
            (0xb1, 0, 2, 0, 0),
            (0x56, 0, 2, 0, 1),
            (0xc5, 0, 0, 2, 1)
        ]
    );
    assert_eq!(modes(&l.ir), [0, 0, 2, 1, 1, 0, 0]);
}

#[test]
fn crypto_ops() {
    let l = lower(
        r#"
declare i32 @__svm_gf2p8affine_mask(i32, i32, i32, i64, i32)
declare i32 @__svm_aesenc(i32, i32)
declare i32 @__svm_sha1msg2(i32, i32)
define void @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_gf2p8affine_mask(i32 %a, i32 %a, i32 %a, i64 %k, i32 99)
  %c = call i32 @__svm_aesenc(i32 %b, i32 %a)
  %d = call i32 @__svm_sha1msg2(i32 %c, i32 %b)
  call void @__svm_st(i32 %d, ptr %p)
  ret void
}
"#,
    )
    .unwrap();
    // The merge source is copied, sha1msg2 reads and writes the dying %c
    assert_eq!(
        l.words,
        [
            (0x00, 0, 0, 0, 0),
            (0x09, 0, 1, 0, 0),
            (0xdb, 99, 1, 0, 0),
            (0xdd, 0, 0, 1, 0),
            (0xe7, 0, 0, 1, 0),
            (0x01, 0, 0, 0, 0)
        ]
    );
    assert_eq!(modes(&l.ir), [0, 0, 1, 0, 0, 0]);
}

#[test]
fn syscall_transport_passes_descriptor_scalar_and_mask() {
    let body = r#"
define i64 @f(ptr %p, i64 %k) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_add_d_maskz(i32 %a, i32 %a, i64 %k)
  %c = call i32 @__svm_add_d(i32 %b, i32 %a)
  %d = call i32 @__svm_add_d(i32 %c, i32 %a)
  %h = call i64 @__svm_hsum_d(i32 %d)
  ret i64 %h
}
"#;
    let call = lower_as(body, Transport::Call).unwrap();
    // one call per op; the ops that don't touch program memory only
    // touch the vector registers
    let calls: Vec<&str> = call
        .ir
        .lines()
        .filter(|line| line.contains("call i64 @sol_avx512("))
        .collect();
    assert_eq!(calls.len(), 5, "{}", call.ir);
    assert!(calls[0].contains("(ptr @svm.desc, i64 %"), "{}", call.ir);
    assert!(calls[1].contains(", i64 poison, i64 %k) #1"), "{}", call.ir);
    assert!(calls[1..].iter().all(|c| c.ends_with("#1")), "{}", call.ir);
    assert!(
        call.ir.contains(
            "attributes #1 = { nounwind memory(inaccessiblemem: readwrite) }"
        ),
        "{}",
        call.ir
    );
    let asm = lower_as(body, Transport::Asm).unwrap();
    assert!(asm.ir.contains(r#""r1 = ${0} ll\0Acall sol_avx512", "i,{r2},~{r1},~{r0},~{memory}""#), "{}", asm.ir);
    assert!(asm.ir.contains(r#""i,{r3},~{r1},~{r0}""#), "{}", asm.ir);
    assert!(
        asm.ir
            .contains(r#""r1 = ${1} ll\0Acall sol_avx512", "={r0},i,~{r1}""#),
        "{}",
        asm.ir
    );
}

#[test]
fn syscall_transport_select_is_blendm() {
    let body = r#"
define void @f(ptr %p, i1 %c) {
  %a = call i32 @__svm_ld(ptr %p)
  %b = call i32 @__svm_ld(ptr %p)
  %s = select i1 %c, i32 %a, i32 %b
  call void @__svm_st(i32 %s, ptr %p)
  call void @__svm_st(i32 %a, ptr %p)
  call void @__svm_st(i32 %b, ptr %p)
  ret void
}
"#;
    for transport in [Transport::Call, Transport::Asm] {
        let l = lower_as(body, transport).unwrap();
        let d = descriptors(&l.ir);
        let blendm: Vec<_> =
            d.iter().filter(|d| d[0] == super::ops::BLENDM).collect();
        assert_eq!(blendm.len(), 1, "{}", l.ir);
        // out ← mask ? in1 (a) : in0 (b), q lanes
        assert_eq!(
            blendm[0][..5],
            [super::ops::BLENDM, 3, 2, 1, 0],
            "{}",
            l.ir
        );
        assert!(l.ir.contains("sext i1 %c to i64"), "{}", l.ir);
    }
}
