//! IR-level tests: parse textual IR, run the pass, and decode the emitted
//! `.quad` words. Every successful allocation is also checked by the
//! allocator's own register simulation.

use std::ffi::CStr;

use bpf_linker::llvm_sys::core::*;

use super::{lower_module, parse_ir};

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
declare void @sink(i32)
"#;

/// A decoded vector op: `(op, imm8, out, in0, in1)`.
type Word = (u8, u8, u8, u8, u8);

struct Lowered {
    ir: String,
    /// Decoded ops in the order they appear in the printed module.
    words: Vec<Word>,
}

fn lower(body: &str) -> Result<Lowered, String> {
    unsafe {
        let context = LLVMContextCreate();
        let module = parse_ir(context, &format!("{DECLS}\n{body}"));
        let result =
            lower_module(module).map_err(|e| e.to_string()).map(|_| {
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
            assert_eq!(word & 0xffff, 0x2085, "CALL_IMM src=2");
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
