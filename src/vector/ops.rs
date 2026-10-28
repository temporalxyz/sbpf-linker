//! The vector op table from `SPEC.md`: marker name, opcode, operand kind and
//! immediate range.

/// Which instruction fields an op uses (see the `kinds` table in the spec).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    R3,
    A3,
    A3i,
    R3i,
    R2,
    R2i,
    R0,
    Ls,
    St,
    Sr,
    Sri,
    Ai,
}

impl Kind {
    /// Number of vector arguments of the marker (the accumulator included).
    pub fn vector_args(self) -> usize {
        match self {
            Kind::R0 | Kind::Ls => 0,
            Kind::R2
            | Kind::R2i
            | Kind::St
            | Kind::Sr
            | Kind::Sri
            | Kind::Ai => 1,
            Kind::R3 | Kind::R3i => 2,
            Kind::A3 | Kind::A3i => 3,
        }
    }

    /// Whether the marker takes a scalar `r1` argument after the vectors.
    pub fn has_scalar(self) -> bool {
        matches!(self, Kind::Ls | Kind::St | Kind::Ai)
    }

    /// Whether the marker takes an imm8 as its last argument.
    pub fn has_imm(self) -> bool {
        matches!(
            self,
            Kind::A3i | Kind::R3i | Kind::R2i | Kind::Sri | Kind::Ai
        )
    }

    /// Whether the op writes a vector register (`out`).
    pub fn defines_vector(self) -> bool {
        !matches!(self, Kind::St | Kind::Sr | Kind::Sri)
    }

    /// Whether the op returns a scalar in `r0`.
    pub fn defines_scalar(self) -> bool {
        matches!(self, Kind::Sr | Kind::Sri)
    }

    /// Whether `out` is also read (the first vector argument is the
    /// accumulator and must share `out`'s register).
    pub fn tied(self) -> bool {
        matches!(self, Kind::A3 | Kind::A3i | Kind::Ai)
    }
}

/// Valid imm8 values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Imm {
    None,
    Below(u32),
    Clmul,
}

impl Imm {
    pub fn accepts(self, imm: u64) -> bool {
        match self {
            Imm::None => imm == 0,
            Imm::Below(limit) => imm < u64::from(limit),
            Imm::Clmul => matches!(imm, 0x00 | 0x01 | 0x10 | 0x11),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OpInfo {
    /// Spec name, e.g. `add.d`. The marker is `__svm_` + name with `.` → `_`.
    pub name: &'static str,
    pub code: u8,
    pub kind: Kind,
    pub imm: Imm,
}

impl OpInfo {
    /// `ld`, `st`, `ldbc.d`, `ldbc.q`: they access `r1 + off`.
    pub fn is_memory(&self) -> bool {
        self.code <= 0x03
    }
}

pub const MARKER_PREFIX: &str = "__svm_";

/// `mov`, inserted by the pass for phi copies and tied accumulators.
pub const MOV: u8 = 0x09;
/// `xor`, used to swap two registers in cyclic phi copies.
pub const XOR: u8 = 0x2a;

macro_rules! ops {
    ($($name:literal $code:literal $kind:ident $($imm:expr)?;)*) => {
        pub const OPS: &[OpInfo] = &[$(OpInfo {
            name: $name,
            code: $code,
            kind: Kind::$kind,
            imm: ops!(@imm $($imm)?),
        },)*];
    };
    (@imm) => { Imm::None };
    (@imm $imm:expr) => { $imm };
}

use Imm::{Below, Clmul};

ops! {
    "ld" 0x00 Ls;
    "st" 0x01 St;
    "ldbc.d" 0x02 Ls;
    "ldbc.q" 0x03 Ls;
    "splat.b" 0x04 Ls;
    "splat.w" 0x05 Ls;
    "splat.d" 0x06 Ls;
    "splat.q" 0x07 Ls;
    "zero" 0x08 R0;
    "add.b" 0x10 R3;
    "add.w" 0x11 R3;
    "add.d" 0x12 R3;
    "add.q" 0x13 R3;
    "sub.b" 0x14 R3;
    "sub.w" 0x15 R3;
    "sub.d" 0x16 R3;
    "sub.q" 0x17 R3;
    "mullo.w" 0x18 R3;
    "mullo.d" 0x19 R3;
    "mullo.q" 0x1a R3;
    "mulwu" 0x1b R3;
    "mulws" 0x1c R3;
    "mulhi.w" 0x1d R3;
    "mulhiu.w" 0x1e R3;
    "max.d" 0x20 R3;
    "min.d" 0x21 R3;
    "maxu.d" 0x22 R3;
    "minu.d" 0x23 R3;
    "max.w" 0x24 R3;
    "min.w" 0x25 R3;
    "maxu.q" 0x26 R3;
    "minu.q" 0x27 R3;
    "and" 0x28 R3;
    "or" 0x29 R3;
    "xor" 0x2a R3;
    "andn" 0x2b R3;
    "ternlog" 0x2c A3i Below(256);
    "maxu.w" 0x2e R3;
    "minu.w" 0x2f R3;
    "shli.w" 0x30 R2i Below(16);
    "shli.d" 0x31 R2i Below(32);
    "shli.q" 0x32 R2i Below(64);
    "shri.w" 0x33 R2i Below(16);
    "shri.d" 0x34 R2i Below(32);
    "shri.q" 0x35 R2i Below(64);
    "sari.w" 0x36 R2i Below(16);
    "sari.d" 0x37 R2i Below(32);
    "sari.q" 0x38 R2i Below(64);
    "roli.d" 0x39 R2i Below(32);
    "roli.q" 0x3a R2i Below(64);
    "shlv.d" 0x3b R3;
    "shlv.q" 0x3c R3;
    "shrv.d" 0x3d R3;
    "shrv.q" 0x3e R3;
    "rolv.d" 0x3f R3;
    "rolv.q" 0x40 R3;
    "cmpeq.b" 0x48 R3;
    "cmpeq.d" 0x49 R3;
    "cmpeq.q" 0x4a R3;
    "cmpgt.d" 0x4b R3;
    "cmpgt.q" 0x4c R3;
    "cmpgtu.q" 0x4d R3;
    "dpbusd" 0x50 A3;
    "dpwssd" 0x51 A3;
    "maddubs" 0x52 R3;
    "maddwd" 0x53 R3;
    "madd52lo" 0x54 A3;
    "madd52hi" 0x55 A3;
    "packs.d" 0x58 R3;
    "packs.w" 0x59 R3;
    "packus.w" 0x5a R3;
    "permd" 0x5b R3;
    "permq" 0x5c R3;
    "shufb" 0x5d R3;
    "perm2.w" 0x5e A3;
    "narrowus.d.b" 0x60 R2;
    "narrows.d.b" 0x61 R2;
    "narrows.d.w" 0x62 R2;
    "narrow.d.b" 0x63 R2;
    "zx.b.d" 0x64 R2;
    "sx.b.d" 0x65 R2;
    "zx.w.d" 0x66 R2;
    "sx.w.d" 0x67 R2;
    "zx.d.q" 0x68 R2;
    "sx.d.q" 0x69 R2;
    "popcnt.b" 0x6a R2;
    "popcnt.q" 0x6b R2;
    "hsum.d" 0x70 Sr;
    "hsum.q" 0x71 Sr;
    "extract.q" 0x72 Sri Below(8);
    "extract.d" 0x73 Sri Below(16);
    "insert.q" 0x74 Ai Below(8);
    "movemask.b" 0x75 Sr;
    "clmul" 0x78 R3i Clmul;
}

/// Looks up a marker symbol (`__svm_add_d`) in the op table.
pub fn by_marker(symbol: &str) -> Option<&'static OpInfo> {
    let name = symbol.strip_prefix(MARKER_PREFIX)?;
    OPS.iter().find(|op| {
        op.name.len() == name.len()
            && op
                .name
                .bytes()
                .zip(name.bytes())
                .all(|(a, b)| a == b || (a == b'.' && b == b'_'))
    })
}

/// Encodes a vector op as the 8 bytes of a `CALL_IMM src=2` instruction,
/// returned as a little-endian u64.
pub fn encode(code: u8, imm8: u8, out: u8, in0: u8, in1: u8) -> u64 {
    debug_assert!(out < 30 && in0 < 30 && in1 < 30);
    let imm = u32::from(code) << 24
        | u32::from(imm8) << 16
        | u32::from(out) << 10
        | u32::from(in0) << 5
        | u32::from(in1);
    0x85 | 0x20 << 8 | u64::from(imm) << 32
}

/// Sets the `off` field (bytes 2–3) of an encoded op.
pub fn with_off(word: u64, off: i16) -> u64 {
    word | u64::from(off as u16) << 16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_lookup() {
        assert_eq!(by_marker("__svm_add_d").unwrap().code, 0x12);
        assert_eq!(by_marker("__svm_narrowus_d_b").unwrap().code, 0x60);
        assert_eq!(by_marker("__svm_and").unwrap().code, 0x28);
        assert!(by_marker("__svm_mov").is_none());
        assert!(by_marker("__svm_add").is_none());
        assert!(by_marker("svm_add_d").is_none());
    }

    #[test]
    fn opcodes_unique() {
        for (i, a) in OPS.iter().enumerate() {
            for b in &OPS[i + 1..] {
                assert_ne!(a.code, b.code, "{} / {}", a.name, b.name);
            }
        }
    }

    #[test]
    fn encoding() {
        // add.d v3 <- v1, v2
        let word = encode(0x12, 0, 3, 1, 2);
        assert_eq!(word.to_le_bytes()[..4], [0x85, 0x20, 0, 0]);
        assert_eq!(word >> 32, 0x12 << 24 | 3 << 10 | 1 << 5 | 2);
        assert_eq!(encode(0x2c, 0xff, 29, 29, 29) >> 32, 0x2cff_77bd);
    }
}
