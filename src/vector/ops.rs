//! The vector op table from `SPEC.md`: marker name, opcode, operand kind,
//! mask modes and immediate range.

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
    S3,
    S3i,
    A2,
    A2i,
    Lsi,
    Sti,
    Ga,
    Sc,
    S0,
    /// One register of `_svm512_loadu_n`: `__svm_ldn(p, k, n)` is register
    /// `k` of an `ldn` of `n` registers at `p`.
    Ldn,
    /// One register of `_svm512_storeu_n`: `__svm_stn(a, p, k, n)`.
    Stn,
}

impl Kind {
    /// Number of vector arguments of the unmasked marker (the accumulator
    /// included).
    fn vector_args(self) -> usize {
        match self {
            Kind::R0 | Kind::Ls | Kind::Lsi | Kind::S0 | Kind::Ldn => 0,
            Kind::R2
            | Kind::R2i
            | Kind::St
            | Kind::Sr
            | Kind::Sri
            | Kind::Ai
            | Kind::Sti
            | Kind::Ga
            | Kind::Stn => 1,
            Kind::R3
            | Kind::R3i
            | Kind::S3
            | Kind::S3i
            | Kind::A2
            | Kind::A2i
            | Kind::Sc => 2,
            Kind::A3 | Kind::A3i => 3,
        }
    }

    /// Whether the marker takes a scalar `r1` argument after the vectors.
    pub fn has_scalar(self) -> bool {
        matches!(
            self,
            Kind::Ls
                | Kind::St
                | Kind::Ai
                | Kind::Lsi
                | Kind::Sti
                | Kind::Ga
                | Kind::Sc
                | Kind::Ldn
                | Kind::Stn
        )
    }

    /// Number of immediate arguments at the end of the marker.
    pub fn imm_args(self) -> usize {
        match self {
            Kind::A3i
            | Kind::R3i
            | Kind::R2i
            | Kind::Sri
            | Kind::Ai
            | Kind::S3i
            | Kind::A2i
            | Kind::Lsi
            | Kind::Sti
            | Kind::Ga
            | Kind::Sc => 1,
            Kind::Ldn | Kind::Stn => 2,
            _ => 0,
        }
    }

    /// Whether the op writes a vector register (`out`).
    pub fn defines_vector(self) -> bool {
        !matches!(
            self,
            Kind::St
                | Kind::Sr
                | Kind::Sri
                | Kind::S3
                | Kind::S3i
                | Kind::Sti
                | Kind::Sc
                | Kind::S0
                | Kind::Stn
        )
    }

    /// Whether the op returns a scalar in `r0`.
    pub fn defines_scalar(self) -> bool {
        matches!(self, Kind::Sr | Kind::Sri | Kind::S3 | Kind::S3i | Kind::S0)
    }

    /// Whether the unmasked op reads `out` (the first vector argument is the
    /// accumulator).
    fn reads_out(self) -> bool {
        matches!(self, Kind::A3 | Kind::A3i | Kind::Ai | Kind::A2 | Kind::A2i)
    }
}

/// Valid imm8 values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Imm {
    None,
    Below(u32),
    Clmul,
    /// A lane width code (0 = b, 1 = w, 2 = d, 3 = q) in the bit set.
    Widths(u8),
    /// Predicate (not 3 or 7) in bits 2:0, unsigned in bit 3, width in
    /// bits 5:4.
    Cmp,
}

impl Imm {
    pub fn accepts(self, imm: u64) -> bool {
        match self {
            Imm::None => imm == 0,
            Imm::Below(limit) => imm < u64::from(limit),
            Imm::Clmul => matches!(imm, 0x00 | 0x01 | 0x10 | 0x11),
            Imm::Widths(set) => imm < 4 && set >> imm & 1 == 1,
            Imm::Cmp => imm < 0x40 && !matches!(imm & 7, 3 | 7),
        }
    }
}

/// Mask modes (the `dst` field): 0 = unmasked, 1 = merge (`_mask` marker),
/// 2 = zero (`_maskz` marker).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    /// Mode 0 only.
    None,
    /// Mode 0 only, the mask is an operand (passed in `r2`).
    Data,
    /// Bit set of the allowed modes.
    Modes(u8),
}

#[derive(Clone, Copy, Debug)]
pub struct OpInfo {
    /// Spec name, e.g. `add.d`. The marker is `__svm_` + name with `.` → `_`,
    /// plus `_mask` or `_maskz` for the masked modes.
    pub name: &'static str,
    pub code: u8,
    pub kind: Kind,
    pub mask: Mask,
    pub imm: Imm,
}

impl OpInfo {
    /// Whether the op accesses memory at `r1 + off`.
    pub fn is_memory(&self) -> bool {
        matches!(self.code, 0x00..=0x03 | 0x0e | 0x0f | 0xf0..=0xf5)
            || matches!(
                self.kind,
                Kind::Lsi
                    | Kind::Sti
                    | Kind::Ga
                    | Kind::Sc
                    | Kind::Ldn
                    | Kind::Stn
            )
    }

    pub fn allows_mode(&self, mode: u8) -> bool {
        match self.mask {
            Mask::None | Mask::Data => mode == 0,
            Mask::Modes(set) => set >> mode & 1 == 1,
        }
    }

    /// Whether the marker in `mode` takes a mask argument (passed in `r2`).
    pub fn reads_mask(&self, mode: u8) -> bool {
        mode != 0 || self.mask == Mask::Data
    }

    /// Whether `out` is also read in `mode`: the first vector argument is
    /// the accumulator, or the merge source, and must share `out`'s
    /// register.
    pub fn tied(&self, mode: u8) -> bool {
        self.kind.reads_out() || (mode == 1 && self.kind.defines_vector())
    }

    /// Number of vector arguments of the marker in `mode`.
    pub fn vector_args(&self, mode: u8) -> usize {
        self.kind.vector_args()
            + usize::from(self.tied(mode) && !self.kind.reads_out())
    }
}

pub const MARKER_PREFIX: &str = "__svm_";

/// `ld` and `st`, used for spills and for the registers of `ldn` and `stn`
/// that are not fused.
pub const LD: u8 = 0x00;
pub const ST: u8 = 0x01;
/// `mov`, inserted by the pass for phi copies and tied accumulators.
pub const MOV: u8 = 0x09;
/// `xor`, used to swap two registers in cyclic phi copies.
pub const XOR: u8 = 0x2a;
pub const LDN: u8 = 0x0c;
pub const STN: u8 = 0x0d;

macro_rules! ops {
    ($($name:literal $code:literal $kind:ident $mask:ident $($imm:expr)?;)*) => {
        pub const OPS: &[OpInfo] = &[$(OpInfo {
            name: $name,
            code: $code,
            kind: Kind::$kind,
            mask: ops!(@mask $mask),
            imm: ops!(@imm $($imm)?),
        },)*];
    };
    (@mask N) => { Mask::None };
    (@mask R2) => { Mask::Data };
    (@mask M) => { Mask::Modes(0b111) };
    (@mask Mreq) => { Mask::Modes(0b110) };
    (@mask M01) => { Mask::Modes(0b011) };
    (@mask M1) => { Mask::Modes(0b010) };
    (@imm) => { Imm::None };
    (@imm $imm:expr) => { $imm };
}

use Imm::{Below, Clmul, Cmp, Widths};

ops! {
    "ld" 0x00 Ls N;
    "st" 0x01 St N;
    "ldbc.d" 0x02 Ls M;
    "ldbc.q" 0x03 Ls M;
    "splat.b" 0x04 Ls M;
    "splat.w" 0x05 Ls M;
    "splat.d" 0x06 Ls M;
    "splat.q" 0x07 Ls M;
    "zero" 0x08 R0 N;
    "add.b" 0x10 R3 M;
    "add.w" 0x11 R3 M;
    "add.d" 0x12 R3 M;
    "add.q" 0x13 R3 M;
    "sub.b" 0x14 R3 M;
    "sub.w" 0x15 R3 M;
    "sub.d" 0x16 R3 M;
    "sub.q" 0x17 R3 M;
    "mullo.w" 0x18 R3 M;
    "mullo.d" 0x19 R3 M;
    "mullo.q" 0x1a R3 M;
    "mulwu" 0x1b R3 M;
    "mulws" 0x1c R3 M;
    "mulhi.w" 0x1d R3 M;
    "mulhiu.w" 0x1e R3 M;
    "max.d" 0x20 R3 M;
    "min.d" 0x21 R3 M;
    "maxu.d" 0x22 R3 M;
    "minu.d" 0x23 R3 M;
    "max.w" 0x24 R3 M;
    "min.w" 0x25 R3 M;
    "maxu.q" 0x26 R3 M;
    "minu.q" 0x27 R3 M;
    "and" 0x28 R3 M;
    "or" 0x29 R3 M;
    "xor" 0x2a R3 M;
    "andn" 0x2b R3 M;
    "ternlog" 0x2c A3i M Below(256);
    "maxu.w" 0x2e R3 M;
    "minu.w" 0x2f R3 M;
    "shli.w" 0x30 R2i M Below(16);
    "shli.d" 0x31 R2i M Below(32);
    "shli.q" 0x32 R2i M Below(64);
    "shri.w" 0x33 R2i M Below(16);
    "shri.d" 0x34 R2i M Below(32);
    "shri.q" 0x35 R2i M Below(64);
    "sari.w" 0x36 R2i M Below(16);
    "sari.d" 0x37 R2i M Below(32);
    "sari.q" 0x38 R2i M Below(64);
    "roli.d" 0x39 R2i M Below(32);
    "roli.q" 0x3a R2i M Below(64);
    "shlv.d" 0x3b R3 M;
    "shlv.q" 0x3c R3 M;
    "shrv.d" 0x3d R3 M;
    "shrv.q" 0x3e R3 M;
    "rolv.d" 0x3f R3 M;
    "rolv.q" 0x40 R3 M;
    "cmpeq.b" 0x48 R3 N;
    "cmpeq.d" 0x49 R3 N;
    "cmpeq.q" 0x4a R3 N;
    "cmpgt.d" 0x4b R3 N;
    "cmpgt.q" 0x4c R3 N;
    "cmpgtu.q" 0x4d R3 N;
    "dpbusd" 0x50 A3 M;
    "dpwssd" 0x51 A3 M;
    "maddubs" 0x52 R3 M;
    "maddwd" 0x53 R3 M;
    "madd52lo" 0x54 A3 M;
    "madd52hi" 0x55 A3 M;
    "packs.d" 0x58 R3 M;
    "packs.w" 0x59 R3 M;
    "packus.w" 0x5a R3 M;
    "permd" 0x5b R3 M;
    "permq" 0x5c R3 M;
    "shufb" 0x5d R3 M;
    "perm2.w" 0x5e A3 M;
    "narrowus.d.b" 0x60 R2 M;
    "narrows.d.b" 0x61 R2 M;
    "narrows.d.w" 0x62 R2 M;
    "narrow.d.b" 0x63 R2 M;
    "zx.b.d" 0x64 R2 M;
    "sx.b.d" 0x65 R2 M;
    "zx.w.d" 0x66 R2 M;
    "sx.w.d" 0x67 R2 M;
    "zx.d.q" 0x68 R2 M;
    "sx.d.q" 0x69 R2 M;
    "popcnt.b" 0x6a R2 M;
    "popcnt.q" 0x6b R2 M;
    "hsum.d" 0x70 Sr N;
    "hsum.q" 0x71 Sr N;
    "extract.q" 0x72 Sri N Below(8);
    "extract.d" 0x73 Sri N Below(16);
    "insert.q" 0x74 Ai N Below(8);
    "movemask.b" 0x75 Sr N;
    "clmul" 0x78 R3i N Clmul;
    "mpopcnt" 0xf6 S0 R2;
    "mtzcnt" 0xf7 S0 R2;
    "mlzcnt" 0xf8 S0 R2;
    "blendm" 0x79 R3i R2 Widths(0xf);
    "bcast.q" 0x7a R2i M Below(8);
    "bcast.d" 0x7b R2i M Below(16);
    "bcast.w" 0x7c R2i M Below(32);
    "bcast.b" 0x7d R2i M Below(64);
    "permqi" 0x7e R2i M Below(256);
    "extract256" 0x7f R2i M Below(2);
    "perm2i" 0x5f A3i M Widths(0xf);
    "align.q" 0x80 R3i M Below(8);
    "align.d" 0x81 R3i M Below(16);
    "extract128" 0x82 R2i M Below(4);
    "insert256" 0x83 R3i M Below(2);
    "insert128" 0x84 R3i M Below(4);
    "perm2t" 0x85 A3i M Widths(0b1101);
    "permb" 0x86 R3 M;
    "permw" 0x87 R3 M;
    "shufd" 0x88 R2i M Below(256);
    "shufhw" 0x89 R2i M Below(256);
    "shuflw" 0x8a R2i M Below(256);
    "shuf128" 0x8b R3i M Below(256);
    "shufps" 0x8c R3i M Below(256);
    "alignr" 0x8d R3i M Below(32);
    "bshli" 0x8e R2i N Below(16);
    "bshri" 0x8f R2i N Below(16);
    "unpacklo" 0x90 R3i M Widths(0xf);
    "unpackhi" 0x91 R3i M Widths(0xf);
    "compress" 0x92 R2i Mreq Widths(0xf);
    "expand" 0x93 R2i Mreq Widths(0xf);
    "packus.d" 0x94 R3 M;
    "narrow.d.w" 0x95 R2 M;
    "narrowus.d.w" 0x96 R2 M;
    "narrow.q.b" 0x97 R2 M;
    "narrows.q.b" 0x98 R2 M;
    "narrowus.q.b" 0x99 R2 M;
    "narrow.q.w" 0x9a R2 M;
    "narrows.q.w" 0x9b R2 M;
    "narrowus.q.w" 0x9c R2 M;
    "narrow.q.d" 0x9d R2 M;
    "narrows.q.d" 0x9e R2 M;
    "narrowus.q.d" 0x9f R2 M;
    "narrow.w.b" 0xa0 R2 M;
    "narrows.w.b" 0xa1 R2 M;
    "narrowus.w.b" 0xa2 R2 M;
    "zx.b.w" 0xa3 R2 M;
    "sx.b.w" 0xa4 R2 M;
    "zx.b.q" 0xa5 R2 M;
    "sx.b.q" 0xa6 R2 M;
    "zx.w.q" 0xa7 R2 M;
    "sx.w.q" 0xa8 R2 M;
    "extract.b" 0xa9 Sri N Below(64);
    "extract.w" 0xaa Sri N Below(32);
    "insert.b" 0xab Ai N Below(64);
    "insert.w" 0xac Ai N Below(32);
    "insert.d" 0xad Ai N Below(16);
    "movr.q" 0xae Ls N;
    "mul52lo" 0xaf R3 M;
    "mul52hi" 0xb0 R3 M;
    "abs.q" 0xb1 R2 M;
    "max.q" 0xb2 R3 M;
    "min.q" 0xb3 R3 M;
    "abs.b" 0xb4 R2 M;
    "abs.w" 0xb5 R2 M;
    "abs.d" 0xb6 R2 M;
    "max.b" 0xb7 R3 M;
    "min.b" 0xb8 R3 M;
    "maxu.b" 0xb9 R3 M;
    "minu.b" 0xba R3 M;
    "adds.b" 0xbb R3 M;
    "adds.w" 0xbc R3 M;
    "addus.b" 0xbd R3 M;
    "addus.w" 0xbe R3 M;
    "subs.b" 0xbf R3 M;
    "subs.w" 0xc0 R3 M;
    "subus.b" 0xc1 R3 M;
    "subus.w" 0xc2 R3 M;
    "avg.b" 0xc3 R3 M;
    "avg.w" 0xc4 R3 M;
    "sad" 0xc5 R3 N;
    "mulhrs.w" 0x1f R3 M;
    "dpbusds" 0x56 A3 M;
    "dpwssds" 0x57 A3 M;
    "cmp" 0x46 S3i N Cmp;
    "bitshuf" 0x4f S3 N;
    "test" 0x47 S3i N Widths(0xf);
    "testn" 0x4e S3i N Widths(0xf);
    "movemask.w" 0x76 Sr N;
    "movemask.d" 0x77 Sr N;
    "movemask.q" 0xc6 Sr N;
    "movm.b" 0xc7 R0 R2;
    "movm.w" 0xc8 R0 R2;
    "movm.d" 0xc9 R0 R2;
    "movm.q" 0xca R0 R2;
    "ternlog.d" 0x2d A3i M Below(256);
    "shlv.w" 0x41 R3 M;
    "shrv.w" 0x42 R3 M;
    "sarv.w" 0x43 R3 M;
    "sarv.d" 0x44 R3 M;
    "sarv.q" 0x45 R3 M;
    "shldi.w" 0xce R3i M Below(16);
    "shldi.d" 0xcf R3i M Below(32);
    "shldi.q" 0xd0 R3i M Below(64);
    "shrdi.w" 0xd1 R3i M Below(16);
    "shrdi.d" 0xd2 R3i M Below(32);
    "shrdi.q" 0xd3 R3i M Below(64);
    "shldv.w" 0xd4 A3 M;
    "shldv.d" 0xd5 A3 M;
    "shldv.q" 0xd6 A3 M;
    "shrdv.w" 0xd7 A3 M;
    "shrdv.d" 0xd8 A3 M;
    "shrdv.q" 0xd9 A3 M;
    "conflict.d" 0xcb R2 M;
    "conflict.q" 0xcc R2 M;
    "lzcnt.d" 0x6e R2 M;
    "lzcnt.q" 0x6f R2 M;
    "popcnt.w" 0x6c R2 M;
    "popcnt.d" 0x6d R2 M;
    "multishift" 0xcd R3 M;
    "sha256rnds2" 0xe1 A3 N;
    "sha256msg1" 0xe2 A2 N;
    "sha1rnds4" 0xe4 A2i N Below(4);
    "gf2p8mul" 0xda R3 M;
    "gf2p8affine" 0xdb R3i M Below(256);
    "gf2p8affineinv" 0xdc R3i M Below(256);
    "aesenc" 0xdd R3 N;
    "aesenclast" 0xde R3 N;
    "aesdec" 0xdf R3 N;
    "aesdeclast" 0xe0 R3 N;
    "sha256msg2" 0xe3 A2 N;
    "sha1nexte" 0xe5 A2 N;
    "sha1msg1" 0xe6 A2 N;
    "sha1msg2" 0xe7 A2 N;
    "gather.qq" 0xe8 Ga M Below(4);
    "scatter.qq" 0xec Sc M01 Below(4);
    "ld.m" 0x0a Lsi Mreq Widths(0xf);
    "st.m" 0x0b Sti M1 Widths(0xf);
    "ldn" 0x0c Ldn N;
    "stn" 0x0d Stn N;
    "gather.dd" 0xe9 Ga M Below(4);
    "gather.dq" 0xea Ga M Below(4);
    "gather.qd" 0xeb Ga M Below(4);
    "scatter.dd" 0xed Sc M01 Below(4);
    "scatter.dq" 0xee Sc M01 Below(4);
    "scatter.qd" 0xef Sc M01 Below(4);
    "ld.y" 0x0e Ls N;
    "st.y" 0x0f St N;
    "ld.x" 0xf0 Ls N;
    "st.x" 0xf1 St N;
    "ldbc.b" 0xf2 Ls M;
    "ldbc.w" 0xf3 Ls M;
    "ldbc.x" 0xf4 Ls M;
    "ldbc.y" 0xf5 Ls M;
}

/// `perm2i` with its imm8 for a two-table permute that ties the first table
/// (`perm2t` with imm8 `imm8`, or `perm2.w`), which ties the index instead.
pub fn perm2i_of(op: &OpInfo, imm8: u8) -> Option<(&'static OpInfo, u8)> {
    let imm8 = match op.code {
        0x5e => 1,
        0x85 => imm8,
        _ => return None,
    };
    Some((OPS.iter().find(|op| op.code == 0x5f)?, imm8))
}

fn find(name: &str) -> Option<&'static OpInfo> {
    OPS.iter().find(|op| {
        op.name.len() == name.len()
            && op
                .name
                .bytes()
                .zip(name.bytes())
                .all(|(a, b)| a == b || (a == b'.' && b == b'_'))
    })
}

/// The op a marker names and the mask mode of its suffix, allowed or not.
fn parse_marker(symbol: &str) -> Option<(&'static OpInfo, u8)> {
    let name = symbol.strip_prefix(MARKER_PREFIX)?;
    if let Some(op) = find(name) {
        Some((op, 0))
    } else if let Some(op) = name.strip_suffix("_maskz").and_then(find) {
        Some((op, 2))
    } else {
        Some((name.strip_suffix("_mask").and_then(find)?, 1))
    }
}

/// Looks up a marker symbol (`__svm_add_d`, `__svm_add_d_mask`) in the op
/// table, with the mask mode it names.
pub fn by_marker(symbol: &str) -> Option<(&'static OpInfo, u8)> {
    parse_marker(symbol).filter(|(op, mode)| op.allows_mode(*mode))
}

/// The op of a marker whose mask mode the op does not allow.
pub fn unmasked(symbol: &str) -> Option<&'static OpInfo> {
    parse_marker(symbol)
        .filter(|(op, mode)| !op.allows_mode(*mode))
        .map(|(op, _)| op)
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

/// Sets the mask mode (the `dst` field) of an encoded op.
pub fn with_mode(word: u64, mode: u8) -> u64 {
    word | u64::from(mode) << 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_lookup() {
        let code = |m| by_marker(m).map(|(op, mode)| (op.code, mode));
        assert_eq!(code("__svm_add_d"), Some((0x12, 0)));
        assert_eq!(code("__svm_add_d_mask"), Some((0x12, 1)));
        assert_eq!(code("__svm_add_d_maskz"), Some((0x12, 2)));
        assert_eq!(code("__svm_narrowus_d_b"), Some((0x60, 0)));
        assert_eq!(code("__svm_and"), Some((0x28, 0)));
        assert_eq!(code("__svm_ld_m_maskz"), Some((0x0a, 2)));
        assert_eq!(code("__svm_scatter_qq_mask"), Some((0xec, 1)));
        assert_eq!(code("__svm_mul52lo_maskz"), Some((0xaf, 2)));
        assert_eq!(code("__svm_mulhrs_w_mask"), Some((0x1f, 1)));
        assert_eq!(code("__svm_dpwssds"), Some((0x57, 0)));
        assert_eq!(code("__svm_gather_qd_maskz"), Some((0xeb, 2)));
        assert_eq!(code("__svm_ldbc_x_mask"), Some((0xf4, 1)));
        assert_eq!(code("__svm_st_y"), Some((0x0f, 0)));
        assert!(by_marker("__svm_scatter_dd_maskz").is_none());
        assert!(by_marker("__svm_ld_x_mask").is_none());
        // modes the op does not allow
        assert!(by_marker("__svm_ld_m").is_none());
        assert!(by_marker("__svm_st_m_maskz").is_none());
        assert!(by_marker("__svm_scatter_qq_maskz").is_none());
        assert!(by_marker("__svm_hsum_d_mask").is_none());
        assert!(by_marker("__svm_blendm_mask").is_none());
        assert!(by_marker("__svm_sad_mask").is_none());
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
    fn marker_arguments() {
        let op = |m| by_marker(m).unwrap();
        let (add, _) = op("__svm_add_d");
        assert_eq!((add.vector_args(0), add.vector_args(1)), (2, 3));
        assert!(!add.tied(0) && add.tied(1) && !add.tied(2));
        let (dpbusd, _) = op("__svm_dpbusd");
        assert_eq!((dpbusd.vector_args(1), dpbusd.vector_args(2)), (3, 3));
        assert!(dpbusd.tied(2));
        let (scatter, _) = op("__svm_scatter_qq");
        assert_eq!(scatter.vector_args(1), 2);
        assert!(!scatter.tied(1) && scatter.reads_mask(1));
        let (blendm, _) = op("__svm_blendm");
        assert!(blendm.reads_mask(0) && !blendm.tied(0));
        let (compress, _) = op("__svm_compress_mask");
        assert_eq!(compress.vector_args(1), 2);
        assert!(by_marker("__svm_compress").is_none());
        let (perm2i, mode) = op("__svm_perm2i_mask");
        assert!(perm2i.tied(mode) && perm2i.vector_args(mode) == 3);
        let (perm2t, _) = op("__svm_perm2t_maskz");
        assert_eq!(
            perm2i_of(perm2t, 3).map(|(op, imm8)| (op.code, imm8)),
            Some((0x5f, 3))
        );
        let (perm2w, _) = op("__svm_perm2_w");
        assert_eq!(
            perm2i_of(perm2w, 0).map(|(op, imm8)| (op.code, imm8)),
            Some((0x5f, 1))
        );
        let (extract, _) = op("__svm_extract_b");
        assert!(extract.kind.defines_scalar() && extract.vector_args(0) == 1);
        let (insert, _) = op("__svm_insert_w");
        assert!(insert.tied(0) && insert.kind.has_scalar());
        assert!(by_marker("__svm_bshli_mask").is_none());
        assert!(by_marker("__svm_narrow_q_b_maskz").is_some());
        let (movm, _) = op("__svm_movm_q");
        assert!(movm.reads_mask(0) && movm.vector_args(0) == 0);
        let (shldv, _) = op("__svm_shldv_w_maskz");
        assert!(shldv.tied(2) && shldv.vector_args(2) == 3);
    }

    #[test]
    fn immediates() {
        assert!(Cmp.accepts(0x3e) && !Cmp.accepts(0x03) && !Cmp.accepts(0x40));
        assert!(Widths(0b1101).accepts(0) && !Widths(0b1101).accepts(1));
        assert!(!Widths(0xf).accepts(4));
    }

    #[test]
    fn encoding() {
        // add.d v3 <- v1, v2
        let word = encode(0x12, 0, 3, 1, 2);
        assert_eq!(word.to_le_bytes()[..4], [0x85, 0x20, 0, 0]);
        assert_eq!(word >> 32, 0x12 << 24 | 3 << 10 | 1 << 5 | 2);
        assert_eq!(encode(0x2c, 0xff, 29, 29, 29) >> 32, 0x2cff_77bd);
        assert_eq!(with_mode(word, 2).to_le_bytes()[..2], [0x85, 0x22]);
    }
}
