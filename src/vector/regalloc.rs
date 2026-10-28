//! Register allocation for vector tokens, independent of LLVM.
//!
//! The input is one function in SSA form with critical edges split: blocks
//! with token phis and a list of instructions that use and define tokens.
//! Values are colored greedily in reverse postorder, which follows dominance,
//! so the SSA interference graph is colored with at most max-live registers.
//! Phis are then destructed by parallel copies on their incoming edges, and
//! the result is checked by simulating the register contents.
//!
//! Before coloring, values are spilled to 64-byte stack slots wherever more
//! than 30 would be live, as long as their remaining uses are in the same
//! block (straight-line overflow).

use std::{collections::HashMap, fmt};

/// An SSA vector value.
pub type Tok = u32;
/// A vector register `v0`–`v29`.
pub type Reg = u8;
/// A bit set of registers.
pub type RegSet = u32;

pub const NUM_REGS: u8 = 30;
pub const ALL_REGS: RegSet = (1 << NUM_REGS) - 1;

pub struct Func {
    /// Block 0 is the entry.
    pub blocks: Vec<Block>,
    pub num_tokens: usize,
}

#[derive(Default)]
pub struct Block {
    pub succs: Vec<usize>,
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
}

pub struct Phi {
    pub def: Tok,
    /// `(predecessor block, value)`; `None` is undef.
    pub incoming: Vec<(usize, Option<Tok>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstKind {
    /// A vector op. If `tied`, `uses[0]` is the accumulator and shares the
    /// register of `def`.
    Op { tied: bool },
    /// `def ← cond ? uses[0] : uses[1]`.
    Select,
    /// A call that may overwrite `clobbers`.
    Call { clobbers: RegSet },
    /// Stores `uses[0]` to a stack slot (inserted by spilling).
    Spill { slot: u32 },
    /// Loads `def` from a stack slot (inserted by spilling).
    Reload { slot: u32 },
}

pub struct Inst {
    pub kind: InstKind,
    pub def: Option<Tok>,
    /// `None` is undef (any register may be encoded).
    pub uses: Vec<Option<Tok>>,
    /// Caller-defined position: spills and reloads take the origin of the
    /// instruction they are inserted before.
    pub origin: usize,
    /// Set if the op is one of a run of loads (stores) that becomes one
    /// `ldn` (`stn`) where the registers turn out consecutive.
    pub run: Option<Run>,
}

/// Member `k` of run `id`. A load run's defs, or a store run's `uses[0]`,
/// are given ascending consecutive registers where possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub id: u32,
    pub k: u8,
    pub store: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Mov {
        dst: Reg,
        src: Reg,
    },
    /// Exchange two registers (three `xor`s).
    Swap(Reg, Reg),
}

pub struct Allocation {
    /// Register of each token; `None` for a phi that is never used.
    pub reg: Vec<Option<Reg>>,
    /// Per block and instruction: `mov dst ← src` to emit before it (a tied
    /// accumulator that stays live is copied first).
    pub pre_copy: Vec<Vec<Option<(Reg, Reg)>>>,
    /// Moves on the edge `(pred, succ)` that implement the phis of `succ`.
    pub edge_moves: Vec<(usize, usize, Vec<Move>)>,
    /// Every register written by the function.
    pub written: RegSet,
    /// Number of 64-byte stack slots used by spills.
    pub slots: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum AllocError {
    /// More than 30 values are live at instruction `inst` of `block`
    /// (`None` for the phis at the block's top).
    TooManyLive { block: usize, inst: Option<usize>, live: usize },
    /// The value defined there is live across calls that, together with
    /// the other live values, leave no register.
    ClobberedByCalls { block: usize, inst: Option<usize> },
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::TooManyLive { live, .. } => write!(
                f,
                "{live} vector values are live at the same time, but there are only {NUM_REGS} vector registers"
            ),
            AllocError::ClobberedByCalls { .. } => write!(
                f,
                "a vector value is live across calls to functions that use the remaining vector registers"
            ),
        }
    }
}

/// A dense bit set over tokens.
#[derive(Clone, PartialEq, Eq)]
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Self {
        Bits(vec![0; n.div_ceil(64)])
    }
    fn insert(&mut self, t: Tok) {
        self.0[t as usize / 64] |= 1 << (t % 64);
    }
    fn remove(&mut self, t: Tok) {
        self.0[t as usize / 64] &= !(1 << (t % 64));
    }
    fn contains(&self, t: Tok) -> bool {
        self.0[t as usize / 64] & (1 << (t % 64)) != 0
    }
    fn union(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= b;
        }
    }
    fn iter(&self) -> impl Iterator<Item = Tok> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &w)| {
            (0..64)
                .filter(move |b| w & (1 << b) != 0)
                .map(move |b| (i * 64 + b) as Tok)
        })
    }
}

/// Blocks reachable from the entry, in reverse postorder.
pub fn reverse_postorder(func: &Func) -> Vec<usize> {
    let mut seen = vec![false; func.blocks.len()];
    let mut order = Vec::new();
    let mut stack = vec![(0, 0)];
    seen[0] = true;
    while let Some((block, next)) = stack.pop() {
        if let Some(&succ) = func.blocks[block].succs.get(next) {
            stack.push((block, next + 1));
            if !seen[succ] {
                seen[succ] = true;
                stack.push((succ, 0));
            }
        } else {
            order.push(block);
        }
    }
    order.reverse();
    order
}

struct Liveness {
    /// Live at the top of each block, including its live phis.
    top: Vec<Bits>,
    /// Live after each instruction, per block.
    after: Vec<Vec<Bits>>,
}

fn liveness(func: &Func, rpo: &[usize]) -> Liveness {
    let n = func.num_tokens;
    let mut top = vec![Bits::new(n); func.blocks.len()];
    let live_out = |top: &[Bits], b: usize| {
        let mut out = Bits::new(n);
        for &s in &func.blocks[b].succs {
            let mut live = top[s].clone();
            for phi in &func.blocks[s].phis {
                live.remove(phi.def);
            }
            out.union(&live);
            for phi in &func.blocks[s].phis {
                if top[s].contains(phi.def) {
                    for &(pred, v) in &phi.incoming {
                        if let (true, Some(v)) = (pred == b, v) {
                            out.insert(v);
                        }
                    }
                }
            }
        }
        out
    };
    let backward =
        |mut live: Bits, block: &Block, after: Option<&mut Vec<Bits>>| {
            let mut record = Vec::with_capacity(block.insts.len());
            for inst in block.insts.iter().rev() {
                record.push(live.clone());
                if let Some(d) = inst.def {
                    live.remove(d);
                }
                for &u in inst.uses.iter().flatten() {
                    live.insert(u);
                }
            }
            if let Some(after) = after {
                record.reverse();
                *after = record;
            }
            live
        };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().rev() {
            let live = backward(live_out(&top, b), &func.blocks[b], None);
            if live != top[b] {
                top[b] = live;
                changed = true;
            }
        }
    }
    let mut after = vec![Vec::new(); func.blocks.len()];
    for &b in rpo {
        backward(live_out(&top, b), &func.blocks[b], Some(&mut after[b]));
    }
    Liveness { top, after }
}

fn bit(r: Reg) -> RegSet {
    1 << r
}

/// Picks the first preferred register in `free`, else the lowest one.
fn pick(free: RegSet, prefs: impl IntoIterator<Item = Reg>) -> Option<Reg> {
    prefs
        .into_iter()
        .find(|&r| free & bit(r) != 0)
        .or_else(|| (free != 0).then(|| free.trailing_zeros() as Reg))
}

/// Inserts spills and reloads until at most `NUM_REGS` values are live at
/// every definition. A victim must not be used by the instruction at the
/// overflow and must have all its remaining uses later in the same block; of
/// those, the one used furthest away is spilled. Returns the number of slots.
fn spill(func: &mut Func) -> Result<u32, AllocError> {
    let mut slot_of: HashMap<Tok, u32> = HashMap::new();
    let mut slots = 0;
    let budget: usize = func.blocks.iter().map(|b| b.insts.len()).sum();
    for _ in 0..=4 * budget {
        let rpo = reverse_postorder(func);
        let live = liveness(func, &rpo);
        let mut overflow = None;
        'search: for &b in &rpo {
            let at_top = live.top[b].iter().count();
            if at_top > NUM_REGS as usize {
                return Err(AllocError::TooManyLive {
                    block: b,
                    inst: None,
                    live: at_top,
                });
            }
            for (i, (inst, after)) in
                func.blocks[b].insts.iter().zip(&live.after[b]).enumerate()
            {
                let Some(d) = inst.def else { continue };
                let pressure = after.iter().filter(|&t| t != d).count() + 1;
                if pressure > NUM_REGS as usize {
                    overflow = Some((b, i, pressure));
                    break 'search;
                }
            }
        }
        let Some((b, i, pressure)) = overflow else { return Ok(slots) };
        let block = &func.blocks[b];
        let live_out = live.after[b].last().unwrap();
        let inst = &block.insts[i];
        let next_use = |v: Tok| {
            let later = &block.insts[i + 1..];
            later
                .iter()
                .position(|x| x.uses.contains(&Some(v)))
                .map(|p| i + 1 + p)
        };
        let victim = live.after[b][i]
            .iter()
            .filter(|&v| {
                inst.def != Some(v)
                    && !inst.uses.contains(&Some(v))
                    && !live_out.contains(v)
            })
            .filter_map(|v| next_use(v).map(|j| (j, v)))
            .max();
        let Some((j, v)) = victim else {
            return Err(AllocError::TooManyLive {
                block: b,
                inst: Some(i),
                live: pressure,
            });
        };

        // Uses from `j` on read the reloaded copy `r`.
        let r = func.num_tokens as Tok;
        func.num_tokens += 1;
        let block = &mut func.blocks[b];
        for later in &mut block.insts[j..] {
            for u in later.uses.iter_mut().filter(|u| **u == Some(v)) {
                *u = Some(r);
            }
        }
        // A value that was reloaded is still in its slot: no store needed.
        let slot = match slot_of.get(&v) {
            Some(&slot) => slot,
            None => {
                slots += 1;
                slots - 1
            }
        };
        slot_of.insert(r, slot);
        let reload = Inst {
            kind: InstKind::Reload { slot },
            def: Some(r),
            uses: Vec::new(),
            origin: block.insts[j].origin,
            run: None,
        };
        block.insts.insert(j, reload);
        if slot_of.insert(v, slot).is_none() {
            let store = Inst {
                kind: InstKind::Spill { slot },
                def: None,
                uses: vec![Some(v)],
                origin: block.insts[i].origin,
                run: None,
            };
            block.insts.insert(i, store);
        }
    }
    Err(AllocError::TooManyLive {
        block: 0,
        inst: None,
        live: NUM_REGS as usize + 1,
    })
}

/// Spills where needed, then assigns registers. Spills and reloads are
/// inserted into `func`; error positions refer to the modified `func`.
pub fn allocate(func: &mut Func) -> Result<Allocation, AllocError> {
    let slots = spill(func)?;
    let func = &*func;
    let rpo = reverse_postorder(func);
    let live = liveness(func, &rpo);
    let n = func.num_tokens;

    // Registers each token must avoid because it is live across a call.
    let mut forbidden = vec![0 as RegSet; n];
    // Phi-related tokens: sharing a register saves a move.
    let mut affinity: Vec<Vec<Tok>> = vec![Vec::new(); n];
    for &b in &rpo {
        let block = &func.blocks[b];
        for (inst, after) in block.insts.iter().zip(&live.after[b]) {
            if let InstKind::Call { clobbers } = inst.kind {
                for t in after.iter() {
                    forbidden[t as usize] |= clobbers;
                }
            }
        }
        for phi in &block.phis {
            for &(_, v) in &phi.incoming {
                if let Some(v) = v {
                    affinity[phi.def as usize].push(v);
                    affinity[v as usize].push(phi.def);
                }
            }
        }
    }

    // Store runs: the value of each member, and the members of each value.
    let mut stored: HashMap<(u32, u8), Tok> = HashMap::new();
    let mut stores_of: HashMap<Tok, Vec<(u32, u8)>> = HashMap::new();
    for block in &func.blocks {
        for inst in &block.insts {
            if let (Some(run), Some(&Some(t))) = (inst.run, inst.uses.first())
                && run.store
            {
                stored.insert((run.id, run.k), t);
                stores_of.entry(t).or_default().push((run.id, run.k));
            }
        }
    }

    let mut reg: Vec<Option<Reg>> = vec![None; n];
    let mut pre_copy: Vec<Vec<Option<(Reg, Reg)>>> =
        func.blocks.iter().map(|b| vec![None; b.insts.len()]).collect();
    let mut written: RegSet = 0;
    let prefs = |reg: &[Option<Reg>], t: Tok| -> Vec<Reg> {
        // Next to the values stored before and after it in a store run.
        let neighbours =
            stores_of.get(&t).into_iter().flatten().flat_map(|&(id, k)| {
                let at = |k: Option<u8>, step: i8| {
                    let r = reg[*stored.get(&(id, k?))? as usize]?;
                    r.checked_add_signed(step).filter(|&r| r < NUM_REGS)
                };
                [at(k.checked_sub(1), 1), at(k.checked_add(1), -1)]
            });
        neighbours
            .flatten()
            .chain(
                affinity[t as usize].iter().filter_map(|&a| reg[a as usize]),
            )
            .collect()
    };
    let regs_of = |reg: &[Option<Reg>], live: &Bits| -> RegSet {
        live.iter().filter_map(|t| reg[t as usize]).fold(0, |s, r| s | bit(r))
    };
    let no_reg = |live: usize, block, inst| {
        if live > NUM_REGS as usize {
            AllocError::TooManyLive { block, inst, live }
        } else {
            AllocError::ClobberedByCalls { block, inst }
        }
    };

    for &b in &rpo {
        let block = &func.blocks[b];
        let top = &live.top[b];
        let mut occupied = 0;
        for t in top.iter() {
            if !block.phis.iter().any(|p| p.def == t) {
                occupied |= bit(reg[t as usize]
                    .expect("live-in value dominates the block"));
            }
        }
        for phi in block.phis.iter().filter(|p| top.contains(p.def)) {
            let free = ALL_REGS & !occupied & !forbidden[phi.def as usize];
            let r = pick(free, prefs(&reg, phi.def))
                .ok_or_else(|| no_reg(top.iter().count(), b, None))?;
            reg[phi.def as usize] = Some(r);
            occupied |= bit(r);
            written |= bit(r);
        }
        for (i, (inst, after)) in
            block.insts.iter().zip(&live.after[b]).enumerate()
        {
            let Some(d) = inst.def else { continue };
            let live_regs = regs_of(&reg, after);
            let use_regs = inst.uses.iter().flatten().fold(0, |s, &u| {
                s | bit(reg[u as usize].expect("use is dominated by its def"))
            });
            let free = ALL_REGS & !live_regs & !forbidden[d as usize];
            let err =
                || no_reg(live_regs.count_ones() as usize + 1, b, Some(i));
            let r = match (inst.kind, inst.uses.first()) {
                (InstKind::Op { tied: true }, Some(&Some(acc))) => {
                    let acc_reg = reg[acc as usize].unwrap();
                    if free & bit(acc_reg) != 0 {
                        acc_reg
                    } else {
                        // The accumulator stays live (or its register is
                        // clobbered later): copy it to a register that no
                        // operand of this op is read from.
                        let r = pick(free & !use_regs, prefs(&reg, d))
                            .ok_or_else(err)?;
                        pre_copy[b][i] = Some((r, acc_reg));
                        r
                    }
                }
                (InstKind::Select, _) => {
                    let ops = inst
                        .uses
                        .iter()
                        .flatten()
                        .map(|&u| reg[u as usize].unwrap());
                    pick(free, ops.chain(prefs(&reg, d))).ok_or_else(err)?
                }
                _ => {
                    let run = load_run(func, &reg, &forbidden, b, i, free);
                    pick(free, run.into_iter().chain(prefs(&reg, d)))
                        .ok_or_else(err)?
                }
            };
            reg[d as usize] = Some(r);
            written |= bit(r);
        }
    }

    // Phi destruction: parallel copies on each incoming edge.
    let mut edge_moves = Vec::new();
    for &b in &rpo {
        let block = &func.blocks[b];
        let mut preds: Vec<usize> = block
            .phis
            .iter()
            .flat_map(|p| p.incoming.iter().map(|&(pred, _)| pred))
            .collect();
        preds.sort_unstable();
        preds.dedup();
        for pred in preds {
            let copies = block
                .phis
                .iter()
                .filter_map(|phi| {
                    let dst = reg[phi.def as usize]?;
                    let (_, v) =
                        phi.incoming.iter().find(|&&(p, _)| p == pred)?;
                    Some((dst, reg[(*v)? as usize]?))
                })
                .collect();
            let moves = sequentialize(copies);
            for m in &moves {
                written |= match *m {
                    Move::Mov { dst, .. } => bit(dst),
                    Move::Swap(a, b) => bit(a) | bit(b),
                };
            }
            if !moves.is_empty() {
                edge_moves.push((pred, b, moves));
            }
        }
    }

    let alloc = Allocation { reg, pre_copy, edge_moves, written, slots };
    verify(func, &rpo, &alloc);
    Ok(alloc)
}

/// The preferred register of the def of load run member `i` of block `b`:
/// one above the previous member, or, for the first member of a sequence,
/// the lowest register that starts a window free for the whole sequence.
fn load_run(
    func: &Func,
    reg: &[Option<Reg>],
    forbidden: &[RegSet],
    b: usize,
    i: usize,
    free: RegSet,
) -> Option<Reg> {
    let insts = &func.blocks[b].insts;
    insts[i].run.filter(|r| !r.store)?;
    let follows = |prev: &Inst, next: &Inst| {
        matches!((prev.run, next.run), (Some(p), Some(n))
            if p.id == n.id && p.k + 1 == n.k && prev.def.is_some())
    };
    if i > 0 && follows(&insts[i - 1], &insts[i]) {
        return reg[insts[i - 1].def? as usize].map(|r| r + 1);
    }
    let mut defs = vec![insts[i].def?];
    for pair in insts[i..].windows(2) {
        match pair[1].def {
            Some(d) if follows(&pair[0], &pair[1]) => defs.push(d),
            _ => break,
        }
    }
    (0..=NUM_REGS.checked_sub(defs.len() as u8)?).find(|&r| {
        defs.iter().enumerate().all(|(j, &t)| {
            let r = r + j as u8;
            free & bit(r) != 0 && forbidden[t as usize] & bit(r) == 0
        })
    })
}

/// Orders a parallel copy `dst ← src` (distinct destinations) into moves,
/// breaking cycles with swaps.
pub fn sequentialize(mut copies: Vec<(Reg, Reg)>) -> Vec<Move> {
    let mut moves = Vec::new();
    loop {
        copies.retain(|&(dst, src)| dst != src);
        if copies.is_empty() {
            return moves;
        }
        let ready = copies
            .iter()
            .position(|&(dst, _)| copies.iter().all(|&(_, src)| src != dst));
        match ready {
            Some(i) => {
                let (dst, src) = copies.remove(i);
                moves.push(Move::Mov { dst, src });
            }
            None => {
                // Only cycles remain: every register is read exactly once.
                let (dst, src) = copies.remove(0);
                moves.push(Move::Swap(dst, src));
                for copy in &mut copies {
                    if copy.1 == dst {
                        copy.1 = src;
                    }
                }
            }
        }
    }
}

/// Simulates which token every register holds and asserts that each
/// instruction reads the tokens it names. A failure is a bug in the pass.
fn verify(func: &Func, rpo: &[usize], alloc: &Allocation) {
    const TOP: Tok = Tok::MAX;
    const BAD: Tok = Tok::MAX - 1;
    type State = [Tok; NUM_REGS as usize];
    let reg = |t: Tok| alloc.reg[t as usize].unwrap() as usize;

    let mut entry = vec![[TOP; NUM_REGS as usize]; func.blocks.len()];
    entry[0] = [BAD; NUM_REGS as usize];
    let run = |entry: &mut Vec<State>, check: bool| -> bool {
        let mut changed = false;
        for &b in rpo {
            let block = &func.blocks[b];
            let mut state = entry[b];
            for (i, inst) in block.insts.iter().enumerate() {
                if let Some((dst, src)) = alloc.pre_copy[b][i] {
                    state[dst as usize] = state[src as usize];
                }
                if check {
                    for (k, u) in inst.uses.iter().enumerate() {
                        let Some(u) = *u else { continue };
                        let r = match (inst.kind, k, inst.def) {
                            (InstKind::Op { tied: true }, 0, Some(d)) => {
                                reg(d)
                            }
                            _ => reg(u),
                        };
                        assert_eq!(
                            state[r], u,
                            "block {b} inst {i} reads v{r}"
                        );
                    }
                }
                if let InstKind::Call { clobbers } = inst.kind {
                    for r in 0..NUM_REGS {
                        if clobbers & bit(r) != 0 {
                            state[r as usize] = BAD;
                        }
                    }
                }
                if let Some(d) = inst.def {
                    state[reg(d)] = d;
                }
            }
            for &s in &block.succs {
                let mut out = state;
                for (_, _, moves) in
                    alloc.edge_moves.iter().filter(|e| e.0 == b && e.1 == s)
                {
                    for m in moves {
                        match *m {
                            Move::Mov { dst, src } => {
                                out[dst as usize] = out[src as usize]
                            }
                            Move::Swap(x, y) => {
                                out.swap(x as usize, y as usize)
                            }
                        }
                    }
                }
                let mut renamed = out;
                for phi in &func.blocks[s].phis {
                    let Some(r) = alloc.reg[phi.def as usize] else {
                        continue;
                    };
                    if check {
                        for &(pred, v) in &phi.incoming {
                            if let (true, Some(v)) = (pred == b, v) {
                                assert_eq!(
                                    out[r as usize], v,
                                    "edge {b}->{s} phi in v{r}"
                                );
                            }
                        }
                    }
                    renamed[r as usize] = phi.def;
                }
                for (e, &v) in entry[s].iter_mut().zip(&renamed) {
                    let met = if *e == TOP || *e == v { v } else { BAD };
                    if met != *e {
                        *e = met;
                        changed = true;
                    }
                }
            }
        }
        changed
    };
    while run(&mut entry, false) {}
    run(&mut entry, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(def: Option<Tok>, uses: &[Option<Tok>]) -> Inst {
        Inst {
            kind: InstKind::Op { tied: false },
            def,
            uses: uses.to_vec(),
            origin: 0,
            run: None,
        }
    }
    fn acc(def: Tok, uses: &[Option<Tok>]) -> Inst {
        Inst {
            kind: InstKind::Op { tied: true },
            def: Some(def),
            uses: uses.to_vec(),
            origin: 0,
            run: None,
        }
    }
    fn call(clobbers: RegSet) -> Inst {
        Inst {
            kind: InstKind::Call { clobbers },
            def: None,
            uses: vec![],
            origin: 0,
            run: None,
        }
    }
    fn block(succs: &[usize], phis: Vec<Phi>, insts: Vec<Inst>) -> Block {
        Block { succs: succs.to_vec(), phis, insts }
    }

    #[test]
    fn straight_line_reuses_dead_registers() {
        // t0 = ld; t1 = ld; t2 = add t0 t1; st t2
        let mut f = Func {
            blocks: vec![block(
                &[],
                vec![],
                vec![
                    op(Some(0), &[]),
                    op(Some(1), &[]),
                    op(Some(2), &[Some(0), Some(1)]),
                    op(None, &[Some(2)]),
                ],
            )],
            num_tokens: 3,
        };
        let a = allocate(&mut f).unwrap();
        assert_eq!(a.reg, [Some(0), Some(1), Some(0)]);
        assert!(a.edge_moves.is_empty());
    }

    #[test]
    fn straight_line_overflow_spills() {
        // 31 values defined, then each used once: one must be spilled.
        let n = 31;
        let mut insts: Vec<Inst> = (0..n).map(|t| op(Some(t), &[])).collect();
        insts.extend((0..n).map(|t| op(None, &[Some(t)])));
        let mut f = Func {
            blocks: vec![block(&[], vec![], insts)],
            num_tokens: n as usize,
        };
        let a = allocate(&mut f).unwrap();
        assert_eq!(a.slots, 1);
        let insts = &f.blocks[0].insts;
        // t29 is used last, so it is the one stored before t30's definition
        // and reloaded (as t31) right before its use.
        assert_eq!(insts[30].kind, InstKind::Spill { slot: 0 });
        assert_eq!(insts[30].uses, [Some(29)]);
        assert_eq!(insts[31].def, Some(30));
        assert_eq!(insts[61].kind, InstKind::Reload { slot: 0 });
        assert_eq!(insts[61].def, Some(31));
        assert_eq!(insts[62].uses, [Some(31)]);
        assert_eq!(insts.len(), 64);
    }

    #[test]
    fn too_many_live_across_blocks() {
        // 31 values live out of the entry block cannot be spilled.
        let n = 31;
        let insts: Vec<Inst> = (0..n).map(|t| op(Some(t), &[])).collect();
        let uses: Vec<Inst> = (0..n).map(|t| op(None, &[Some(t)])).collect();
        let mut f = Func {
            blocks: vec![block(&[1], vec![], insts), block(&[], vec![], uses)],
            num_tokens: n as usize,
        };
        assert_eq!(
            allocate(&mut f).err(),
            Some(AllocError::TooManyLive {
                block: 0,
                inst: Some(30),
                live: 31
            })
        );
    }

    #[test]
    fn thirty_live_fit() {
        let n = 30;
        let mut insts: Vec<Inst> = (0..n).map(|t| op(Some(t), &[])).collect();
        insts.push(op(None, &(0..n).map(Some).collect::<Vec<_>>()));
        let mut f = Func {
            blocks: vec![block(&[], vec![], insts)],
            num_tokens: n as usize,
        };
        assert_eq!(allocate(&mut f).unwrap().written, ALL_REGS);
    }

    #[test]
    fn loop_carried_accumulator_needs_no_moves() {
        // b0: t0 = zero          -> b1
        // b1: t1 = phi(t0, t3); t2 = ld; t3 = dpbusd t1 t2 t2  -> b1, b2
        // b2: st t3
        let mut f = Func {
            blocks: vec![
                block(&[1], vec![], vec![op(Some(0), &[])]),
                block(
                    &[1, 2],
                    vec![Phi {
                        def: 1,
                        incoming: vec![(0, Some(0)), (1, Some(3))],
                    }],
                    vec![
                        op(Some(2), &[]),
                        acc(3, &[Some(1), Some(2), Some(2)]),
                    ],
                ),
                block(&[], vec![], vec![op(None, &[Some(3)])]),
            ],
            num_tokens: 4,
        };
        let a = allocate(&mut f).unwrap();
        assert!(a.edge_moves.is_empty(), "{:?}", a.edge_moves);
        assert_eq!(a.reg[0], a.reg[1]);
        assert_eq!(a.reg[1], a.reg[3]);
        assert!(a.pre_copy.iter().flatten().all(Option::is_none));
    }

    #[test]
    fn accumulator_live_after_is_copied() {
        // t0 = ld; t1 = ld; t2 = dpbusd t0 t1 t1; st t0; st t2
        let mut f = Func {
            blocks: vec![block(
                &[],
                vec![],
                vec![
                    op(Some(0), &[]),
                    op(Some(1), &[]),
                    acc(2, &[Some(0), Some(1), Some(1)]),
                    op(None, &[Some(0)]),
                    op(None, &[Some(2)]),
                ],
            )],
            num_tokens: 3,
        };
        let a = allocate(&mut f).unwrap();
        let (dst, src) = a.pre_copy[0][2].unwrap();
        assert_eq!(src, a.reg[0].unwrap());
        assert_eq!(dst, a.reg[2].unwrap());
        assert_ne!(dst, a.reg[1].unwrap());
    }

    #[test]
    fn swapped_phis_use_swap() {
        // b0: t0 = ld; t1 = ld -> b1
        // b1: t2 = phi(t0, t3); t3 = phi(t1, t2); use both -> b1, b2
        let mut f = Func {
            blocks: vec![
                block(&[1], vec![], vec![op(Some(0), &[]), op(Some(1), &[])]),
                block(
                    &[2, 1],
                    vec![
                        Phi {
                            def: 2,
                            incoming: vec![(0, Some(0)), (1, Some(3))],
                        },
                        Phi {
                            def: 3,
                            incoming: vec![(0, Some(1)), (1, Some(2))],
                        },
                    ],
                    vec![op(None, &[Some(2), Some(3)])],
                ),
                block(&[], vec![], vec![]),
            ],
            num_tokens: 4,
        };
        let a = allocate(&mut f).unwrap();
        assert_eq!(a.edge_moves.len(), 1);
        assert!(matches!(a.edge_moves[0].2[..], [Move::Swap(..)]));
    }

    #[test]
    fn if_else_merge() {
        // b0: t0 = ld -> b1, b2
        // b1: t1 = add t0 t0 -> b3
        // b2: t2 = sub t0 t0 -> b3
        // b3: t3 = phi(t1, t2); st t3; st t0
        let mut f = Func {
            blocks: vec![
                block(&[1, 2], vec![], vec![op(Some(0), &[])]),
                block(&[3], vec![], vec![op(Some(1), &[Some(0), Some(0)])]),
                block(&[3], vec![], vec![op(Some(2), &[Some(0), Some(0)])]),
                block(
                    &[],
                    vec![Phi {
                        def: 3,
                        incoming: vec![(1, Some(1)), (2, Some(2))],
                    }],
                    vec![op(None, &[Some(3)]), op(None, &[Some(0)])],
                ),
            ],
            num_tokens: 4,
        };
        let a = allocate(&mut f).unwrap();
        assert!(a.edge_moves.is_empty());
        assert_ne!(a.reg[3], a.reg[0]);
    }

    #[test]
    fn call_clobbers_are_avoided() {
        // t0 = ld; call (clobbers v0..v9); st t0
        let mut f = Func {
            blocks: vec![block(
                &[],
                vec![],
                vec![op(Some(0), &[]), call(0x3ff), op(None, &[Some(0)])],
            )],
            num_tokens: 1,
        };
        assert_eq!(allocate(&mut f).unwrap().reg[0], Some(10));

        let mut f = Func {
            blocks: vec![block(
                &[],
                vec![],
                vec![op(Some(0), &[]), call(ALL_REGS), op(None, &[Some(0)])],
            )],
            num_tokens: 1,
        };
        assert_eq!(
            allocate(&mut f).err(),
            Some(AllocError::ClobberedByCalls { block: 0, inst: Some(0) })
        );
    }

    fn member(mut inst: Inst, id: u32, k: u8, store: bool) -> Inst {
        inst.run = Some(Run { id, k, store });
        inst
    }

    #[test]
    fn load_runs_get_a_free_window() {
        // t0, t1, t2 = ld; use t1; t3, t4, t5 = run of loads; use all
        let mut insts = vec![
            op(Some(0), &[]),
            op(Some(1), &[]),
            op(Some(2), &[]),
            op(None, &[Some(1)]),
        ];
        insts.extend(
            (0..3).map(|k| member(op(Some(3 + k), &[]), 0, k as u8, false)),
        );
        insts.push(op(None, &[0, 2, 3, 4, 5].map(Some)));
        let mut f =
            Func { blocks: vec![block(&[], vec![], insts)], num_tokens: 6 };
        let a = allocate(&mut f).unwrap();
        // v1 is free but too small a hole
        assert_eq!(a.reg[3..], [Some(3), Some(4), Some(5)]);
    }

    #[test]
    fn store_runs_prefer_consecutive_values() {
        // t0, t1, t2 = ld; t3 = op t2; use t0; t4 = op t1; store run t3, t4
        let insts = vec![
            op(Some(0), &[]),
            op(Some(1), &[]),
            op(Some(2), &[]),
            op(Some(3), &[Some(2)]),
            op(None, &[Some(0)]),
            op(Some(4), &[Some(1)]),
            member(op(None, &[Some(3)]), 0, 0, true),
            member(op(None, &[Some(4)]), 0, 1, true),
        ];
        let mut f =
            Func { blocks: vec![block(&[], vec![], insts)], num_tokens: 5 };
        let a = allocate(&mut f).unwrap();
        // v0 and v1 are free at t4, but v3 follows t3
        assert_eq!(a.reg[3..], [Some(2), Some(3)]);
    }

    #[test]
    fn sequentialize_parallel_copies() {
        use Move::*;
        assert_eq!(
            sequentialize(vec![(1, 0), (2, 1)]),
            [Mov { dst: 2, src: 1 }, Mov { dst: 1, src: 0 }]
        );
        assert_eq!(sequentialize(vec![(1, 0), (0, 1)]), [Swap(1, 0)]);
        // 3-cycle plus a fan-out reading from the cycle.
        let moves = sequentialize(vec![(0, 1), (1, 2), (2, 0), (5, 0)]);
        let mut regs: Vec<u32> = (0..6).collect();
        for m in &moves {
            match *m {
                Mov { dst, src } => regs[dst as usize] = regs[src as usize],
                Swap(a, b) => regs.swap(a as usize, b as usize),
            }
        }
        assert_eq!(regs, [1, 2, 0, 3, 4, 0]);
    }
}
