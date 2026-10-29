"""
Fog-of-War Chess with a Dempster-Shafer uncertainty estimator.

Run with:
    pip install -r requirements.txt
    streamlit run app.py

Layout of this file
-------------------
1. Frame of discernment + MassFunction   (DST algebra: Dempster's rule, Bel, Pl, BetP)
2. Fog-of-war rules                      (visibility, piece movement on bitboards)
3. PlayerKnowledge                       (each player's private memory / information set)
4. ObservationContext                    (the ONLY thing estimators may read - no peeking)
5. Evidence sources + estimators         (modular: add a class, register it, done)
6. Game controller                       (session-state backed, hot-seat 2 players)
7. Streamlit UI                          (board HTML with CSS hover tooltips, sidebar, panels)
"""
from __future__ import annotations

import hashlib
import html
import math
import time
from pathlib import Path
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import accumulate
from random import Random
from typing import Dict, List, Optional, Tuple

import chess
import streamlit as st
import streamlit.components.v1 as components

# =============================================================================
# 1. FRAME OF DISCERNMENT & MASS FUNCTIONS
# =============================================================================
# Hidden squares can only hold an OPPONENT piece or nothing (your own pieces are
# always visible), so colour is not part of the frame.
#   Theta = {K, Q, R, B, N, P, Em}
# Subsets of Theta are encoded as 7-bit integers, so intersection is a bitwise AND.

HYPOTHESES = ["K", "Q", "R", "B", "N", "P", "Em"]
DISPLAY_ORDER = ["K", "Q", "B", "R", "N", "P", "Em"]
HYP_NAMES = {"K": "King", "Q": "Queen", "R": "Rook", "B": "Bishop",
             "N": "Knight", "P": "Pawn", "Em": "Empty"}
BIT = {h: 1 << i for i, h in enumerate(HYPOTHESES)}
THETA = (1 << len(HYPOTHESES)) - 1
OCCUPIED = THETA & ~BIT["Em"]                  # "some piece is here"
NON_KING = OCCUPIED & ~BIT["K"]

PT_TO_H = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
           chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}
H_TO_PT = {v: k for k, v in PT_TO_H.items()}
INITIAL_COUNT = {chess.KING: 1, chess.QUEEN: 1, chess.ROOK: 2,
                 chess.BISHOP: 2, chess.KNIGHT: 2, chess.PAWN: 8}


def mask_label(mask: int) -> str:
    if mask == THETA:
        return "Θ"
    if mask == OCCUPIED:
        return "{any piece}"
    return "{" + ",".join(h for h in HYPOTHESES if mask & BIT[h]) + "}"


class TotalConflict(Exception):
    """Raised when two bodies of evidence are fully contradictory (K = 1)."""


class MassFunction:
    """A basic probability assignment m: 2^Theta -> [0,1] with m(empty set) = 0.

    Any mass not explicitly assigned goes to Theta (total ignorance), which is
    how DST expresses "I don't know" instead of spreading it uniformly.
    """

    __slots__ = ("m",)

    def __init__(self, masses: Optional[Dict[int, float]] = None):
        self.m: Dict[int, float] = {}
        for focal, v in (masses or {}).items():
            if focal and v > 1e-12:
                self.m[focal] = self.m.get(focal, 0.0) + v
        total = sum(self.m.values())
        if total > 1.0 + 1e-9:                      # defensive renormalisation
            self.m = {k: v / total for k, v in self.m.items()}
        elif total < 1.0 - 1e-12:
            self.m[THETA] = self.m.get(THETA, 0.0) + (1.0 - total)

    @classmethod
    def vacuous(cls) -> "MassFunction":
        return cls({THETA: 1.0})

    @classmethod
    def simple(cls, focal: int, weight: float) -> "MassFunction":
        """Simple support function: `weight` on one set, the rest on Theta."""
        return cls({focal: min(1.0, max(0.0, weight))})

    def combine(self, other: "MassFunction") -> Tuple["MassFunction", float]:
        """Dempster's rule of combination. Returns (m1 (+) m2, conflict K)."""
        joint: Dict[int, float] = {}
        conflict = 0.0
        for a, ma in self.m.items():
            for b, mb in other.m.items():
                c = a & b
                if c:
                    joint[c] = joint.get(c, 0.0) + ma * mb
                else:
                    conflict += ma * mb
        if conflict >= 1.0 - 1e-9:
            raise TotalConflict()
        norm = 1.0 / (1.0 - conflict)
        out = MassFunction.__new__(MassFunction)
        out.m = {k: v * norm for k, v in joint.items()}
        return out, conflict

    def bel(self, mask: int) -> float:
        """Belief: total mass of evidence that implies `mask` (focal set is a subset)."""
        return sum(v for k, v in self.m.items() if k & ~mask == 0)

    def pl(self, mask: int) -> float:
        """Plausibility: total mass of evidence that does not rule `mask` out."""
        return sum(v for k, v in self.m.items() if k & mask)

    def betp(self, h: str) -> float:
        """Pignistic probability (Smets): split each focal mass evenly over its members."""
        bit = BIT[h]
        return sum(v / k.bit_count() for k, v in self.m.items() if k & bit)


# =============================================================================
# 2. FOG-OF-WAR RULES (bitboard helpers)
# =============================================================================

def sq_bb(sq: int) -> int:
    return chess.BB_SQUARES[sq]


def squares_of(mask: int) -> List[int]:
    return list(chess.scan_forward(mask))


def pawn_pushes(color: chess.Color, sq: int, occ: int) -> int:
    """Non-capturing pawn moves on a board with occupancy `occ`."""
    step = 8 if color == chess.WHITE else -8
    one = sq + step
    if not 0 <= one < 64 or occ & sq_bb(one):
        return 0
    mask = sq_bb(one)
    start_rank = 1 if color == chess.WHITE else 6
    if chess.square_rank(sq) == start_rank:
        two = one + step
        if not occ & sq_bb(two):
            mask |= sq_bb(two)
    return mask


def piece_reach(pt: int, color: chess.Color, sq: int, occ: int) -> int:
    """Squares a piece could move to without capturing, given occupancy `occ`."""
    if pt == chess.PAWN:
        return pawn_pushes(color, sq, occ)
    if pt == chess.KNIGHT:
        att = chess.BB_KNIGHT_ATTACKS[sq]
    elif pt == chess.KING:
        att = chess.BB_KING_ATTACKS[sq]
    else:
        att = 0
        if pt in (chess.BISHOP, chess.QUEEN):
            att |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
        if pt in (chess.ROOK, chess.QUEEN):
            att |= (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
                    | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ])
    return att & ~occ


def visible_squares(board: chess.Board, color: chess.Color) -> int:
    """Fog-of-war visibility: squares you occupy + squares your pieces could move to.

    Sliding pieces see up to and including the first blocker. Pawns only see
    squares they could move to: forward if empty, diagonal only if they could
    capture there. A pawn blocked head-on does NOT see what blocks it.
    """
    vis = board.occupied_co[color]
    enemy = board.occupied_co[not color]
    for sq in chess.scan_forward(board.occupied_co[color]):
        pt = board.piece_type_at(sq)
        if pt == chess.PAWN:
            vis |= pawn_pushes(color, sq, board.occupied)
            for t in chess.scan_forward(chess.BB_PAWN_ATTACKS[color][sq]):
                if enemy & sq_bb(t) or (board.turn == color and board.ep_square == t):
                    vis |= sq_bb(t)
        else:
            vis |= board.attacks_mask(sq)
    return vis


def opponent_moves_between(opp: chess.Color, ply_from: int, ply_to: int) -> int:
    """Number of moves the opponent made in plies [ply_from, ply_to). White moves on even plies."""
    parity = 0 if opp == chess.WHITE else 1
    first = ply_from + ((parity - ply_from) % 2)
    return 0 if first >= ply_to else (ply_to - 1 - first) // 2 + 1


def _min_move_tables() -> Dict[Tuple[bool, int], List[int]]:
    """BFS on an empty board: fewest moves for any piece of a type to reach each square.

    This is a lower bound in the real game (blockers only make it longer), so if
    the opponent has made fewer moves than this, the piece type cannot be there.
    """
    start = chess.Board()
    tables = {}
    for color in chess.COLORS:
        back = 0 if color == chess.WHITE else 7
        for pt in PT_TO_H:
            dist = [99] * 64
            frontier = list(start.pieces(pt, color))
            for s in frontier:
                dist[s] = 0
            if pt == chess.KING:          # castling lands the king on c/g file
                extra = [chess.square(2, back), chess.square(6, back)]
            elif pt == chess.ROOK:        # ...and the rook on d/f file
                extra = [chess.square(3, back), chess.square(5, back)]
            else:
                extra = []
            for s in extra:
                dist[s] = min(dist[s], 1)
            frontier += extra
            d = 0
            while frontier:
                d += 1
                nxt = []
                for s in frontier:
                    for t in squares_of(piece_reach(pt, color, s, 0)):
                        if dist[t] > d:
                            dist[t] = d
                            nxt.append(t)
                frontier = nxt
            tables[(color, pt)] = dist
    return tables


MIN_MOVES = _min_move_tables()


# =============================================================================
# 3. PLAYER KNOWLEDGE (each side's private information set)
# =============================================================================

@dataclass
class Sighting:
    symbol: str   # python-chess piece symbol, "." = seen empty, "?" = unknown enemy piece
    ply: int      # ply at which this was last confirmed


@dataclass
class Ghost:
    """An enemy piece last seen at `square` at `ply`, whose current location is unknown."""
    piece_type: int
    square: int
    ply: int


class PlayerKnowledge:
    """Everything one player legitimately knows. Updated after every ply."""

    MAX_DEPARTED = 32

    def __init__(self, color: chess.Color):
        self.color = color
        start = chess.Board()
        # Both players know the starting position, so memory starts complete at ply 0.
        self.memory: Dict[int, Sighting] = {
            sq: Sighting(start.piece_at(sq).symbol() if start.piece_at(sq) else ".", 0)
            for sq in chess.SQUARES
        }
        self.departed: List[Ghost] = []                 # enemy pieces seen leaving a square
        self.captured: List[Tuple[int, int]] = []       # (piece_type, square) we took
        self.events: List[str] = []                     # news to show at our next turn
        self.version = 0

    def _is_enemy(self, symbol: str) -> bool:
        return symbol not in ".?" and symbol.isupper() != (self.color == chess.WHITE)

    def _is_own(self, symbol: str) -> bool:
        return symbol not in ".?" and symbol.isupper() == (self.color == chess.WHITE)

    def observe(self, board: chess.Board, move_info: Optional[dict] = None) -> None:
        ply = board.ply()
        vis = visible_squares(board, self.color)

        if move_info is not None:
            # Our own pieces are never hidden from us: a square we just vacated is empty.
            for sq, s in self.memory.items():
                if self._is_own(s.symbol):
                    p = board.piece_at(sq)
                    if p is None or p.symbol() != s.symbol:
                        self.memory[sq] = Sighting(".", ply)
            # One of our pieces was taken: an unknown enemy piece now stands there.
            if move_info["victim"] == self.color:
                self.memory[move_info["capturer_square"]] = Sighting("?", ply)

        departed_now, appeared = [], []
        skip_sq = move_info["captured_square"] if (move_info and move_info["mover"] == self.color) else None
        for sq in chess.scan_forward(vis):
            p = board.piece_at(sq)
            sym = p.symbol() if p else "."
            old = self.memory[sq]
            if old.symbol != sym:
                if self._is_enemy(old.symbol) and sq != skip_sq:
                    departed_now.append(Ghost(chess.Piece.from_symbol(old.symbol).piece_type, sq, old.ply))
                if self._is_enemy(sym):
                    appeared.append(p.piece_type)
            self.memory[sq] = Sighting(sym, ply)

        # Pair "a knight left f6" with "a knight appeared on g4" in the same update.
        for g in departed_now:
            if g.piece_type in appeared:
                appeared.remove(g.piece_type)
            else:
                self.departed.append(g)
        # A piece that appears from nowhere probably explains our freshest ghost of that type.
        for pt in appeared:
            for i in range(len(self.departed) - 1, -1, -1):
                if self.departed[i].piece_type == pt:
                    del self.departed[i]
                    break
        self.departed = self.departed[-self.MAX_DEPARTED:]
        self.version += 1


# =============================================================================
# 4. OBSERVATION CONTEXT (what an estimator is allowed to see)
# =============================================================================

@dataclass
class ObservationContext:
    color: chess.Color                       # the observer
    ply: int
    visible: int                             # bitboard of visible squares
    known_pieces: Dict[int, chess.Piece]     # pieces on visible squares only
    knowledge: PlayerKnowledge

    @property
    def opp(self) -> chess.Color:
        return not self.color

    @property
    def fog(self) -> int:
        return chess.BB_ALL & ~self.visible

    @property
    def known_occupancy(self) -> int:
        occ = 0
        for sq in self.known_pieces:
            occ |= sq_bb(sq)
        return occ

    @property
    def opp_moves_total(self) -> int:
        return opponent_moves_between(self.opp, 0, self.ply)

    def opp_moves_since(self, ply: int) -> int:
        return opponent_moves_between(self.opp, ply, self.ply)

    def hidden_inventory(self) -> Tuple[Dict[int, float], int, float]:
        """(expected hidden count per type, exact total hidden pieces, unexplained extra).

        Total enemy pieces = 16 - captures we made (promotion keeps the count), and we
        know how many we can see, so the number of hidden enemy pieces is exact.
        Only the TYPE split is uncertain (because of promotions).
        """
        captured = Counter(pt for pt, _ in self.knowledge.captured)
        visible = Counter(p.piece_type for p in self.known_pieces.values() if p.color == self.opp)
        total_hidden = max(0, 16 - len(self.knowledge.captured) - sum(visible.values()))
        exp = {pt: float(max(0, INITIAL_COUNT[pt] - captured[pt] - visible[pt])) for pt in PT_TO_H}
        excess = sum(exp.values()) - total_hidden
        if excess > 0:                       # promotions: pawns are what went missing
            take = min(excess, exp[chess.PAWN])
            exp[chess.PAWN] -= take
            excess -= take
            others = sum(v for pt, v in exp.items() if pt != chess.KING)
            if excess > 0 and others > 0:
                f = max(0.0, (others - excess) / others)
                for pt in exp:
                    if pt != chess.KING:
                        exp[pt] *= f
        extra = max(0.0, total_hidden - sum(exp.values()))
        return exp, total_hidden, extra

    def ghost_candidates(self, inventory: Dict[int, float]) -> List[Ghost]:
        """Last known whereabouts of hidden enemy pieces, newest first, capped per type."""
        by_type: Dict[int, List[Ghost]] = defaultdict(list)
        fog = self.fog
        for sq, s in self.knowledge.memory.items():
            if fog & sq_bb(sq) and self.knowledge._is_enemy(s.symbol):
                by_type[chess.Piece.from_symbol(s.symbol).piece_type].append(Ghost(
                    chess.Piece.from_symbol(s.symbol).piece_type, sq, s.ply))
        for g in self.knowledge.departed:
            by_type[g.piece_type].append(g)
        out = []
        for pt, lst in by_type.items():
            keep = math.ceil(inventory.get(pt, 0.0) - 1e-9)
            lst.sort(key=lambda g: g.ply, reverse=True)
            out.extend(lst[:keep])
        return out


def build_context(board: chess.Board, knowledge: PlayerKnowledge) -> ObservationContext:
    vis = visible_squares(board, knowledge.color)
    known = {sq: board.piece_at(sq) for sq in chess.scan_forward(vis & board.occupied)}
    return ObservationContext(knowledge.color, board.ply(), vis, known, knowledge)


# =============================================================================
# 5. EVIDENCE SOURCES & ESTIMATORS
# =============================================================================

def effort_params(effort: int) -> Tuple[int, int]:
    """Estimator Effort -> (Monte Carlo samples per ghost, propagation depth in moves).

    Depth is the analogue of minimax search depth: trajectories that need more
    moves than the horizon are not explored and their mass is left on Theta.
    """
    return 60 * effort, effort


Evidence = Dict[int, List[Tuple[str, MassFunction]]]


class EvidenceSource(ABC):
    name = "evidence"

    @abstractmethod
    def collect(self, ctx: ObservationContext, effort: int, rng: Random) -> Evidence:
        ...


class MaterialEvidence(EvidenceSource):
    """Inventory BPA: hidden pieces of each type spread over the fogged squares.

    m({t}) = r * hidden_t / F,  m({Em}) = r * (F - hidden) / F,  m(Theta) = 1 - r
    """
    name = "Material inventory"

    def __init__(self, reliability: float = 0.9):
        self.r = reliability

    def collect(self, ctx, effort, rng):
        fog_sqs = squares_of(ctx.fog)
        F = len(fog_sqs)
        if F == 0:
            return {}
        exp, total_hidden, extra = ctx.hidden_inventory()
        m = {BIT[PT_TO_H[pt]]: self.r * c / F for pt, c in exp.items() if c > 0}
        m[BIT["Em"]] = self.r * max(0, F - total_hidden) / F
        if extra > 0:
            m[NON_KING] = m.get(NON_KING, 0.0) + self.r * extra / F
        mf = MassFunction(m)
        return {sq: [(self.name, mf)] for sq in fog_sqs}


class ConstraintEvidence(EvidenceSource):
    """Rules of chess that rule hypotheses out (categorical, mass on a complement set).

    - pawns never stand on the 1st or 8th rank
    - reachability: a type needing more moves (empty-board BFS) than the opponent has played
    - bishops are bound to a square colour
    - a type whose whole inventory is accounted for cannot be hidden
    Rules that a promotion could break are given weight < 1 instead of 1.
    """
    name = "Chess-rule constraints"

    def collect(self, ctx, effort, rng):
        out: Evidence = {}
        opp, n_opp = ctx.opp, ctx.opp_moves_total
        promo_possible = n_opp >= 5
        soft_w = 0.97 if promo_possible else 1.0
        exp, _, _ = ctx.hidden_inventory()

        captured_b = Counter(chess.BB_LIGHT_SQUARES & sq_bb(sq) != 0
                             for pt, sq in ctx.knowledge.captured if pt == chess.BISHOP)
        visible_b = Counter(chess.BB_LIGHT_SQUARES & sq_bb(sq) != 0
                            for sq, p in ctx.known_pieces.items()
                            if p.color == opp and p.piece_type == chess.BISHOP)
        bishops_left = {light: 1 - captured_b[light] - visible_b[light] for light in (True, False)}

        for sq in squares_of(ctx.fog):
            hard, soft = 0, 0
            if chess.square_rank(sq) in (0, 7):
                hard |= BIT["P"]
            for pt, h in PT_TO_H.items():
                if MIN_MOVES[(opp, pt)][sq] > n_opp and (pt in (chess.KING, chess.PAWN) or not promo_possible):
                    hard |= BIT[h]
            if bishops_left[bool(chess.BB_LIGHT_SQUARES & sq_bb(sq))] <= 0:
                soft |= BIT["B"]
            for pt, c in exp.items():
                if c <= 0:
                    if pt == chess.KING:
                        hard |= BIT["K"]
                    else:
                        soft |= BIT[PT_TO_H[pt]]
            soft &= ~hard
            lst = []
            if hard:
                lst.append((self.name, MassFunction.simple(THETA & ~hard, 1.0)))
            if soft:
                lst.append((self.name + " (soft)", MassFunction.simple(THETA & ~soft, soft_w)))
            if lst:
                out[sq] = lst
        return out


class MemoryEvidence(EvidenceSource):
    """What this square looked like when we last saw it, discounted by elapsed enemy moves.

    seen empty      -> m({Em})          = 0.95 * decay^d
    our piece taken -> m({any piece})   = 0.95 * decay^d
    (Enemy pieces we saw are handled by GhostTrajectoryEvidence.)
    """
    name = "Last sighting"

    def __init__(self, decay: float = 0.8):
        self.decay = decay

    def collect(self, ctx, effort, rng):
        out: Evidence = {}
        for sq in squares_of(ctx.fog):
            s = ctx.knowledge.memory[sq]
            w = 0.95 * self.decay ** ctx.opp_moves_since(s.ply)
            if s.symbol == ".":
                out[sq] = [(self.name, MassFunction.simple(BIT["Em"], w))]
            elif s.symbol == "?":
                out[sq] = [(self.name + " (capture)", MassFunction.simple(OCCUPIED, w))]
        return out


class GhostTrajectoryEvidence(EvidenceSource):
    """Monte Carlo propagation of enemy pieces from where we last saw them.

    For each ghost (type t, square s, seen d enemy moves ago):
      j ~ Binomial(d, 1/n_active)       how many of those moves were made by this piece
      random-walk j legal moves from s  (known pieces block; fog is treated as empty)
      reject end squares we can see     (we'd be looking at it: conditioning on observation)
      j > depth                         beyond the effort horizon -> ignorance (Theta)
    Each ghost gives an independent BPA per square: m({t}) = rho * share of samples.
    """
    name = "Ghost trajectories"

    def __init__(self, reliability: float = 0.85, max_active: int = 8):
        self.rho = reliability
        self.max_active = max_active

    def collect(self, ctx, effort, rng):
        samples, depth = effort_params(effort)
        inventory, total_hidden, _ = ctx.hidden_inventory()
        fog = ctx.fog
        occ_known = ctx.known_occupancy
        n_enemy = total_hidden + sum(1 for p in ctx.known_pieces.values() if p.color == ctx.opp)
        p_move = 1.0 / max(1, min(n_enemy, self.max_active))
        promo_rank = 7 if ctx.opp == chess.WHITE else 0
        out: Evidence = defaultdict(list)

        for g in ctx.ghost_candidates(inventory):
            n = ctx.opp_moves_since(g.ply)
            pmf = [math.comb(n, j) * p_move ** j * (1 - p_move) ** (n - j) for j in range(min(n, depth) + 1)]
            tail = max(0.0, 1.0 - sum(pmf))
            cum = list(accumulate(pmf + [tail]))
            horizon = len(pmf)                         # index meaning "beyond depth"
            occ = occ_known & ~sq_bb(g.square)
            hits: Counter = Counter()
            accepted = 0
            for _ in range(samples):
                j = min(bisect_right(cum, rng.random() * cum[-1]), horizon)
                if j == horizon:
                    accepted += 1                     # unexplored: contributes ignorance only
                    continue
                sq, pt = g.square, g.piece_type
                for _ in range(j):
                    dests = piece_reach(pt, ctx.opp, sq, occ)
                    if not dests:
                        break
                    sq = rng.choice(squares_of(dests))
                    if pt == chess.PAWN and chess.square_rank(sq) == promo_rank:
                        pt = chess.QUEEN
                if fog & sq_bb(sq):
                    accepted += 1
                    hits[(sq, pt)] += 1
            if not hits:
                continue
            per_sq: Dict[int, Dict[int, float]] = defaultdict(dict)
            for (sq, pt), c in hits.items():
                per_sq[sq][BIT[PT_TO_H[pt]]] = self.rho * c / accepted
            label = f"{self.name}: {PT_TO_H[g.piece_type]} last seen {chess.square_name(g.square)}"
            for sq, m in per_sq.items():
                out[sq].append((label, MassFunction(m)))
        return out


@dataclass
class SquareEstimate:
    mass: MassFunction
    conflict: float                 # overall K of the n-way combination
    sources: List[str]
    skipped: List[str]

    def row(self, h: str) -> Tuple[float, float, float]:
        b = BIT[h]
        return self.mass.bel(b), self.mass.pl(b), self.mass.betp(h)

    def best_guess(self) -> Tuple[str, float]:
        return max(((h, self.mass.betp(h)) for h in HYPOTHESES), key=lambda x: x[1])


class UncertaintyEstimator(ABC):
    """Interface every estimator implements. Register new ones in ESTIMATORS."""
    name = "estimator"
    description = ""

    @abstractmethod
    def estimate(self, ctx: ObservationContext, effort: int, rng: Random) -> Dict[int, SquareEstimate]:
        ...


class DempsterShaferEstimator(UncertaintyEstimator):
    name = "Dempster-Shafer (evidence fusion)"
    description = ("Fuses material, rule constraints, last sightings and Monte Carlo ghost "
                   "trajectories with Dempster's rule. Effort = ghost search depth and samples.")

    def __init__(self, sources: Optional[List[EvidenceSource]] = None):
        self.sources = sources or [MaterialEvidence(), ConstraintEvidence(),
                                   MemoryEvidence(), GhostTrajectoryEvidence()]

    def estimate(self, ctx, effort, rng):
        per_square: Evidence = defaultdict(list)
        for src in self.sources:
            for sq, lst in src.collect(ctx, effort, rng).items():
                per_square[sq].extend(lst)
        result = {}
        for sq in squares_of(ctx.fog):
            m = MassFunction.vacuous()
            keep_product = 1.0
            used, skipped = [], []
            for label, mf in per_square.get(sq, []):
                try:
                    m, k = m.combine(mf)
                    keep_product *= (1.0 - k)
                    used.append(label)
                except TotalConflict:
                    skipped.append(label)
            result[sq] = SquareEstimate(m, 1.0 - keep_product, used, skipped)
        return result


class MaterialBaselineEstimator(UncertaintyEstimator):
    """Bayesian baseline: a BPA whose focal sets are all singletons is just a probability."""
    name = "Material frequency (Bayesian baseline)"
    description = "Uniform spread of the hidden inventory over the fog. No memory, no ignorance mass."

    def estimate(self, ctx, effort, rng):
        ev = MaterialEvidence(reliability=1.0).collect(ctx, effort, rng)
        return {sq: SquareEstimate(lst[0][1], 0.0, [lst[0][0]], []) for sq, lst in ev.items()}


ESTIMATORS: Dict[str, UncertaintyEstimator] = {
    e.name: e for e in (DempsterShaferEstimator(), MaterialBaselineEstimator())
}


# =============================================================================
# 6. GAME CONTROLLER (session state)
# =============================================================================

COLOR_NAME = {chess.WHITE: "White", chess.BLACK: "Black"}


def new_game() -> None:
    ss = st.session_state
    ss.board = chess.Board()
    ss.knowledge = {c: PlayerKnowledge(c) for c in chess.COLORS}
    for k in ss.knowledge.values():
        k.observe(ss.board)
    ss.full_log = []                          # revealed only when the game ends
    ss.own_log = {chess.WHITE: [], chess.BLACK: []}
    ss.last_move = {}                         # color -> (from, to) of that player's own last move
    ss.game_over = False
    ss.result = ""
    ss.awaiting_handoff = False
    ss.est_cache = {}


def describe_move(board: chess.Board, move: chess.Move) -> str:
    if board.is_castling(move):
        return "O-O" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "O-O-O"
    piece = board.piece_at(move.from_square)
    letter = "" if piece.piece_type == chess.PAWN else piece.symbol().upper()
    sep = "x" if board.is_capture(move) else "-"
    promo = f"={chess.piece_symbol(move.promotion).upper()}" if move.promotion else ""
    return f"{letter}{chess.square_name(move.from_square)}{sep}{chess.square_name(move.to_square)}{promo}"


def fow_moves(board: chess.Board) -> List[chess.Move]:
    """FOW rules: no check, so every pseudo-legal move is allowed (kings can be captured)."""
    return list(board.pseudo_legal_moves)


def play_move(move: chess.Move) -> None:
    ss = st.session_state
    board: chess.Board = ss.board
    mover, victim = board.turn, not board.turn
    text = describe_move(board, move)
    captured_sq = move.to_square
    if board.is_en_passant(move):
        captured_sq = move.to_square + (-8 if mover == chess.WHITE else 8)
    captured = board.piece_at(captured_sq) if board.is_capture(move) else None

    board.push(move)
    ss.full_log.append(f"{COLOR_NAME[mover]}: {text}")
    ss.own_log[mover].append(text)
    ss.setdefault("last_move", {})[mover] = (move.from_square, move.to_square)

    info = {"mover": mover, "victim": victim if captured else None,
            "captured_square": captured_sq if captured else None,
            "capturer_square": move.to_square}
    if captured:
        ss.knowledge[mover].captured.append((captured.piece_type, captured_sq))
        ss.knowledge[victim].events.append(
            f"Your {HYP_NAMES[PT_TO_H[captured.piece_type]].lower()} on "
            f"{chess.square_name(captured_sq)} was captured.")
    for k in ss.knowledge.values():
        k.observe(board, info)

    if captured is not None and captured.piece_type == chess.KING:
        ss.game_over, ss.result = True, f"{COLOR_NAME[mover]} wins by capturing the king!"
    elif not fow_moves(board):
        ss.game_over, ss.result = True, "Draw: no moves available."
    ss.est_cache = {}
    ss.awaiting_handoff = ss.privacy and not ss.game_over


def get_estimates(estimator_name: str, effort: int) -> Tuple[Dict[int, SquareEstimate], float]:
    ss = st.session_state
    board: chess.Board = ss.board
    know = ss.knowledge[board.turn]
    key = f"{board.fen()}|{board.turn}|{know.version}|{estimator_name}|{effort}"
    if key not in ss.est_cache:
        rng = Random(int(hashlib.sha1(key.encode()).hexdigest()[:12], 16))  # stable between reruns
        t0 = time.perf_counter()
        est = ESTIMATORS[estimator_name].estimate(build_context(board, know), effort, rng)
        ss.est_cache = {key: (est, time.perf_counter() - t0)}
    return ss.est_cache[key]


# =============================================================================
# 7. STREAMLIT UI
# =============================================================================

GLYPH = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
         chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
VS15 = "︎"  # force text (non-emoji) rendering

# Drag-and-drop board: a dependency-free custom component in ./fow_board/index.html
_fow_board = components.declare_component("fow_board", path=str(Path(__file__).parent / "fow_board"))

LEGEND = ("Hover a fogged square. Bar: solid = belief, faded = plausibility, "
          "yellow tick = pignistic probability.")


def _estimate_payload(est: SquareEstimate) -> dict:
    composite = sorted(((k, v) for k, v in est.mass.m.items() if k.bit_count() > 1),
                       key=lambda kv: -kv[1])[:3]
    return {
        "rows": [[h, *est.row(h)] for h in DISPLAY_ORDER],
        "comp": html.escape(", ".join(f"m{mask_label(k)}={v:.3f}" for k, v in composite) or "none"),
        "K": est.conflict,
        "n": len(est.sources),
        "skipped": len(est.skipped),
        "guess": list(est.best_guess()),
        "occ": 1.0 - est.mass.betp("Em"),
    }


def board_widget(board: chess.Board, pov: chess.Color, visible: Optional[int],
                 estimates: Optional[Dict[int, SquareEstimate]], show_guess: bool, heat: bool,
                 interactive: bool, last: Optional[Tuple[int, int]], key: str) -> Optional[dict]:
    """Render the board. Returns {"uci", "ply"} when the player drops a piece on a legal square.

    Only what `pov` may see is sent to the browser: fogged squares carry no piece data.
    """
    cells = []
    for sq in chess.SQUARES:
        if visible is None or visible & sq_bb(sq):
            p = board.piece_at(sq)
            cells.append({"fog": False, "p": p.symbol() if p else None})
        else:
            est = estimates.get(sq) if estimates else None
            cells.append({"fog": True, "est": _estimate_payload(est) if est else None})
    moves: Dict[str, list] = defaultdict(list)
    if interactive:
        for mv in fow_moves(board):
            if mv.promotion in (None, chess.QUEEN):      # one entry per target; JS asks for the piece
                moves[str(mv.from_square)].append([mv.to_square, mv.promotion is not None])
    return _fow_board(cells=cells, pov="w" if pov == chess.WHITE else "b", moves=moves,
                      ply=board.ply(), interactive=interactive, showGuess=show_guess, heat=heat,
                      last=list(last) if last else None, legend=LEGEND if estimates else "",
                      key=key, default=None)


def apply_board_value(board: chess.Board, value: Optional[dict]) -> None:
    """Play the move the board component reported, if it is for this turn and pseudo-legal."""
    if not value or value.get("ply") != board.ply():
        return
    try:
        move = chess.Move.from_uci(value["uci"])
    except (KeyError, ValueError):
        return
    if move in board.pseudo_legal_moves:
        play_move(move)
        st.rerun()


STRATEGY_MD = r"""
**Frame of discernment.** A fogged square can only hide an *enemy* piece or nothing, so
$\Theta=\{K,Q,R,B,N,P,Em\}$. Every piece of evidence is a basic probability assignment
$m:2^\Theta\to[0,1]$, $m(\emptyset)=0$, $\sum m=1$. Mass on a set like $\{Q,R,B,N,P\}$ means
"a piece, type unknown"; mass on $\Theta$ means "no idea" - something a single probability cannot say.

**Evidence sources (one BPA each, per fogged square):**
1. *Material inventory* - hidden enemy count is exact ($16-$captures$-$visible), so
   $m(\{t\})=r\,h_t/F$, $m(\{Em\})=r(F-h)/F$, $m(\Theta)=1-r$.
2. *Chess-rule constraints* - categorical mass on a complement: no pawns on ranks 1/8,
   bishops bound to colour, types that need more moves than the opponent has played (BFS lower bound),
   types whose whole inventory is accounted for.
3. *Last sighting* - "seen empty $d$ enemy moves ago": $m(\{Em\})=0.95\cdot0.8^d$; a square where
   your piece was captured: $m(\{K,Q,R,B,N,P\})=0.95\cdot0.8^d$.
4. *Ghost trajectories* - Monte Carlo random walks of every enemy piece from its last sighting,
   conditioned on not ending on a square you can see. Each ghost is a separate source with
   $m(\{t\})=\rho\cdot$(share of samples ending here).

**Combination.** Dempster's rule: $(m_1\oplus m_2)(A)=\frac{1}{1-K}\sum_{B\cap C=A}m_1(B)m_2(C)$,
$K=\sum_{B\cap C=\emptyset}m_1(B)m_2(C)$. The tooltip shows $Bel(\{x\})=m(\{x\})$,
$Pl(\{x\})=\sum_{A\ni x}m(A)$, the pignistic probability $BetP$ and the total conflict $K$.

**Estimator Effort** sets the ghost search: depth $=$ effort (max moves a piece is traced, like
minimax depth) and $60\times$effort samples per ghost. Trajectories beyond the horizon are not
explored; their mass stays on $\Theta$, so low effort = wider $[Bel, Pl]$ intervals.
"""


def sidebar() -> Tuple[str, int, bool, bool]:
    ss = st.session_state
    with st.sidebar:
        st.header("Estimator")
        name = st.selectbox("Uncertainty estimator", list(ESTIMATORS), key="estimator_name")
        st.caption(ESTIMATORS[name].description)
        effort = st.slider("Estimator Effort", 1, 10, 4, key="effort",
                           help="Ghost propagation depth (moves) and Monte Carlo samples. "
                                "Analogous to minimax search depth.")
        samples, depth = effort_params(effort)
        st.caption(f"Depth {depth} moves · {samples} samples per tracked piece")
        show_guess = st.checkbox("Show best guess on fog", value=True)
        heat = st.checkbox("Occupancy heat tint", value=False)
        st.divider()
        st.header("Game")
        st.checkbox("Hot-seat privacy screen", key="privacy",   # default set in main()
                    help="Hide the board between turns so players can't see each other's view.")
        c1, c2 = st.columns(2)
        if c1.button("New game", use_container_width=True):
            new_game()
            st.rerun()
        if c2.button("Resign", use_container_width=True, disabled=ss.game_over):
            ss.game_over = True
            ss.result = f"{COLOR_NAME[ss.board.turn]} resigned. {COLOR_NAME[not ss.board.turn]} wins."
            st.rerun()
        with st.expander("How the estimator works"):
            st.markdown(STRATEGY_MD)
    return name, effort, show_guess, heat


def move_panel(board: chess.Board) -> None:
    moves = fow_moves(board)
    by_from: Dict[int, List[chess.Move]] = defaultdict(list)
    for mv in moves:
        by_from[mv.from_square].append(mv)
    ply = board.ply()

    def from_label(sq):
        p = board.piece_at(sq)
        return f"{HYP_NAMES[PT_TO_H[p.piece_type]]} {chess.square_name(sq)}"

    from_sq = st.selectbox("Piece", sorted(by_from, key=lambda s: (board.piece_type_at(s), s)),
                           format_func=from_label, key=f"from_{ply}")
    options = [m for m in by_from[from_sq] if m.promotion in (None, chess.QUEEN)]
    to_mv = st.selectbox("Move", options, format_func=lambda m: describe_move(board, m), key=f"to_{ply}")
    promo = None
    if to_mv.promotion:
        promo = st.radio("Promote to", [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT],
                         format_func=lambda pt: HYP_NAMES[PT_TO_H[pt]], horizontal=True, key=f"promo_{ply}")
    if st.button("Make move", type="primary", use_container_width=True):
        play_move(chess.Move(to_mv.from_square, to_mv.to_square, promotion=promo))
        st.rerun()


def inspector(estimates: Dict[int, SquareEstimate]) -> None:
    with st.expander("Inspect a fogged square (table view)"):
        sq = st.selectbox("Square", sorted(estimates), format_func=chess.square_name, key="inspect_sq")
        est = estimates[sq]
        st.dataframe(
            [{"Hypothesis": f"{h} ({HYP_NAMES[h]})", "m({x}) = Bel": round(est.row(h)[0], 4),
              "Pl": round(est.row(h)[1], 4), "BetP": round(est.row(h)[2], 4)} for h in DISPLAY_ORDER],
            hide_index=True, use_container_width=True)
        st.caption("Focal elements: " + ", ".join(
            f"m{mask_label(k)} = {v:.4f}" for k, v in sorted(est.mass.m.items(), key=lambda kv: -kv[1])[:8]))
        st.caption("Sources: " + "; ".join(est.sources) + (f" | skipped: {'; '.join(est.skipped)}" if est.skipped else ""))


def main() -> None:
    st.set_page_config(page_title="Fog-of-War Chess · DST", page_icon="♞", layout="wide")
    ss = st.session_state
    if "board" not in ss:
        ss.privacy = True
        new_game()
    est_name, effort, show_guess, heat = sidebar()

    st.title("Fog-of-War Chess")
    st.caption("Two players, one screen. You only see squares your pieces stand on or can move to. "
               "No check: win by capturing the king.")
    unc_on = st.toggle("**Uncertainty Estimation**", key="unc_on",
                       help="Hover any fogged square to see its Dempster-Shafer mass assignment.")

    board: chess.Board = ss.board

    if ss.game_over:
        st.success(ss.result)
        left, right = st.columns([3, 2])
        with left:
            board_widget(board, chess.WHITE, None, None, False, False, interactive=False,
                         last=None, key="final_board")
        with right:
            st.subheader("Full move log (revealed)")
            st.write(" · ".join(ss.full_log) or "No moves.")
        return

    turn = board.turn
    if ss.awaiting_handoff:
        st.info(f"🔒 Board hidden. Pass the device to **{COLOR_NAME[turn]}**.")
        if st.button(f"I'm {COLOR_NAME[turn]} — show my board", type="primary"):
            ss.awaiting_handoff = False
            st.rerun()
        return

    know = ss.knowledge[turn]
    vis = visible_squares(board, turn)
    estimates, elapsed = (get_estimates(est_name, effort) if unc_on else ({}, 0.0))

    left, right = st.columns([3, 2], gap="large")
    with left:
        value = board_widget(board, turn, vis, estimates, show_guess and unc_on, heat and unc_on,
                             interactive=True, last=ss.get("last_move", {}).get(turn), key=f"board_{board.ply()}")
        apply_board_value(board, value)
    with right:
        st.subheader(f"{COLOR_NAME[turn]} to move")
        st.caption("Drag a piece to move it, or click a piece and then a highlighted square.")
        for ev in know.events:
            st.warning(ev)
        know.events.clear()
        with st.expander("Or choose a move from menus"):
            move_panel(board)

        ctx = build_context(board, know)
        _, hidden, _ = ctx.hidden_inventory()
        c1, c2, c3 = st.columns(3)
        c1.metric("Fogged squares", chess.popcount(ctx.fog))
        c2.metric("Hidden enemy pieces", hidden)
        c3.metric("Enemy moves so far", ctx.opp_moves_total)
        if unc_on:
            st.caption(f"Estimator: {est_name} · effort {effort} · computed in {elapsed*1000:.0f} ms")
            if estimates:
                inspector(estimates)
        taken = [GLYPH[pt] + VS15 for pt, _ in know.captured]
        st.markdown(f"**You captured:** {' '.join(taken) if taken else '—'}")
        st.markdown(f"**Your moves:** {' · '.join(ss.own_log[turn]) or '—'}")


if __name__ == "__main__":
    from streamlit import runtime

    if runtime.exists():
        main()
    else:
        # Started with `python app.py`: relaunch properly via `streamlit run`.
        import subprocess
        import sys

        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))
