# The Dempster-Shafer Fog Estimator, Explained

This document walks through how the app estimates what is under each fogged square: the
theory, every evidence source, the numbers used, and a fully worked example. All constants
quoted here are the ones in [`app.py`](app.py).

**Contents**

1. [Why Dempster-Shafer instead of Bayes?](#1-why-dempster-shafer-instead-of-bayes)
2. [The frame of discernment](#2-the-frame-of-discernment)
3. [Mass functions, belief, plausibility, pignistic probability](#3-mass-functions-belief-plausibility-pignistic-probability)
4. [Dempster's rule of combination](#4-dempsters-rule-of-combination)
5. [What the observer knows (the information set)](#5-what-the-observer-knows-the-information-set)
6. [The four evidence sources](#6-the-four-evidence-sources)
7. [Estimator Effort: the analogue of search depth](#7-estimator-effort-the-analogue-of-search-depth)
8. [Putting it together: a full worked example](#8-putting-it-together-a-full-worked-example)
9. [Limitations and ideas for future estimators](#9-limitations-and-ideas-for-future-estimators)

---

## 1. Why Dempster-Shafer instead of Bayes?

The original Bayesian Chess Bot modelled *aleatoric* uncertainty: given a fully known position,
how noisy is the evaluation? Fog-of-war adds *epistemic* uncertainty: the position itself is
partly unknown.

A Bayesian model has to commit every bit of probability to individual outcomes. With no
information about a square, it has to pick a prior, usually uniform. That makes "I have no idea
what's on h5" look exactly like "I have strong evidence that every option on h5 is equally
likely".

Dempster-Shafer Theory (DST) separates the two:

| Situation | Bayesian | Dempster-Shafer |
|---|---|---|
| No evidence at all | $P(x)=1/7$ for each | $m(\Theta)=1$: all mass on "anything" |
| "A piece is here, type unknown" | Must split the probability among the types somehow | $m(\{K,Q,R,B,N,P\})=1$, an honest set-valued statement |
| "Probably a knight, otherwise no idea" | $P(N)=0.6$, the remaining 0.4 split over the others | $m(\{N\})=0.6,\ m(\Theta)=0.4$ |

DST assigns mass to **sets** of possibilities. Evidence that only narrows things down to a group
stays at that group. It isn't forced down to individual piece types until you ask for a single
decision.

---

## 2. The frame of discernment

The **frame of discernment** $\Theta$ is the set of mutually exclusive answers to the question
*"what is on this square?"*. Your own pieces are never hidden from you, so a fogged square holds
either an **enemy** piece or nothing. Colour is therefore not part of the frame:

$$\Theta = \{K,\ Q,\ R,\ B,\ N,\ P,\ Em\}$$

### Bitmask encoding

Each hypothesis gets one bit, so every subset of $\Theta$ is a 7-bit integer. Set intersection
becomes a single bitwise AND, which keeps Dempster's rule fast.

| Hypothesis | K | Q | R | B | N | P | Em |
|---|---|---|---|---|---|---|---|
| Bit | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| Value | 1 | 2 | 4 | 8 | 16 | 32 | 64 |

Named sets used in the code:

| Name | Members | Mask |
|---|---|---|
| `THETA` | everything | 127 |
| `OCCUPIED` | $\{K,Q,R,B,N,P\}$, "some piece" | 63 |
| `NON_KING` | $\{Q,R,B,N,P\}$ | 62 |

For example $\{N\}\cap\{N,P\}$ = `16 & 48` = `16` = $\{N\}$, while $\{N\}\cap\{P\}$ = `16 & 32` = `0`
= $\emptyset$, which is a contradiction.

---

## 3. Mass functions, belief, plausibility, pignistic probability

### Basic probability assignment (mass function)

A mass function $m : 2^\Theta \to [0,1]$ satisfies

$$m(\emptyset)=0, \qquad \sum_{A\subseteq\Theta} m(A) = 1.$$

$m(A)$ is the weight of evidence that supports **exactly** $A$ and nothing more specific. Sets with
$m(A)>0$ are called **focal sets**.

In `MassFunction`, any mass a source doesn't assign is automatically put on $\Theta$. That is how
every source expresses "and the rest I don't know".

Two special shapes come up constantly:

- **Vacuous** mass function: $m(\Theta)=1$, total ignorance. Combination starts from this.
- **Simple support function**: $m(A)=w,\ m(\Theta)=1-w$, meaning "I support $A$ with
  strength $w$". Created with `MassFunction.simple(A, w)`. With $w=1$ it becomes *categorical*:
  $A$ is certain.

### Belief and plausibility

From $m$ we get two bounds for any hypothesis set $A$:

$$\mathrm{Bel}(A)=\sum_{B\subseteq A} m(B) \qquad\qquad \mathrm{Pl}(A)=\sum_{B\cap A\neq\emptyset} m(B) = 1-\mathrm{Bel}(\bar A)$$

- **Belief** is the evidence that *forces* $A$ to be true.
- **Plausibility** is the evidence that *doesn't rule $A$ out*.
- $\mathrm{Bel}(A)\le \mathrm{Pl}(A)$, and the gap $\mathrm{Pl}-\mathrm{Bel}$ is the ignorance about $A$.

For a single piece type $\{x\}$, the only subset is $\{x\}$ itself, so
$\mathrm{Bel}(\{x\}) = m(\{x\})$. That's why the tooltip column is labelled **m=Bel**.

### Pignistic probability

To make a decision (such as the "best guess" label on a fogged square), the mass is turned into
an ordinary probability. This is Smets' **pignistic transform**: split each focal set's mass
evenly among its members.

$$\mathrm{BetP}(x) = \sum_{A \ni x} \frac{m(A)}{|A|}$$

It always lies within $[\mathrm{Bel}(\{x\}),\ \mathrm{Pl}(\{x\})]$ and sums to 1 over $\Theta$.
The test suite checks both properties on every square of every test game.

---

## 4. Dempster's rule of combination

Given two mass functions from **independent** sources, Dempster's rule fuses them. Every focal set
of one is intersected with every focal set of the other, and the product of their masses goes to
the intersection:

$$K = \sum_{B\cap C=\emptyset} m_1(B)\,m_2(C)
\qquad
(m_1\oplus m_2)(A) = \frac{1}{1-K}\sum_{B\cap C = A} m_1(B)\,m_2(C), \quad A\neq\emptyset$$

- $K$ is the **conflict**: the mass that fell on the empty set because the two sources
  contradicted each other.
- Dividing by $1-K$ **renormalises**: the conflicting mass is thrown away and the rest is scaled
  back up to sum to 1.

### Worked example

Source 1 says "knight, with strength 0.6". Source 2 says "pawn, with strength 0.5".

| | $m_2(\{P\})=0.5$ | $m_2(\Theta)=0.5$ |
|---|---|---|
| $m_1(\{N\})=0.6$ | $\{N\}\cap\{P\}=\emptyset$ → **0.30 conflict** | $\{N\}$ → 0.30 |
| $m_1(\Theta)=0.4$ | $\{P\}$ → 0.20 | $\Theta$ → 0.20 |

$K=0.30$, so divide everything else by $0.70$:

| Focal set | Mass | | Hypothesis | Bel | Pl | BetP |
|---|---|---|---|---|---|---|
| $\{N\}$ | 0.429 | | N | 0.429 | 0.714 | 0.469 |
| $\{P\}$ | 0.286 | | P | 0.286 | 0.571 | 0.327 |
| $\Theta$ | 0.286 | | each other type | 0 | 0.286 | 0.041 |

This exact case is one of the unit tests.

### Combining many sources

The rule is **commutative and associative**, so the order the sources are combined in doesn't
matter. The estimator starts from the vacuous mass function and folds in each source in turn.
If source $i$ contributes conflict $K_i$, the conflict of the whole combination is

$$K_{\text{total}} = 1 - \prod_i (1-K_i),$$

which is the **Conflict K** shown in the tooltip. A high value means the sources disagree a lot,
which is useful to know in its own right. For instance, a ghost trajectory may say "the knight
went here" while your memory says "this square looked empty two moves ago".

### Total conflict

If $K=1$, the two sources contradict each other completely and Dempster's rule is undefined.
The estimator then **skips** that source for that square and lists it under "skipped" in the
tooltip and the inspector. Every source except the hard chess-rule constraints leaves some mass
on $\Theta$, so this rarely happens.

---

## 5. What the observer knows (the information set)

An estimator must only use what the player in front of the screen can legitimately know. Two
classes enforce this.

### `PlayerKnowledge`: each player's private memory

Updated after **every** ply, for both players:

| Field | Meaning |
|---|---|
| `memory[sq] = Sighting(symbol, ply)` | What this square last looked like, and when. Starts as the full **initial position at ply 0**, which both players know. Symbols: a piece letter, `.` for empty, `?` for an unknown enemy piece. |
| `departed` | "Ghost" records: an enemy piece of type *t* was seen on square *s* at ply *p*, and has since left. |
| `captured` | Enemy pieces this player has taken, with type and square. |
| `events` | Messages shown at the start of the player's next turn ("Your bishop on c4 was captured"). |

Update rules, applied in this order on each observation:

1. **Own vacated squares become empty.** If one of your pieces left a square, that square is
   known to be empty *at this ply*, even if you can no longer see it.
2. **Captures leave a `?`.** If your piece was captured, the capturing square holds an enemy
   piece of unknown type.
3. **Visible squares are overwritten** with what is there now.
4. **Departures and appearances are paired.** If a knight disappears from f6 and a knight
   appears on g4 in the same update, it's assumed to be the same knight, and no ghost is
   created. A departure with no matching appearance creates a ghost. An appearance with no
   matching departure deletes the newest ghost of that type, because that piece has probably
   just been found.
5. At most 32 departed ghosts are kept.

### `ObservationContext`: the estimator's whole view of the world

Built fresh for each estimate. It contains the observer's colour, the current ply, the
visibility bitboard, the pieces on visible squares, and the `PlayerKnowledge`. **The true board
is never passed in**, so an estimator can't cheat even by accident. The context also derives:

- `fog`: every square not in the visible set;
- `opp_moves_total` and `opp_moves_since(ply)`: how many moves the opponent has made in total,
  and since a given ply (White moves on even plies, Black on odd);
- `hidden_inventory()`: see §6.1.

---

## 6. The four evidence sources

Each source produces, for every fogged square it has an opinion about, one or more mass
functions. They are all combined with Dempster's rule (§4).

### 6.1 Material inventory: "how much is hidden?"

**Idea.** You know exactly how many enemy pieces are hidden. You also know roughly how many of
each type, so spread them over the fog.

**The exact count.** The enemy starts with 16 pieces. Every capture *you* make removes exactly
one. A promotion turns a pawn into another piece, so it doesn't change the count. Therefore:

$$h = 16 - |\text{your captures}| - |\text{visible enemy pieces}|$$

is **exact**.

**The split by type is approximate**, because promotions are invisible:

$$\hat h_t = \max\bigl(0,\ \text{initial}_t - \text{captured}_t - \text{visible}_t\bigr)$$

If $\sum_t \hat h_t > h$, some pawns must have promoted, for example if you can see two enemy
queens. The excess is taken from pawns first, and any remainder by scaling down the non-king
types. If $\sum_t \hat h_t < h$, the shortfall $e$ is put on $\{Q,R,B,N,P\}$ ("a piece of some
non-king type").

**Mass function.** With $F$ fogged squares and reliability $r = 0.9$:

$$m(\{t\}) = r\,\frac{\hat h_t}{F}, \qquad m(\{Em\}) = r\,\frac{F-h}{F}, \qquad m(\{Q,R,B,N,P\}) = r\,\frac{e}{F}, \qquad m(\Theta) = 1-r$$

The same mass function applies to every fogged square. On its own it is essentially a Bayesian
prior. The reliability discount of 0.9 means the sources below can still override it.

### 6.2 Chess-rule constraints: "what is impossible here?"

**Idea.** Some piece types simply cannot be on some squares. These are expressed as
**categorical** evidence: mass on the *complement* of the excluded types.

| Rule | Excludes | Strength |
|---|---|---|
| Pawns never stand on rank 1 or 8 | P | hard (1.0) |
| **Reachability:** the type needs more moves to reach this square than the opponent has played | that type | hard (1.0) |
| The king is accounted for (visible) | K | hard (1.0) |
| Every enemy bishop on this square colour is accounted for (captured or visible) | B | soft |
| Every piece of a type is accounted for | that type | soft |

**Reachability** uses a table built once at start-up with a breadth-first search on an *empty*
board. For each colour and type, it records the fewest moves for any starting piece of that type
to reach each square, with castling counted as one move that lands the king on the c- or g-file
and the rook on the d- or f-file. Blockers only make real paths longer, so this is a valid
**lower bound**. Examples:

- The enemy king starting on e8 needs at least 7 moves to reach e1.
- After 1 enemy move, no enemy pawn can be on rank 4 of its own side (that needs 2 moves), and
  no enemy bishop can be anywhere that takes 2 bishop moves to reach.

This rule makes the estimator sharp in the opening, when the fog covers half the board but the
enemy has only made a handful of moves.

**Promotions.** A promotion needs at least 5 pawn moves (a double step, then four single steps).
So once the opponent has made 5 or more moves:

- reachability for Q, R, B and N is **switched off**, since a promoted piece could appear on the
  back rank. It stays on for K and P, which can never be created by promotion;
- "soft" rules are given strength **0.97** instead of 1.0, leaving 3% on $\Theta$ for the chance
  of a promoted piece.

All hard exclusions for a square are merged into one mass function $m(\Theta\setminus H)=1$. The
soft ones go into a second, $m(\Theta\setminus S)=w$.

### 6.3 Last sighting: "what did this square look like?"

**Idea.** If you saw a square empty a moment ago, it's probably still empty. That evidence
weakens with every move the opponent makes.

If a square was last seen $d$ enemy moves ago:

| Last seen as | Mass function |
|---|---|
| empty (`.`) | $m(\{Em\}) = 0.95\cdot 0.8^{d}$ |
| a capture square (`?`) | $m(\{K,Q,R,B,N,P\}) = 0.95\cdot 0.8^{d}$ |

| $d$ | 0 | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| strength | 0.95 | 0.76 | 0.61 | 0.49 | 0.31 | 0.10 |

The capture case shows why set-valued mass is useful. You know an enemy piece took your rook on
d4, but not which piece, so the evidence sits on "any piece" without favouring a type. Other
sources (material, ghosts, reachability) then decide which types are more likely.

Squares last seen holding an *enemy* piece are not handled here. That piece may have moved, so
it is the job of the ghost source.

### 6.4 Ghost trajectories: "where could that piece have gone?"

This is the most important source and the one tied to **Estimator Effort**. It uses the move
history to track every enemy piece you have seen.

**Step 1: choose the ghosts.** Candidates are:

- enemy pieces remembered on squares that are now fogged ("the knight was on c6 when I last
  looked, and c6 is now fogged"); and
- `departed` records ("the bishop left f5").

For each type $t$, only the $\lceil \hat h_t\rceil$ **most recent** candidates are kept. For
example, if only one enemy knight is hidden, older and probably stale knight sightings are
dropped.

**Step 2: how often did it move?** If $n$ enemy moves have passed since the sighting and the
enemy has $N$ pieces, assume each move is made by one of $\min(N, 8)$ "active" pieces, chosen
uniformly. The number of times this ghost moved is then

$$j \sim \mathrm{Binomial}\!\left(n,\ p\right), \qquad p = \frac{1}{\min(N,8)}.$$

The cap of 8 reflects that players move a handful of pieces much more than the rest.

**Step 3: simulate.** For each of $S$ samples:

1. Draw $j$.
2. **If $j$ is larger than the search depth $D$**, the sample is *beyond the horizon*. It isn't
   explored, and its share becomes ignorance (see below).
3. Otherwise random-walk $j$ moves from the last-seen square, picking uniformly among the
   piece's **non-capturing** moves on the *known* board. Your pieces and visible enemy pieces
   block; fogged squares count as empty. Pawns only push. A pawn reaching the last rank becomes
   a queen. A piece with no moves stays where it is.
4. **If the walk ends on a square you can see, reject the sample.** You'd be looking at the
   piece if it were there, and you aren't. This is *conditioning on the observation*: the same
   trick a particle filter uses, applied to the visibility mask.

**Step 4: turn the samples into mass.** Let `accepted` be the samples that ended in the fog, plus
the beyond-horizon samples. With reliability $\rho = 0.85$:

$$m_g(\{t\}) \text{ on square } s = \rho\cdot\frac{\#\{\text{samples ending on } s \text{ as type } t\}}{\text{accepted}}, \qquad m_g(\Theta) = \text{the rest}$$

Each ghost is its **own** evidence source on each square it can reach, labelled in the inspector
as e.g. `Ghost trajectories: N last seen c6`. Beyond-horizon samples count in the denominator but
put mass nowhere, so they increase $m(\Theta)$. That is exactly how a shallow search should
behave: it admits it didn't look far enough.

A pawn ghost can contribute to both $\{P\}$ and $\{Q\}$ on a back-rank square, through promotion.

---

## 7. Estimator Effort: the analogue of search depth

The sidebar slider (1–10) sets two ghost-search parameters:

$$S = 60\times\text{effort samples per ghost}, \qquad D = \text{effort (max moves traced per ghost)}$$

| Effort | Samples | Depth | Behaviour |
|---|---|---|---|
| 1 | 60 | 1 | Only pieces that moved at most once are traced. Most ghost mass stays on $\Theta$: wide [Bel, Pl] intervals, cautious guesses. |
| 4 (default) | 240 | 4 | A good balance for normal play. |
| 10 | 600 | 10 | Long histories are explored and noise is low. Sharper, more committed estimates. |

This mirrors minimax depth in the original engine. A deeper search costs more and commits to
stronger conclusions. Unlike minimax, a shallow DST search doesn't give a *wrong* answer; it
gives a *less committed* one, because unexplored possibilities are left as ignorance rather than
guessed. Even at effort 10, a full estimate took under 0.11 s per position in testing, and
results are cached, so hovering is free.

The other three sources are cheap closed-form calculations and don't depend on effort.

---

## 8. Putting it together: a full worked example

Take a fogged square early in the game, seen **empty 2 enemy moves ago**, where the enemy
**king cannot be** (it is too far from its starting square). Suppose 32 squares are fogged and
all 16 enemy pieces are hidden, and no ghost reaches this square.

**Source A, material** ($r=0.9$, $F=32$, $h=16$):

| Focal set | $\{P\}$ | $\{R\}$ | $\{B\}$ | $\{N\}$ | $\{Q\}$ | $\{K\}$ | $\{Em\}$ | $\Theta$ |
|---|---|---|---|---|---|---|---|---|
| mass | 0.225 | 0.05625 | 0.05625 | 0.05625 | 0.028125 | 0.028125 | 0.45 | 0.10 |

**Source B, last sighting** ($d=2$): $m(\{Em\}) = 0.95\cdot0.8^2 = 0.608$, $m(\Theta)=0.392$.

**A ⊕ B.** Every piece-type singleton (total mass 0.45) clashes with $\{Em\}$:
$K_1 = 0.45\times0.608 = 0.2736$, so the normaliser is $1-K_1 = 0.7264$.

- $\{Em\}$: $(0.45\cdot0.608 + 0.45\cdot0.392 + 0.10\cdot0.608)/0.7264 = 0.5108/0.7264 = 0.7032$
- $\{P\}$: $0.225\cdot0.392/0.7264 = 0.1214$
- $\{R\},\{B\},\{N\}$: $0.05625\cdot0.392/0.7264 = 0.0304$ each
- $\{Q\},\{K\}$: $0.028125\cdot0.392/0.7264 = 0.0152$ each
- $\Theta$: $0.10\cdot0.392/0.7264 = 0.0540$

**Source C, constraint:** $m(\Theta\setminus\{K\}) = 1$ (hard).

**(A ⊕ B) ⊕ C.** Only $\{K\}$ clashes: $K_2 = 0.0152$, normaliser $0.9848$. $\Theta$ becomes
$\Theta\setminus\{K\}$.

| Hypothesis | m = Bel | Pl | BetP |
|---|---|---|---|
| K King | 0.000 | 0.000 | 0.000 |
| Q Queen | 0.015 | 0.070 | 0.025 |
| B Bishop | 0.031 | 0.086 | 0.040 |
| R Rook | 0.031 | 0.086 | 0.040 |
| N Knight | 0.031 | 0.086 | 0.040 |
| P Pawn | 0.123 | 0.178 | 0.132 |
| Em Empty | 0.714 | 0.769 | 0.723 |

The remaining $m(\Theta\setminus\{K\}) = 0.055$ is spread across all six plausibilities and
split evenly (÷6) in BetP. Total conflict:
$K = 1-(1-0.2736)(1-0.0152) = 0.285$.

**Reading it:** the square is very likely empty (Bel 0.71). If anything is there, it's most
likely a pawn. A king is ruled out entirely. The 0.055 band between Bel and Pl is what the
evidence couldn't decide.

If a ghost knight's random walks ended on this square in 20% of samples, its mass function
$m(\{N\}) = 0.85\times0.2 = 0.17$ would be folded in too. That would raise N, lower Em, and
increase the conflict, because the "seen empty" evidence disagrees with it.

### The Bayesian baseline, for comparison

The *Material frequency (Bayesian baseline)* estimator uses source A alone with $r=1$. Every
focal set is then a single type, so $\mathrm{Bel}=\mathrm{Pl}=\mathrm{BetP}$: an ordinary
probability with no way to express ignorance. On randomly played test games, the average
probability given to the *true* content of a fogged square was:

| Estimator | Mean BetP on the true content |
|---|---|
| Material frequency (Bayesian baseline) | 0.33 |
| Dempster-Shafer (all four sources) | **0.42** |

Random games make for a weak opponent model, so treat this as a sanity check rather than a
benchmark.

---

## 9. Limitations and ideas for future estimators

**Independence is assumed but not fully true.** Dempster's rule assumes the sources are
independent. Material and ghost evidence both encode "there is one hidden knight", so that
information is partly counted twice. The reliability discounts (0.9, 0.85) soften this but don't
remove it. Cautious rules, such as Denœux's cautious conjunctive rule, are designed for dependent
sources.

**Squares are estimated one at a time.** Each square gets its own mass function. Nothing forces
the expected number of hidden knights across all squares to equal the true count, and the
estimates can't express joint statements like "the knight is on c6 *or* d4, not both".

**High conflict can mislead.** Normalising away conflict can produce counter-intuitive results
when $K$ is close to 1 (Zadeh's paradox). The conflict is shown in every tooltip so you can
spot this. Alternatives that keep the conflict instead of normalising it away include Yager's
rule (move conflict to $\Theta$) and Smets' unnormalised rule from the transferable belief
model.

**Simple opponent model.** Ghosts walk uniformly at random among legal non-capturing moves.
Real players develop pieces, push central pawns and avoid danger.

**Hand-tuned constants.** Reliability 0.9 and 0.85, decay 0.8, an 8-piece activity cap, soft
weight 0.97. These are sensible defaults, not fitted values.

### Natural next estimators

The `UncertaintyEstimator` interface is designed so these can be dropped in alongside the
current one:

1. **Joint particle filter.** Sample complete hidden boards that match every observation so far,
   rather than per-square marginals. This fixes the joint-statement limitation above.
2. **Learned opponent policy.** Replace the uniform random walk with move probabilities from a
   trained model. The original Bayesian Chess Bot's network is a natural candidate, which would
   bring that project's work into this one.
3. **Alternative combination rules.** Variants using Yager's or Dubois-Prade rules, chosen from
   the same sidebar, to compare how they handle conflict.
4. **Fitted reliabilities.** Fit the discount factors from self-play logs by maximising the
   pignistic likelihood of the true contents.
