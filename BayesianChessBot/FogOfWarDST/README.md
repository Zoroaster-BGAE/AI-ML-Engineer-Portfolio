# Fog-of-War Chess with Dempster-Shafer Uncertainty

A two-player, hot-seat **Fog-of-War (FOW) chess** game built with Streamlit and `python-chess`,
with a live **Dempster-Shafer** estimator that tells you what might be hiding under the fog.

This project is the follow-up to the [Bayesian Chess Bot](../README.md) in this folder. That
engine had full board information: it encoded each position as a 14×8×8 tensor and used a
Bayesian neural network to predict Stockfish evaluations. Fog-of-war adds *imperfect
information*. You don't just lack an evaluation; you don't know where the enemy pieces are.
Dempster-Shafer Theory (DST) fits that kind of uncertainty better than a single probability
distribution, because it can say "I don't know" explicitly instead of spreading ignorance evenly
over every option.

> The maths behind the estimator is covered in **[EstimatorExplained.md](EstimatorExplained.md)**.

---

## Features

- **Fog-of-war rules.** You only see the squares your pieces stand on or could legally move to.
  Everything else is fog.
- **Chess.com-style controls.** Drag pieces, or click a piece and then a target square. Legal
  moves are shown as dots, captures as rings, and promotions open a piece picker.
- **Hot-seat privacy screen.** The board is hidden between turns so the two players never see
  each other's view.
- **Uncertainty Estimation toggle.** Hover over any fogged square to see a mass assignment over
  King, Queen, Bishop, Rook, Knight, Pawn and Empty, with belief, plausibility and pignistic
  probability for each, plus the conflict between the evidence sources.
- **Estimator Effort slider.** Controls how deep and how many samples the estimator's search
  uses. It plays the same role as search depth in a minimax engine.
- **Pluggable estimators.** The full Dempster-Shafer estimator ships alongside a Bayesian
  baseline (material only), and new estimators can be added with one class.

---

## Quick start (Windows)

Double-click **`run.bat`**, or run it from a terminal in this folder:

```bash
run.bat
```

The first run creates a private virtual environment in `.venv` and installs the requirements
into it, which takes about a minute. After that, the app starts straight away at
**http://localhost:8501**. Press `Ctrl+C` in the terminal to stop it.

### Manual setup (any OS)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python -m streamlit run app.py
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`.

> **Common mistakes**
> - `pip install requirements.txt` fails. You need the `-r` flag: `pip install -r requirements.txt`.
> - `python app.py` (without `streamlit run`) now relaunches itself through Streamlit
>   automatically. Use the `.venv` Python, though, or you'll get `No module named 'chess'`.

**Requirements:** Python 3.10+, `streamlit`, `chess` (see [requirements.txt](requirements.txt)).

---

## How to play

1. **White moves first.** Drag a piece, or click it and then click a highlighted square.
2. **Pass the device.** With the privacy screen on, the board hides after every move. The next
   player clicks *"I'm Black — show my board"*.
3. **There is no check.** You can move into check or ignore it. **You win by capturing the
   enemy king.**
4. **News between turns.** If one of your pieces was captured during the opponent's turn, a
   warning tells you which piece and on which square.
5. When the game ends, the whole board and the full move log are revealed.

### Visibility rules

| Piece | Sees |
|---|---|
| Any piece | Its own square |
| Knight, King | Every square it attacks |
| Bishop, Rook, Queen | Along each line, up to and including the first blocker |
| Pawn | The square ahead only if it's empty (and two ahead from its starting square if both are clear); a diagonal square only if it can capture there, including en passant |

A pawn blocked head-on does **not** see what is blocking it.

---

## Using the uncertainty panel

| Control | Where | What it does |
|---|---|---|
| **Uncertainty Estimation** | Toggle above the board | Turns the estimator on. Fogged squares get hover tooltips. |
| **Uncertainty estimator** | Sidebar | Chooses the estimator: Dempster-Shafer or the Bayesian baseline. |
| **Estimator Effort** (1–10) | Sidebar | Search depth (moves) and samples (60 × effort per tracked piece). |
| **Show best guess on fog** | Sidebar | Labels each fogged square with its most likely piece, e.g. `N 34%`. |
| **Occupancy heat tint** | Sidebar | Tints fogged squares red in proportion to the chance that something is there. |
| **Inspect a fogged square** | Right panel | The same numbers as a table, plus all focal sets and the evidence sources used. |

### Reading a tooltip

```
d2 — under the fog
           m=Bel    Pl     BetP    [Bel, Pl]
K King     0.015   0.066  0.023   ▏
P Pawn     0.596   0.647  0.603   ███████▌
Em Empty   0.230   0.281  0.237   ███▌
Non-singleton mass: m{Θ}=0.051
Conflict K = 0.550 · 6 evidence source(s)
```

- **m = Bel**: mass committed to *exactly* that piece type. For a single hypothesis, belief
  equals mass.
- **Pl**: plausibility, meaning how much of the evidence still *allows* that piece type.
- **The gap between Bel and Pl** is the faded part of the bar. It is uncommitted evidence, or
  ignorance. A wide gap means "it could be this, but nothing specifically points to it".
- **BetP**: pignistic probability (the yellow tick), a single best-guess number for making
  decisions.
- **Conflict K**: how much the evidence sources disagreed before normalisation.

---

## Project structure

```
FogOfWarDST/
├── app.py                  # Everything in Python: DST algebra, FOW rules, estimators, UI
├── fow_board/index.html    # Drag-and-drop board (custom Streamlit component, no build step)
├── requirements.txt
├── run.bat                 # One-click launcher (Windows)
├── README.md               # This file
└── EstimatorExplained.md   # Detailed walkthrough of the estimator
```

`app.py` is organised into seven sections:

| # | Section | Responsibility |
|---|---|---|
| 1 | Frame of discernment & `MassFunction` | Bitmask subsets of Θ, Dempster's rule, Bel / Pl / BetP |
| 2 | Fog-of-war rules | Visibility, bitboard move generation, the shortest-path move tables |
| 3 | `PlayerKnowledge` | Each player's private memory: last sightings, departed pieces, captures |
| 4 | `ObservationContext` | The *only* information an estimator may read, so it can't peek at the real board |
| 5 | Evidence sources & estimators | The four evidence sources, `DempsterShaferEstimator`, the baseline, the registry |
| 6 | Game controller | Session-state game loop: moves, captures, memory updates, game end |
| 7 | Streamlit UI | Sidebar, board component wrapper, move panel, inspector |

### State management

Streamlit reruns the whole script on every interaction, so all game state lives in
`st.session_state`:

- `board`: the true `chess.Board`. It is never sent to the browser in full.
- `knowledge[WHITE]` and `knowledge[BLACK]`: each player's `PlayerKnowledge`.
- `own_log`, `full_log`, `last_move`: per-player move history, plus the full log revealed at
  the end.
- `awaiting_handoff`, `game_over`, `result`: turn and game flow.
- `est_cache`: estimates cached by (position, player, knowledge version, estimator, effort).
  Hovering and toggling never recompute them, and a seeded random generator keeps the numbers
  stable.

The board component only receives the squares the current player can see, so hidden pieces
can't be found by inspecting the page. Moves it sends back are checked against the
pseudo-legal moves for the current ply before they are played.

---

## Adding your own estimator

Subclass `UncertaintyEstimator`, return one `SquareEstimate` per fogged square, and register it:

```python
class MyEstimator(UncertaintyEstimator):
    name = "My estimator"
    description = "Shown under the estimator picker."

    def estimate(self, ctx, effort, rng):
        out = {}
        for sq in squares_of(ctx.fog):
            m = MassFunction({BIT["Em"]: 0.5})        # the rest goes to Θ automatically
            out[sq] = SquareEstimate(m, conflict=0.0, sources=["my rule"], skipped=[])
        return out

ESTIMATORS[MyEstimator.name] = MyEstimator()
```

To add evidence *to* the Dempster-Shafer estimator instead, subclass `EvidenceSource` and pass
it in: `DempsterShaferEstimator(sources=[..., MyEvidence()])`.

---

## Known limitations

- Draws by threefold repetition and the 50-move rule are not detected. Use **Resign** or
  **New game**.
- The evidence sources are not fully independent, which Dempster's rule assumes. See
  [EstimatorExplained.md § Limitations](EstimatorExplained.md#9-limitations-and-ideas-for-future-estimators).
- Tooltips need a mouse. On touch screens, use the *Inspect a fogged square* table.
