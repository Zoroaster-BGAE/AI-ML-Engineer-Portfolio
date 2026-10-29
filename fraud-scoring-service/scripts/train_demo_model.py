"""Train a synthetic stand-in for the fraud model and write every artifact the service expects.

Why this exists
---------------
The service was built around a RandomForestClassifier supplied by a third party
(15 trees, depth 5, 30 features, scikit-learn 1.0.2). That artifact is not
redistributed. This script produces a model with the same *shape* from synthetic
data, so every code path (integrity check, version gate, warm-up, fast path,
golden contract test) can be exercised end to end. The scores say nothing about
real fraud.

Run it inside the pinned toolchain, because the loader refuses a pickle written
by a different scikit-learn version:

    make demo-model        # python:3.9 + requirements.txt, in Docker

Outputs (deterministic for a given toolchain and --seed):
    models/fraud-model.pickle                         the artifact
    models/fraud-model.pickle.sha256                  its digest
    models/fraud-model.metrics.json                   hold-out metrics (synthetic data)
    mock-request-body.json                            one synthetic legitimate transaction
    tests/contract/fixtures/mock-request-body.json    the same payload, for the contract tests

Afterwards regenerate the golden fixture inside the production image:  make golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

N_ROWS = 200_000
FRAUD_RATE = 0.0017  # same order of magnitude as card-fraud data: ~1 in 600
N_TREES = 15  # asserted by tests/unit/test_predictor.py
MAX_DEPTH = 5  # asserted by tests/unit/test_predictor.py
REVIEW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.80

#: Which synthetic components separate the classes: {V-index (1-based): mean shift in sigmas}.
SHIFTS = {3: -1.6, 4: 1.3, 10: -1.9, 12: -2.1, 14: -2.4, 17: -2.0}


def make_dataset(rng: np.random.Generator, feature_order: tuple[str, ...], n: int):
    """Synthetic transactions in the service's column order. y is float64 on purpose:
    the loader's `classes_` handling is written for float labels."""
    y = (rng.random(n) < FRAUD_RATE).astype(np.float64)
    is_fraud = y == 1.0

    columns = {"Time": rng.uniform(0, 172_800, n)}
    v = rng.standard_normal((n, 28))
    for index, shift in SHIFTS.items():
        v[is_fraud, index - 1] += shift
    for i in range(28):
        columns[f"V{i + 1}"] = v[:, i]

    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n)
    amount[is_fraud] *= rng.uniform(1.0, 3.0, size=int(is_fraud.sum()))
    columns["Amount"] = amount

    return np.column_stack([columns[name] for name in feature_order]), y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root: Path = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from ml.feature_order import FEATURE_ORDER  # noqa: E402  (single source of the column contract)

    rng = np.random.default_rng(args.seed)
    X, y = make_dataset(rng, FEATURE_ORDER, N_ROWS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    positive = int(np.flatnonzero(np.isclose(model.classes_.astype(float), 1.0))[0])
    proba_test = model.predict_proba(X_test)[:, positive]
    metrics = {
        "note": "Synthetic data. These numbers describe the demo model only.",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "n_rows": N_ROWS,
        "fraud_rate": FRAUD_RATE,
        "n_estimators": N_TREES,
        "max_depth": MAX_DEPTH,
        "holdout_auc_pr": round(float(average_precision_score(y_test, proba_test)), 4),
        "holdout_roc_auc": round(float(roc_auc_score(y_test, proba_test)), 4),
        "holdout_flag_rate_at_review": round(float((proba_test >= REVIEW_THRESHOLD).mean()), 5),
        "holdout_flag_rate_at_block": round(float((proba_test >= BLOCK_THRESHOLD).mean()), 5),
    }

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "fraud-model.pickle"
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    (models_dir / "fraud-model.pickle.sha256").write_text(f"{digest}  fraud-model.pickle\n", encoding="utf-8")
    (models_dir / "fraud-model.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # One legitimate-looking transaction that the demo model allows, rounded the way a client would send it.
    payload = None
    for row in X_test[y_test == 0.0]:
        candidate = {name: round(float(value), 6) for name, value in zip(FEATURE_ORDER, row)}
        candidate["Time"] = int(round(candidate["Time"]))
        candidate["Amount"] = round(candidate["Amount"], 2)
        vector = np.array([[candidate[name] for name in FEATURE_ORDER]], dtype=np.float32)
        if float(model.predict_proba(vector)[0, positive]) < REVIEW_THRESHOLD:
            payload = candidate
            break
    if payload is None:  # pragma: no cover - would mean the synthetic world is degenerate
        raise SystemExit("could not find an 'allow' payload; adjust SHIFTS or the seed")

    text = json.dumps(payload, indent=4) + "\n"
    for target in (
        root / "mock-request-body.json",
        root / "tests" / "contract" / "fixtures" / "mock-request-body.json",
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    print(f"model      : {artifact_path}  (sha256 {digest[:16]}...)")
    print(f"toolchain  : scikit-learn {sklearn.__version__}, numpy {np.__version__}")
    print(f"hold-out   : AUC-PR {metrics['holdout_auc_pr']}  ROC-AUC {metrics['holdout_roc_auc']}  (synthetic data)")
    print("payload    : mock-request-body.json (+ tests/contract/fixtures/)")
    print("next       : make golden && make test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
