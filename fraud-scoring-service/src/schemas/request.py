"""Inbound API schemas (Pydantic v2).

Design notes
------------
* Field names are the WIRE CONTRACT and match `mock-request-body.json` exactly
  ("Time", "V1".."V28", "Amount"). PEP8 naming is deliberately suspended here
  (see `per-file-ignores` in pyproject.toml) — renaming them would break the
  caller.
* `extra="forbid"`: a typo'd or unexpected field is a 422, not a silently
  ignored value. On a fraud path, a caller who thinks they are sending a
  feature that we are dropping is a correctness incident.
* `allow_inf_nan=False`: NaN/Infinity are valid JSON5/Python floats but would
  propagate into the tree comparison and produce garbage. Reject at the edge.
* `frozen=True`: the validated request is immutable, so nothing downstream can
  mutate a feature after validation.
* Vector construction reads `FEATURE_ORDER`, never `self.__dict__` iteration
  order — see `src/ml/feature_order.py` for why that matters.
"""

from __future__ import annotations

from typing import Annotated, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from ml.feature_order import FEATURE_ORDER, N_FEATURES

#: All 30 model inputs are finite floats. Declared once, applied 30 times.
Feature = Annotated[float, Field(allow_inf_nan=False)]

#: scikit-learn's tree module casts inputs to float32 (`sklearn.tree._tree.DTYPE`).
#: Building the array as float32 up front means `check_array` inside
#: `predict_proba` validates and returns it without allocating a copy.
MODEL_DTYPE = np.float32


class Transaction(BaseModel):
    """A single payment transaction's precomputed feature vector."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Time": 0.0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "Amount": 149.62,
            }
        },
    )

    # Optional correlation field — echoed back on the response for the caller's
    # ledger. Not a model input; excluded from FEATURE_ORDER by construction.
    transaction_id: Optional[str] = Field(default=None, alias="transaction-id", max_length=64)

    Time: Feature
    V1: Feature
    V2: Feature
    V3: Feature
    V4: Feature
    V5: Feature
    V6: Feature
    V7: Feature
    V8: Feature
    V9: Feature
    V10: Feature
    V11: Feature
    V12: Feature
    V13: Feature
    V14: Feature
    V15: Feature
    V16: Feature
    V17: Feature
    V18: Feature
    V19: Feature
    V20: Feature
    V21: Feature
    V22: Feature
    V23: Feature
    V24: Feature
    V25: Feature
    V26: Feature
    V27: Feature
    V28: Feature
    Amount: Feature

    def to_vector(self) -> NDArray[np.float32]:
        """Materialise the feature vector in the model's declared column order.

        `np.fromiter` with an explicit `count` allocates exactly once.
        """
        return np.fromiter(
            (getattr(self, name) for name in FEATURE_ORDER),
            dtype=MODEL_DTYPE,
            count=N_FEATURES,
        )


class TransactionBatch(BaseModel):
    """Batch scoring payload.

    Vectorised inference is close to free: the per-call Python/NumPy overhead of
    `predict_proba` dominates a 15-tree/depth-5 forest, so scoring 500 rows costs
    barely more than scoring 1. Offering this endpoint gives batch-capable
    callers ~100x the throughput per vCPU. `max_length` bounds both the payload
    size (DoS control) and the worst-case time the request can occupy a worker.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    transactions: Annotated[list[Transaction], Field(min_length=1, max_length=1000)]

    def to_matrix(self) -> NDArray[np.float32]:
        """Build a C-contiguous (n_rows, 30) float32 matrix."""
        matrix = np.empty((len(self.transactions), N_FEATURES), dtype=MODEL_DTYPE)
        for row, transaction in enumerate(self.transactions):
            matrix[row] = transaction.to_vector()
        return matrix
