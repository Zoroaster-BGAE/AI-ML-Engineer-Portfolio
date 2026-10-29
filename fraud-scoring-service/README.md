# Fraud Scoring Service

Real-time payment fraud scoring API. `POST localhost:8888/fraud-score` returns
`{"fraud-score": <P(fraud)>}` from a `RandomForestClassifier` (a synthetic demo model, see `scripts/train_demo_model.py`).

```bash
make demo-model                 # once: train the synthetic demo model (Docker, pinned toolchain)
make golden                     # once: record the golden-score fixture inside the production image
make build && make run          # http://localhost:8888
make smoke                      # scores mock-request-body.json
make test                       # 36 tests, inside the production image
```

```bash
curl -X POST http://localhost:8888/fraud-score -H 'Content-Type: application/json' -d @mock-request-body.json
```

```json
{
  "fraud-score": 0.08619879877543643,
  "decision": "allow",
  "model-version": "fraud-model@sklearn-1.0.2+eee319704d7c",
  "request-id": "6b23370dae004f388fc16611c4991baa",
  "scored-at": "2026-07-29T16:05:49.904006Z",
  "latency-ms": 0.674,
  "feature-completeness": "complete",
  "imputed-features": [],
  "feature-source": "request"
}
```

| Endpoint | Purpose |
|---|---|
| `POST /fraud-score` | Score one transaction |
| `POST /fraud-score/batch` | Score up to 1,000 (5.9µs/row vs 2,517µs for a single stock call) |
| `GET /health/live` | Liveness — process up |
| `GET /health/ready` | Readiness — model loaded, validated, warm. ALB target-group check |
| `GET /metrics` | Prometheus: RED metrics + score distribution, flag rate, event-loop lag |
| `GET /docs` | OpenAPI |

---

## 1. Performance

> **Provenance.** The tables below were measured on the original 15-tree, depth-5 reference model this
> service was built around. `make demo-model` trains a synthetic model of the same shape, but these numbers
> were **not** re-measured on it. Re-run `tests/load/http_bench.py` before quoting them.

Everything below is **measured**, not estimated. Server: one container capped
at 1 vCPU. Load generator: a separate container on the same Docker network
(`tests/load/http_bench.py`) — never on the same core as the server.

### Where the time actually goes

| Path | p50 | p99 |
|---|---|---|
| Tree traversal only (`tree_.predict`, 1 tree) | 0.007 ms | 0.028 ms |
| `DecisionTreeClassifier.predict_proba` (1 tree) | 0.114 ms | 0.226 ms |
| **`RandomForestClassifier.predict_proba` (stock, 1 row)** | **2.517 ms** | 4.866 ms |
| **Manual tree loop (fast path, 1 row)** | **0.336 ms** | 0.592 ms |
| `predict_proba`, batch of 500 | 2.965 ms total | 5.9 µs/row |

The forest is 15 trees × depth 5 — about 660 float comparisons, 7 µs of real
work. Where the other 2.5 ms goes, measured directly:

| Component | Cost | Share |
|---|---|---|
| joblib `Parallel` dispatch (15 no-op tasks, `n_jobs=1`) | 0.883 ms | 35% |
| 15 × `tree.predict_proba(check_input=False)` | 0.719 ms | 29% |
| forest-level `check_array` (once) | 0.069 ms | 3% |
| accumulation lock, `np.zeros`, `delayed()` wrappers | ~0.85 ms | 33% |

**Input validation is not the story.** `_accumulate_prediction` calls
`predict(X, check_input=False)`, so `check_array` runs *once* per prediction at
the forest level — not once per tree. The dominant cost is joblib's `Parallel`
machinery, which at `n_jobs=1` buys nothing at all.

`Predictor._predict_proba_fast` skips that redundant re-validation of an input
Pydantic already validated. It is **bitwise identical** to sklearn
(`np.array_equal` over 258 probes at startup + 500 random rows in the test
suite), verified before readiness passes, and disabled by a single env var
(`FRAUD_API_USE_FAST_PATH=false`) if it is ever in doubt.

### End-to-end, over HTTP

| Concurrency | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 | 234 rps | 3.33 ms | 5.35 ms | 50.4 ms |
| 8 | 324 rps | 21.4 ms | 33.7 ms | 69.1 ms |
| 32 | 352 rps | 77.6 ms | 107.5 ms | 177.5 ms |

Server-side histogram over 18,130 requests (this is the equivalent of ALB
`TargetResponseTime`, and excludes client cost and client-side queueing):

```
≤   1 ms   1.9%
≤ 2.5 ms  82.3%
≤   5 ms  92.8%
≤  25 ms  94.4%
≤  50 ms  99.4%
≤ 150 ms 100.0%     ← every request inside the SLO, on ONE vCPU at 32× concurrency
```

**Capacity rule: ~350 rps per vCPU at the SLO.** Size the ECS service from
that, keep per-task concurrency near 8 (p99 69 ms leaves real headroom), and
scale horizontally.

### What was done to get there

1. **Model loaded once** in the lifespan hook, then **warmed** with 25 real
   predictions before readiness returns 200 — the first prediction in a fresh
   process costs 3.5 ms versus 0.3 ms warm, and no customer should pay it.
2. **Threads pinned to 1** (`OMP_NUM_THREADS` et al. in the Dockerfile,
   `n_jobs=1` on the estimator). OpenBLAS defaults to one thread per visible
   core; on a 1-vCPU task that is pure context-switch thrash.
3. **float32 end to end** — sklearn's tree module casts to float32 internally,
   so building the vector as float32 avoids a converted copy per request.
4. **Pure-ASGI middleware instead of `BaseHTTPMiddleware`.** Profiling showed
   the framework floor was larger than the model. The rewrite bought +28%
   saturated throughput and cut p99 at 32× concurrency from 266 ms to 178 ms.
5. **uvicorn's access log muted at the logger.** `--no-access-log` only clears
   uvicorn's handlers; records still propagated to root and were written a
   second time by our JSON handler — a duplicate serialise + write per request.
6. **Batch endpoint**, because per-call overhead dominates and amortises ~430×.

Headroom: at 0.34 ms of inference against a 200 ms budget, **inference is
0.17% of the SLO**. The remaining budget is network, TLS, ASGI, GC and
scheduling. That is where the engineering went.

---

## 2. How I'd deploy it

**ECS Fargate**, 1 vCPU / 2 GB, one uvicorn worker per task, minimum 3 tasks
across 3 AZs, in private subnets behind an internal ALB.

| Option | Verdict |
|---|---|
| **ECS Fargate** | ✅ Chosen. No node management, per-task IAM, right ops/benefit ratio for one stateless service |
| ECS on EC2 | Better $/req at sustained volume (and Graviton), at the cost of AMI/capacity management. The natural migration once volume justifies it |
| EKS | Only if k8s is already the org platform — then consistency beats simplicity. Overkill for one service |
| SageMaker endpoint | Managed autoscaling and data capture, but you give up control of the web tier and pay a premium. Better used as the **Model Registry** for lineage, with ECS serving |
| Lambda | ✗ on the synchronous path — sklearn+NumPy cold starts are incompatible with a 200 ms p99. Fine for offline/replay scoring |

**No API Gateway on the internal path.** It adds 10–30 ms and a second
throttling layer for a service called by our own payment gateway inside the
VPC. ALB + IAM/mTLS (or App Mesh) instead. Add API Gateway only to expose this
externally, where usage plans and edge WAF earn their latency.

**Release**: blue/green via CodeDeploy, 10% canary, automatic rollback on
CloudWatch alarms for p99, 5xx rate **and prediction-score distribution shift**
— that third alarm is what makes it an *ML* deployment. New model versions go
out as shadow traffic first: duplicate live requests to a candidate task set,
compare offline, zero customer risk.

**Scaling**: target tracking on `ALBRequestCountPerTarget` (correlates with
latency; CPU does not, for a service that is 0.34 ms of CPU and mostly waiting)
with a CPU policy as a guardrail. Fargate task start is 30–60 s, so autoscaling
handles trends, not spikes — provision headroom at ~50–60% utilisation and use
scheduled scaling for known payment cycles. No Fargate Spot on the auth path.

---

## 3. Services I'd use

| Layer | Service | Why |
|---|---|---|
| Compute | ECS Fargate | Stateless, per-task IAM, no nodes |
| Ingress | ALB (internal), Route 53 | Health-gated routing, multi-AZ |
| Registry | ECR (immutable tags, digest pinning, scan-on-push) | Reproducible deploys |
| CI/CD | GitHub Actions → Terraform/CDK, CodeDeploy | Infra ships with the code |
| Secrets | Secrets Manager / SSM, IAM task roles | Least privilege |
| Network | VPC endpoints for ECR/S3/CloudWatch/Secrets Manager | Lower latency and cost than NAT; traffic never leaves AWS |
| Metrics | CloudWatch + EMF; ADOT/OTel → AMP + Grafana | EMF emits metrics inside the log line already being written — zero extra API calls on the request path |
| Logs | stdout JSON → FireLens → S3/OpenSearch | Structured, queryable, sampled |
| Tracing | X-Ray or Datadog APM | Needed the moment a feature-store hop exists |
| Model lineage | SageMaker Model Registry | Artifact provenance, approval workflow |
| Features | Kinesis/MSK + Flink, Databricks/Delta, Feast → Redis + DynamoDB | See §5 |

---

## 4. How I'd monitor it

Four layers, because they answer different questions and page different people.

**Infrastructure** — CPU/memory, task count and restarts, OOM kills, ALB
healthy host count, target connection errors, deployment state.

**Service (RED + SLO)** — request rate; errors split **4xx (caller's fault)**
from **5xx (ours)**; latency **histograms**, never averages. Track in-app
latency *and* ALB `TargetResponseTime` separately: the gap between them is
queueing, and it is the first thing to move under load. SLI: "p99 of
`TargetResponseTime` over 5 minutes < 200 ms". Alert on **multi-window
multi-burn-rate error budget**, not static thresholds.

**Model & data** — the layer most services skip:

| Signal | Why |
|---|---|
| Score distribution (`fraud_api_score` histogram) → PSI vs frozen baseline | Cheapest drift signal there is; needs no labels and moves first |
| Flag rate (`fraud_api_decisions_total`) | A step change means an attack or a broken upstream pipeline |
| Per-feature drift (PSI/KS), daily | `Amount` and `Time` are raw and shift first; V1–V28 are PCA components |
| Validation failures by field (`fraud_api_validation_failures_total`) | A caller silently changing units shows up here before anywhere else |
| Feature staleness & completeness | A stale velocity counter is worse than a missing one |
| Training/serving skew | 1% of feature vectors → Firehose → S3, replayed through the offline model |
| **AUC-PR, precision@k, recall at threshold** — *not accuracy* | This dataset is ~0.17% fraud: a model that always says "not fraud" is 99.8% accurate and worthless |
| Calibration curve | We return a *probability*; if it is miscalibrated every downstream threshold is meaningless |
| Event-loop lag (`fraud_api_event_loop_lag_seconds`) | Leading indicator of async starvation — moves before p99 does |

Ground truth arrives late: chargebacks land days-to-weeks after the decision, so
a scheduled job joins predictions to outcomes and trends quality over time.

**Business** — approve/decline volume, **false-positive cost** (blocked good
customers is usually the dominant cost in payments), fraud value caught,
review-queue depth, sliced by merchant and region.

Three dashboards — Service Health, Model Health, Business Impact — and a
runbook entry per alert. An alert without a runbook is noise.

---

## 5. Computing and serving features (when they're not in the request)

Today the caller supplies all 30 features. If they didn't:

**Classify by how they're produced.** (1) Request-time: amount, currency, MCC —
free. (2) Streaming aggregates: txn count per card over 1 m/1 h/24 h, distinct
merchants over 7 d — online store lookup. (3) Batch/slow-moving: merchant
chargeback rate, cardholder tenure — online store, refreshed nightly.
(4) Derived: `amount / avg_amount_7d` — computed in the service from (1)+(2).

**Architecture.** Transactions → Kinesis/MSK → Flink/Spark Structured Streaming
→ windowed aggregates → **ElastiCache Redis** (hot path, same VPC/AZ, ~0.5–2 ms)
with **DynamoDB** as the durable store. Offline: Databricks + Delta Lake on S3,
Spark jobs producing feature tables, training sets built with **point-in-time
correct as-of joins** — the property that prevents label leakage. **Feast** (or
Databricks Feature Store) as the definition/registry layer so training and
serving read one definition.

**Skew prevention is architectural, not procedural**: the transformation logic
lives in one versioned Python package imported by both the Spark job and this
service. Never reimplement a feature in SQL for training and Python for serving.

**Latency discipline**: one batched `MGET`/`BatchGetItem` for every key — never
N sequential lookups; connection pool warmed at startup; in-process TTL/LRU
cache for slow-moving entity features; 10–15 ms hard timeout with a circuit
breaker.

**Degradation — already in the response contract.** On timeout or outage we
impute from the training distribution, return **200** with
`"feature-completeness": "partial"` and the list of `imputed-features`, and emit
a metric. **A fraud model must never block a payment.** Availability beats
marginal accuracy on the authorisation path; the caller applies a more
conservative threshold when completeness is degraded. That fallback *policy* is
a business decision — the API's job is to make the degradation visible and
machine-readable. Cold-start entities fall back hierarchically: entity →
segment (MCC/BIN/country) → global default, never a silent zero, because "no
velocity" and "zero velocity" mean opposite things.

---

## 6. Other considerations

**The pickle is version-locked, and I proved it.** Loading this artifact under
scikit-learn 1.7.0 fails outright:

> `ValueError: node array from the pickle has an incompatible dtype: expected [… 'missing_go_to_left'] itemsize 64, got [… 'weighted_n_node_samples']`

sklearn ≥1.3 added `missing_go_to_left` to the Cython `Node` struct (56 → 64
bytes). A pickle is not a serialization format — it is a snapshot of in-memory
object layout. Hence exact pins on scikit-learn **and** numpy (2.x is an ABI
break), Python 3.9 (1.0.2 ships no wheels above 3.10), and
`tests/contract/test_golden_scores.py` as the canary.

The loader reads the training-time version **from the pickle opcode stream**
with `pickletools` rather than from the object, because
`BaseEstimator.__setstate__` *pops* `_sklearn_version` off the state — reading
it from the loaded estimator always yields `"unknown"`. The check therefore runs
*before* deserialisation and can abort startup without executing the pickle.

**Security.** Unpickling is arbitrary code execution, so the artifact's SHA-256
is injected at build time and verified (constant-time) before the file is
opened. Container runs non-root, no compilers in the runtime stage, no stack
traces on the wire, `extra="forbid"` and bounded batch size as DoS controls,
feature vectors sampled at 1% and never logged wholesale.

**Given more time** — in priority order:

1. **Export to ONNX or Treelite.** Removes the version lock *and* the pickle
   RCE surface, and would beat even the fast path.
2. **Thresholds in a config service** with an audit trail — fraud ops retune
   the precision/recall trade-off continuously; that should not need a deploy.
3. **Champion/challenger + shadow traffic** infrastructure for model rollout.
4. **SHAP attributions** on flagged transactions — analysts and regulators need
   to know why a payment was declined.
5. **`--require-hashes` lockfile**, image signing, SLSA provenance.
6. Terraform for ECS/ALB/alarms (sketched in `infra/`, not written).
7. Retraining cadence + drift-triggered retraining: fraud is adversarial, so
   decay is a certainty, not a risk.

---

## Project layout

```
src/
  main.py                  FastAPI app, lifespan, endpoints, concurrency policy
  ml/feature_order.py      single source of truth for the 30-column contract
  ml/loader.py             integrity → version gate → validate → pin → warm up
  ml/predictor.py          typed inference facade + verified fast path
  schemas/                 Pydantic v2 request/response (incl. degradation contract)
  core/                    settings, structured logging
  observability/           Prometheus metrics, pure-ASGI middleware
tests/
  unit/                    schemas, predictor, thresholds, fast-path equivalence
  integration/             assembled ASGI app: routes, errors, headers, metrics
  contract/                golden scores + dependency/ordering guards
  load/                    locust profile + stdlib HTTP bench + cost attribution
scripts/generate_golden.py regenerates the golden fixture inside the image
models/fraud-model.pickle  the artifact (SHA-256 pinned at build time)
```

**Testing strategy.** Unit tests prove the code does what it says. The
**contract tests** prove the *stack* still produces the same numbers — they are
the ones that catch a dependency bump, a reordered feature list, or an inverted
positive class. `test_json_key_order_does_not_change_the_score` and
`test_permuted_values_do_change_the_score` are a matched pair: the first proves
ordering comes from `FEATURE_ORDER` rather than dict iteration order, the second
proves `to_vector()` actually reads its input. Load tests run against the
**container**, from a **separate container**, because a load test sharing a core
with the server measures the load generator.
