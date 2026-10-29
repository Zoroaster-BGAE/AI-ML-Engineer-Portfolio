"""Stdlib load generator: keep-alive connections, N threads, real HTTP."""

import http.client
import json
import os
import statistics
import sys
import threading
import time

HOST, PORT = os.environ.get("TARGET_HOST", "127.0.0.1"), 8888
PAYLOAD = json.dumps(json.load(open("/app/tests/contract/fixtures/mock-request-body.json")))
HEADERS = {"Content-Type": "application/json", "Connection": "keep-alive"}
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PER_THREAD = int(sys.argv[2]) if len(sys.argv) > 2 else 400

results, lock = [], threading.Lock()


def worker():
    conn = http.client.HTTPConnection(HOST, PORT)
    local = []
    for _ in range(30):  # warm the connection
        conn.request("POST", "/fraud-score", PAYLOAD, HEADERS)
        conn.getresponse().read()
    for _ in range(PER_THREAD):
        t = time.perf_counter()
        conn.request("POST", "/fraud-score", PAYLOAD, HEADERS)
        r = conn.getresponse()
        r.read()
        local.append(((time.perf_counter() - t) * 1000, r.status))
    conn.close()
    with lock:
        results.extend(local)


threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
start = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.perf_counter() - start

lat = sorted(x[0] for x in results)
errors = sum(1 for x in results if x[1] != 200)
n = len(lat)
print(
    f"concurrency={CONCURRENCY}  requests={n}  errors={errors}  wall={wall:.2f}s  throughput={n/wall:.0f} rps"
)
print(
    f"  p50={statistics.median(lat):.2f}ms  p90={lat[int(n*.90)]:.2f}ms  "
    f"p95={lat[int(n*.95)]:.2f}ms  p99={lat[int(n*.99)]:.2f}ms  max={lat[-1]:.2f}ms"
)
