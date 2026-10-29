"""Attribute per-request cost: how much is framework vs model?"""

import http.client
import json
import os
import statistics
import time

HOST = os.environ.get("TARGET_HOST", "127.0.0.1")
PORT = 8888
PAYLOAD = json.dumps(json.load(open("/app/tests/contract/fixtures/mock-request-body.json")))
H = {"Content-Type": "application/json", "Connection": "keep-alive"}


def bench(method, path, body=None, n=1500, warm=200):
    c = http.client.HTTPConnection(HOST, PORT)
    for _ in range(warm):
        c.request(method, path, body, H)
        c.getresponse().read()
    s = []
    for _ in range(n):
        t = time.perf_counter()
        c.request(method, path, body, H)
        c.getresponse().read()
        s.append((time.perf_counter() - t) * 1000)
    c.close()
    s.sort()
    return statistics.median(s), s[int(n * 0.99)], n / sum(x / 1000 for x in s)


for label, m, p, b in [
    ("GET  /health/live  (framework floor)", "GET", "/health/live", None),
    ("POST /fraud-score  (floor + validate + model)", "POST", "/fraud-score", PAYLOAD),
]:
    p50, p99, rps = bench(m, p, b)
    print(f"{label:<46} p50={p50:6.3f}ms p99={p99:7.3f}ms  serial={rps:5.0f} rps")
