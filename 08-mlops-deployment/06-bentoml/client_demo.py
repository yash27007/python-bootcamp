"""
Real client hitting the running server.py endpoint with a real HTTP
request via `requests`, plus a load-test loop for the Experiment section.

Requires server.py to already be running on http://127.0.0.1:8000
(started separately, e.g. `.venv/bin/python server.py &`).

Run with:  .venv/bin/python client_demo.py
"""

from __future__ import annotations

import time

import requests

BASE_URL = "http://127.0.0.1:8000"

print("=== Single real request/response ===")
sample = {"features": [5.1, 3.5, 1.4, 0.2]}  # a real setosa sample from Iris
resp = requests.post(f"{BASE_URL}/predict", json=sample)
print(f"request:  POST /predict {sample}")
print(f"status:   {resp.status_code}")
print(f"response: {resp.json()}")

print("\n=== Malformed request (Failure modes: input validation) ===")
bad = {"features": [1, 2]}  # wrong length
resp_bad = requests.post(f"{BASE_URL}/predict", json=bad)
print(f"request:  POST /predict {bad}")
print(f"status:   {resp_bad.status_code}")
print(f"response: {resp_bad.json()}")

print("\n=== Load test: 20 sequential real requests, measuring latency ===")
samples = [
    {"features": [5.1, 3.5, 1.4, 0.2]},   # setosa-like
    {"features": [6.0, 2.7, 5.1, 1.6]},   # versicolor-like
    {"features": [6.9, 3.1, 5.4, 2.1]},   # virginica-like
]
latencies_ms = []
for i in range(20):
    sample = samples[i % len(samples)]
    start = time.perf_counter()
    r = requests.post(f"{BASE_URL}/predict", json=sample)
    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies_ms.append(elapsed_ms)
    assert r.status_code == 200

latencies_ms.sort()
n = len(latencies_ms)
print(f"requests sent: {n}")
print(f"min latency:    {latencies_ms[0]:.3f} ms")
print(f"median latency: {latencies_ms[n // 2]:.3f} ms")
print(f"max latency:    {latencies_ms[-1]:.3f} ms")
print(f"mean latency:   {sum(latencies_ms) / n:.3f} ms")
