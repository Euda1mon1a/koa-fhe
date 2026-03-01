#!/usr/bin/env python3
"""Quickstart — call Koa's confidential coprocessor."""

import koa_fhe

client = koa_fhe.Client()  # default: localhost:3410

# Health check
health = client.health()
print(f"Service: {health.status}, circuits: {health.circuits}")

# Encrypted comparison — server never sees the values
result = client.compare(85, 80)
print(f"85 > 80? {result.greater} ({result.stats.roundtrip_ms}ms)")

# Encrypted addition
result = client.add(500, 300)
print(f"500 + 300 = {result.value} ({result.stats.roundtrip_ms}ms)")

# Encrypted multiplication
result = client.multiply(7, 13)
print(f"7 × 13 = {result.value} ({result.stats.roundtrip_ms}ms)")
