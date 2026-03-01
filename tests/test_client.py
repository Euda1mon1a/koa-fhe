"""Tests for koa_fhe SDK — requires live FHE service on port 3410.

Run: ~/.fhe-env/bin/python3 -m pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import koa_fhe


FHE_URL = os.environ.get("FHE_SERVICE_URL", "http://127.0.0.1:3410")


@pytest.fixture(scope="module")
def client():
    """Shared client instance — keygen happens once per module."""
    return koa_fhe.Client(FHE_URL)


# --- Health & Discovery ---


class TestHealth:
    def test_health_status(self, client):
        h = client.health()
        assert h.status == "ok"
        assert len(h.circuits) >= 5
        assert h.uptime_seconds >= 0

    def test_circuits_catalog(self, client):
        circuits = client.circuits()
        assert "threshold" in circuits
        assert "add" in circuits
        assert "multiply" in circuits
        assert circuits["threshold"]["security_level"] == 128
        assert "circuit_hash" in circuits["threshold"]


# --- Arithmetic ---


class TestAdd:
    def test_basic(self, client):
        r = client.add(100, 200)
        assert r.value == 300

    def test_zero(self, client):
        r = client.add(0, 0)
        assert r.value == 0

    def test_large(self, client):
        r = client.add(999, 1)
        assert r.value == 1000

    def test_stats(self, client):
        r = client.add(1, 2)
        assert r.stats.roundtrip_ms > 0
        assert r.stats.server_ms >= 0


class TestMultiply:
    def test_basic(self, client):
        r = client.multiply(7, 8)
        assert r.value == 56

    def test_by_zero(self, client):
        r = client.multiply(42, 0)
        assert r.value == 0

    def test_by_one(self, client):
        r = client.multiply(99, 1)
        assert r.value == 99


class TestCompare:
    def test_greater(self, client):
        r = client.compare(100, 50)
        assert r.greater is True

    def test_not_greater(self, client):
        r = client.compare(50, 100)
        assert r.greater is False

    def test_equal(self, client):
        r = client.compare(75, 75)
        assert r.greater is False  # not strictly greater


# --- Key Dedup ---


class TestKeyDedup:
    def test_second_client_deduplicates(self):
        """Two clients with same circuit should dedup keys on server."""
        c1 = koa_fhe.Client(FHE_URL)
        r1 = c1.add(10, 20)
        assert r1.value == 30

        # Second client — should find existing key via HEAD
        c2 = koa_fhe.Client(FHE_URL)
        r2 = c2.add(30, 40)
        assert r2.value == 70

    def test_dedup_across_circuits(self):
        """Different circuits should NOT dedup (different keys)."""
        c = koa_fhe.Client(FHE_URL)
        r1 = c.add(5, 5)
        assert r1.value == 10
        r2 = c.compare(10, 5)
        assert r2.greater is True


# --- Warm-up ---


class TestWarmUp:
    def test_warm_up_and_ready(self):
        c = koa_fhe.Client(FHE_URL)
        c.warm_up(["add"])

        import time
        time.sleep(3)

        ready = c.ready
        assert ready.get("add") is True

    def test_warm_up_idempotent(self):
        c = koa_fhe.Client(FHE_URL)
        c.warm_up(["add"])
        c.warm_up(["add"])  # should not raise


# --- Schedule Analysis (if circuit is available) ---


class TestRigidity:
    def test_identical_schedules(self, client):
        old = [1, 2, 3, 4, 5] * 10  # 50 slots
        new = list(old)
        r = client.measure_rigidity(old, new)
        assert r.hamming_distance == 0
        assert r.rigidity_score == 1.0
        assert r.severity == "minimal"

    def test_different_schedules(self, client):
        old = [1] * 50
        new = [2] * 50
        r = client.measure_rigidity(old, new)
        assert r.hamming_distance == 50
        assert r.rigidity_score < 1.0
        assert r.severity in ("high", "critical")


class TestAnalyzeSchedule:
    def test_identical(self, client):
        old = [1, 2, 3, 4, 5] * 10
        new = list(old)
        r = client.analyze_schedule(old, new)
        assert r.rigidity_score == 1.0
        assert r.fairness_score == 1.0
        assert r.objective > 0.9

    def test_with_constraints(self, client):
        old = [1, 2, 3, 4, 5] * 10
        new = list(old)
        constraints = [
            {"satisfied": True, "penalty": 0.0},
            {"satisfied": False, "penalty": 1.0},
        ]
        r = client.analyze_schedule(old, new, constraints=constraints)
        assert r.constraint_score < 1.0

    def test_alpha_beta_validation(self, client):
        with pytest.raises(ValueError):
            client.analyze_schedule([1] * 50, [1] * 50, alpha=0.6, beta=0.6)
