"""Koa FHE client — confidential coprocessor for autonomous agents.

Every method auto-triggers circuit setup on first call.
No FHE internals exposed in the public API.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import tempfile
import threading
import time
import uuid

import numpy as np
from concrete import fhe
from concrete.fhe.compilation.specs import ClientSpecs

from ._transport import Transport
from ._types import (
    ArithmeticResult,
    CompareResult,
    RigidityResult,
    ScheduleAnalysis,
    ServiceHealth,
    Stats,
    WorkloadPrediction,
)

_SEVERITY_THRESHOLDS = [
    (0.95, "minimal"),
    (0.85, "low"),
    (0.70, "moderate"),
    (0.50, "high"),
]


def _classify_severity(rigidity: float) -> str:
    for threshold, label in _SEVERITY_THRESHOLDS:
        if rigidity >= threshold:
            return label
    return "critical"


def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
    """Pad array with zeros to target length, or truncate if longer."""
    if len(arr) == target:
        return arr
    if len(arr) > target:
        return arr[:target]
    padded = np.zeros(target, dtype=arr.dtype)
    padded[: len(arr)] = arr
    return padded


class Client:
    """Client for Koa's confidential coprocessor.

    Handles key generation, encryption, and decryption locally.
    The server never sees plaintext data.

    Args:
        server: FHE service URL. Default ``http://127.0.0.1:3410``.
        payment_proof: Base64-encoded x402 payment proof for paid endpoints.
    """

    def __init__(
        self,
        server: str = "http://127.0.0.1:3410",
        *,
        payment_proof: str | None = None,
    ) -> None:
        self._client_id = str(uuid.uuid4())[:8]
        self._transport = Transport(server, self._client_id)
        self._payment_proof = payment_proof
        self._circuits: dict[str, fhe.Client] = {}
        self._circuit_hashes: dict[str, str] = {}  # circuit_name -> hash at keygen time
        self._circuit_info: dict | None = None  # cached from /circuits
        self._ml_client = None  # lazy, needs concrete-ml
        self._ml_tmpdir: str | None = None
        self._warmup_thread: threading.Thread | None = None
        self._warmup_ready: dict[str, bool] = {}
        self._warmup_lock = threading.Lock()
        self._log = logging.getLogger("koa_fhe")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> ServiceHealth:
        """Check service health and available circuits."""
        resp = self._transport.get("health")
        return ServiceHealth(
            status=resp.get("status", "unknown"),
            circuits=resp.get("circuits", []),
            version=resp.get("version", ""),
            evaluations=resp.get("evaluations", 0),
            uptime_seconds=resp.get("uptime_seconds", 0.0),
        )

    def circuits(self) -> dict:
        """List available circuits with versioning info.

        Returns a dict keyed by circuit name, each containing:
        ``circuit_hash``, ``security_level``, ``latency_ms``, ``price_usdc``.
        """
        resp = self._transport.get("circuits")
        return resp.get("circuits", {})

    # Eval key sizes (compressed), ascending. Determines warm_up order.
    _WARMUP_ORDER = ["add", "workload", "rigidity", "time_crystal", "threshold", "multiply"]

    def warm_up(self, circuits: list[str] | None = None) -> None:
        """Pre-load circuits in the background (lightest eval keys first).

        Starts a background thread that downloads specs, generates keys,
        and uploads evaluation keys for each circuit in priority order.
        The default order is by eval key size: add (64B) → workload (48MB)
        → rigidity (25MB) → time_crystal (64MB) → threshold (208MB)
        → multiply (505MB).

        Calling a method before its circuit is warm blocks as usual —
        no behavior change. ``warm_up()`` just front-loads the setup.

        Args:
            circuits: Specific circuits to warm, or ``None`` for all
                (in default priority order).
        """
        if self._warmup_thread and self._warmup_thread.is_alive():
            return  # already warming

        targets = circuits or self._WARMUP_ORDER
        self._warmup_thread = threading.Thread(
            target=self._warmup_worker, args=(targets,), daemon=True
        )
        self._warmup_thread.start()

    @property
    def ready(self) -> dict[str, bool]:
        """Which circuits have completed key setup.

        Returns a dict of ``{circuit_name: True/False}`` for all known
        circuits. A circuit is ready once its keys have been generated
        and uploaded to the server (whether by ``warm_up()`` or by a
        direct call like ``compare()``).
        """
        known = set(self._WARMUP_ORDER) | set(self._warmup_ready.keys())
        with self._warmup_lock:
            return {
                name: name in self._circuits or self._warmup_ready.get(name, False)
                for name in sorted(known)
            }

    def _warmup_worker(self, targets: list[str]) -> None:
        """Background thread: set up circuits in order."""
        for name in targets:
            if name in self._circuits:
                with self._warmup_lock:
                    self._warmup_ready[name] = True
                continue
            try:
                if name == "workload":
                    # Skip workload in warm_up if concrete-ml not available
                    try:
                        from concrete.ml.deployment import FHEModelClient
                        self._setup_ml(FHEModelClient)
                        with self._warmup_lock:
                            self._warmup_ready[name] = True
                        self._log.info("warm_up: workload ready")
                    except ImportError:
                        self._log.debug("warm_up: skipping workload (no concrete-ml)")
                else:
                    self._ensure_circuit(name)
                    with self._warmup_lock:
                        self._warmup_ready[name] = True
                    self._log.info("warm_up: %s ready", name)
            except Exception as e:
                self._log.warning("warm_up: %s failed: %s", name, e)

    def compare(self, a: int, b: int) -> CompareResult:
        """Encrypted comparison: is *a* > *b*?

        Both values are encrypted locally. The server evaluates the
        comparison on ciphertext and returns an encrypted boolean.

        Args:
            a: First value (will be cast to int64).
            b: Second value (will be cast to int64).

        Returns:
            CompareResult with ``.greater`` (bool) and ``.stats``.
        """
        result, stats = self._evaluate("threshold", np.int64(a), np.int64(b))
        return CompareResult(greater=bool(result), stats=stats)

    def add(self, a: int, b: int) -> ArithmeticResult:
        """Encrypted addition: *a* + *b*.

        Args:
            a: First operand (int64).
            b: Second operand (int64).

        Returns:
            ArithmeticResult with ``.value`` (int) and ``.stats``.
        """
        result, stats = self._evaluate("add", np.int64(a), np.int64(b))
        return ArithmeticResult(value=int(result), stats=stats)

    def multiply(self, a: int, b: int) -> ArithmeticResult:
        """Encrypted multiplication: *a* * *b*.

        Args:
            a: First operand (int64).
            b: Second operand (int64).

        Returns:
            ArithmeticResult with ``.value`` (int) and ``.stats``.
        """
        result, stats = self._evaluate("multiply", np.int64(a), np.int64(b))
        return ArithmeticResult(value=int(result), stats=stats)

    def measure_rigidity(
        self,
        old_schedule: list[int] | np.ndarray,
        new_schedule: list[int] | np.ndarray,
    ) -> RigidityResult:
        """Encrypted schedule rigidity — Hamming distance under FHE.

        The server computes how many slots differ between two schedules
        without seeing who is assigned where.

        Inputs are automatically padded to match the circuit's compiled
        slot count (default 50). Padding uses zeros and does not affect
        the Hamming distance of the original data.

        Args:
            old_schedule: Flattened schedule vector (values 0-7).
            new_schedule: Same shape as *old_schedule*.

        Returns:
            RigidityResult with hamming distance, rigidity score, and severity.
        """
        old = np.asarray(old_schedule, dtype=np.int64)
        new = np.asarray(new_schedule, dtype=np.int64)
        n_real = len(old)

        # Pad to circuit's compiled shape
        expected_slots = self._get_circuit_slots("rigidity")
        old = _pad_to(old, expected_slots)
        new = _pad_to(new, expected_slots)

        result, stats = self._evaluate("rigidity", old, new)
        hamming = int(result)
        rigidity = 1.0 - hamming / (2 * n_real) if n_real > 0 else 1.0
        return RigidityResult(
            hamming_distance=hamming,
            rigidity_score=round(rigidity, 6),
            severity=_classify_severity(rigidity),
            stats=stats,
        )

    def analyze_schedule(
        self,
        old_schedule: list[int] | np.ndarray,
        new_schedule: list[int] | np.ndarray,
        *,
        constraints: list[dict] | None = None,
        alpha: float = 0.3,
        beta: float = 0.1,
    ) -> ScheduleAnalysis:
        """Full time crystal objective — rigidity + fairness + constraints.

        Server computes Hamming distance and per-resident churn vector on
        encrypted data. Client computes rigidity, fairness, and constraint
        scores locally.

        Args:
            old_schedule: Flattened schedule (n_residents * n_blocks), values 0-7.
            new_schedule: Same shape.
            constraints: List of dicts with ``satisfied`` (bool) and
                ``penalty`` (float). From the hospital's own evaluation.
            alpha: Rigidity weight (default 0.3).
            beta: Fairness weight (default 0.1).

        Returns:
            ScheduleAnalysis with full objective breakdown.
        """
        if alpha + beta > 1.0:
            raise ValueError(f"alpha ({alpha}) + beta ({beta}) must be <= 1.0")

        old = np.asarray(old_schedule, dtype=np.int64)
        new = np.asarray(new_schedule, dtype=np.int64)
        n_real = len(old)

        # Pad to circuit's compiled shape
        expected_slots = self._get_circuit_slots("time_crystal")
        old = _pad_to(old, expected_slots)
        new = _pad_to(new, expected_slots)

        # Multi-output circuit
        hamming, churn_vector, stats = self._evaluate_multi_output(
            "time_crystal", old, new
        )

        n_slots = n_real

        # Client-side scoring
        rigidity_score = 1.0 - hamming / (2 * n_slots) if n_slots > 0 else 1.0

        # Fairness = 1 - CV(churn)
        if len(churn_vector) > 0:
            mean_churn = float(np.mean(churn_vector))
            std_churn = float(np.std(churn_vector))
            if mean_churn > 0:
                cv = std_churn / mean_churn
                fairness_score = max(0.0, 1.0 - min(cv, 1.0))
            else:
                fairness_score = 1.0
        else:
            fairness_score = 1.0

        # Constraint score (client-side, from hospital's own evaluation)
        if constraints is not None:
            hard_violations = sum(
                1 for r in constraints if not r.get("satisfied", True)
            )
            total_penalty = sum(r.get("penalty", 0.0) for r in constraints)
            max_penalty = 10.0
            constraint_score = max(
                0.0,
                1.0 - (hard_violations * 0.5) - min(total_penalty / max_penalty, 0.5),
            )
        else:
            constraint_score = 1.0

        objective = (
            (1.0 - alpha - beta) * constraint_score
            + alpha * rigidity_score
            + beta * fairness_score
        )

        return ScheduleAnalysis(
            objective=round(objective, 6),
            rigidity_score=round(rigidity_score, 6),
            fairness_score=round(fairness_score, 6),
            constraint_score=round(constraint_score, 6),
            hamming_distance=hamming,
            churn_vector=churn_vector,
            severity=_classify_severity(rigidity_score),
            alpha=alpha,
            beta=beta,
            stats=stats,
        )

    def predict_workload(
        self, features: list[list[float]] | np.ndarray
    ) -> WorkloadPrediction:
        """Encrypted ML inference — XGBoost workload scoring.

        Requires ``pip install koa-fhe[ml]`` (adds ``concrete-ml``).

        Args:
            features: Array of shape ``(n_samples, 23)`` — AAPM workload features.

        Returns:
            WorkloadPrediction with ``.predictions`` list.
        """
        try:
            from concrete.ml.deployment import FHEModelClient
        except ImportError:
            raise RuntimeError(
                "predict_workload() requires concrete-ml. "
                "Install with: pip install koa-fhe[ml]"
            ) from None

        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self._ml_client is None:
            self._setup_ml(FHEModelClient)

        # Encrypt + send + decrypt
        encrypted_data = self._ml_client.quantize_encrypt_serialize(X)

        headers: dict[str, str] = {"X-Client-Id": self._client_id}
        if self._payment_proof:
            headers["X-Payment-Proof"] = self._payment_proof

        t0 = time.perf_counter()
        resp = self._transport.post_json(
            "evaluate/workload",
            {"encrypted_data_b64": base64.b64encode(encrypted_data).decode()},
            headers=headers,
        )
        roundtrip = time.perf_counter() - t0

        result_bytes = base64.b64decode(resp["encrypted_result_b64"])
        predictions = self._ml_client.deserialize_decrypt_dequantize(result_bytes)

        stats = Stats(
            server_ms=resp.get("elapsed_ms", 0),
            roundtrip_ms=round(roundtrip * 1000, 1),
        )
        return WorkloadPrediction(
            predictions=[float(p) for p in predictions.flatten()],
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_circuit_slots(self, circuit_name: str) -> int:
        """Get the compiled slot count for a circuit from the server."""
        if self._circuit_info is None:
            resp = self._transport.get("circuits")
            self._circuit_info = resp.get("circuits", {})
        info = self._circuit_info.get(circuit_name, {})
        return info.get("slots", 50)

    def _ensure_circuit(self, name: str) -> fhe.Client:
        """Download specs, keygen, upload eval keys. Cached per circuit.

        Tracks ``circuit_hash`` from the server. On first call, downloads
        specs and generates keys. Before uploading, checks if the server
        already has keys with the same content hash (dedup). On subsequent
        calls, returns cached client without hitting the server.
        """
        if name in self._circuits:
            return self._circuits[name]

        resp = self._transport.get(f"client-specs/{name}")
        server_hash = resp.get("circuit_hash", "")

        specs_bytes = base64.b64decode(resp["client_specs_b64"])
        specs = ClientSpecs.deserialize(specs_bytes)

        client = fhe.Client(specs)
        client.keygen()
        self._circuits[name] = client
        if server_hash:
            self._circuit_hashes[name] = server_hash

        ek_bytes = client.evaluation_keys.serialize()
        key_hash = hashlib.sha256(ek_bytes).hexdigest()[:16]

        # Check if server already has this key (dedup)
        existing_cid = self._transport.head_key(name, key_hash)
        if existing_cid:
            # Adopt the server's client_id so evaluate requests match
            self._transport.set_client_id(existing_cid)
            self._log.info(
                "Key dedup: %s key already on server (hash=%s, reusing client=%s)",
                name, key_hash, existing_cid,
            )
        else:
            self._transport.post_binary(f"keys/{name}", ek_bytes)

        return client

    def _check_circuit_version(self, name: str, response_hash: str) -> None:
        """Check if server's circuit hash matches our cached keys.

        Called after each evaluation with the hash from the response.
        If mismatched, evicts cached keys so next call triggers re-keygen.
        """
        if not response_hash:
            return
        cached_hash = self._circuit_hashes.get(name, "")
        if cached_hash and cached_hash != response_hash:
            # Circuit was recompiled on the server — evict stale keys
            self._circuits.pop(name, None)
            self._circuit_hashes.pop(name, None)

    def _evaluate(self, circuit_name: str, *args: np.ndarray) -> tuple:
        """Single-output encrypt → send → decrypt."""
        client = self._ensure_circuit(circuit_name)

        encrypted_args = client.encrypt(*args)
        if not isinstance(encrypted_args, tuple):
            encrypted_args = (encrypted_args,)

        args_b64 = [
            base64.b64encode(val.serialize()).decode() for val in encrypted_args
        ]

        headers: dict[str, str] = {}
        if self._payment_proof:
            headers["X-Payment-Proof"] = self._payment_proof

        t0 = time.perf_counter()
        resp = self._transport.post_json(
            f"evaluate/{circuit_name}",
            {"encrypted_args_b64": args_b64},
            headers=headers,
        )
        roundtrip = time.perf_counter() - t0

        result_bytes = base64.b64decode(resp["encrypted_result_b64"])
        result_encrypted = fhe.Value.deserialize(result_bytes)
        result = client.decrypt(result_encrypted)

        # Check for circuit version changes
        self._check_circuit_version(circuit_name, resp.get("circuit_hash", ""))

        stats = Stats(
            server_ms=resp.get("elapsed_ms", 0),
            roundtrip_ms=round(roundtrip * 1000, 1),
        )
        return result, stats

    def _evaluate_multi_output(
        self, circuit_name: str, *args: np.ndarray
    ) -> tuple[int, list[int], Stats]:
        """Multi-output encrypt → send → decrypt (time_crystal)."""
        client = self._ensure_circuit(circuit_name)

        encrypted_args = client.encrypt(*args)
        if not isinstance(encrypted_args, tuple):
            encrypted_args = (encrypted_args,)

        args_b64 = [
            base64.b64encode(val.serialize()).decode() for val in encrypted_args
        ]

        headers: dict[str, str] = {}
        if self._payment_proof:
            headers["X-Payment-Proof"] = self._payment_proof

        t0 = time.perf_counter()
        resp = self._transport.post_json(
            f"evaluate/{circuit_name}",
            {"encrypted_args_b64": args_b64},
            headers=headers,
        )
        roundtrip = time.perf_counter() - t0

        result_values = []
        for r_b64 in resp["encrypted_results_b64"]:
            r_bytes = base64.b64decode(r_b64)
            result_values.append(fhe.Value.deserialize(r_bytes))

        decrypted = client.decrypt(*result_values)
        hamming = int(decrypted[0])
        churn_vector = [int(x) for x in decrypted[1]]

        # Check for circuit version changes
        self._check_circuit_version(circuit_name, resp.get("circuit_hash", ""))

        stats = Stats(
            server_ms=resp.get("elapsed_ms", 0),
            roundtrip_ms=round(roundtrip * 1000, 1),
        )
        return hamming, churn_vector, stats

    def _setup_ml(self, fhe_model_client_cls: type) -> None:
        """Download ML client specs, keygen, upload eval keys."""
        resp = self._transport.get("client-specs/workload")
        server_hash = resp.get("circuit_hash", "")
        if server_hash:
            self._circuit_hashes["workload"] = server_hash
        client_zip_bytes = base64.b64decode(resp["client_zip_b64"])

        self._ml_tmpdir = tempfile.mkdtemp(prefix="koa_fhe_")
        client_zip_path = os.path.join(self._ml_tmpdir, "client.zip")
        with open(client_zip_path, "wb") as f:
            f.write(client_zip_bytes)

        self._ml_client = fhe_model_client_cls(self._ml_tmpdir)
        self._ml_client.load()
        self._ml_client.generate_private_and_evaluation_keys()

        ek_bytes = self._ml_client.get_serialized_evaluation_keys()
        self._transport.post_binary("keys/workload", ek_bytes)
