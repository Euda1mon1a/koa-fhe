"""Result types for Koa FHE client. No FHE dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stats:
    """Timing stats for an FHE operation."""
    server_ms: float
    roundtrip_ms: float


@dataclass(frozen=True)
class CompareResult:
    """Result of an encrypted comparison (a > b?)."""
    greater: bool
    stats: Stats


@dataclass(frozen=True)
class ArithmeticResult:
    """Result of encrypted arithmetic (add or multiply)."""
    value: int
    stats: Stats


@dataclass(frozen=True)
class RigidityResult:
    """Result of encrypted schedule rigidity analysis."""
    hamming_distance: int
    rigidity_score: float
    severity: str
    stats: Stats


@dataclass(frozen=True)
class ScheduleAnalysis:
    """Full time crystal objective — rigidity + fairness + constraints."""
    objective: float
    rigidity_score: float
    fairness_score: float
    constraint_score: float
    hamming_distance: int
    churn_vector: list[int]
    severity: str
    alpha: float
    beta: float
    stats: Stats


@dataclass(frozen=True)
class WorkloadPrediction:
    """Result of encrypted ML inference (XGBoost workload scoring)."""
    predictions: list[float]
    stats: Stats


@dataclass(frozen=True)
class ServiceHealth:
    """Health status from the FHE service."""
    status: str
    circuits: list[str]
    version: str
    evaluations: int
    uptime_seconds: float
