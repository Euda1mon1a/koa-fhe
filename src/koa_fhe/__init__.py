"""koa-fhe — Confidential coprocessor client for autonomous agents.

Compute on encrypted data via Koa's FHE service. The server never sees
plaintext. Every query pays $0.001-$0.10 USDC via x402.

    import koa_fhe

    client = koa_fhe.Client()
    result = client.compare(85, 80)
    print(result.greater)  # True
"""

__version__ = "0.1.1"

from ._client import Client
from ._types import (
    ArithmeticResult,
    CompareResult,
    RigidityResult,
    ScheduleAnalysis,
    ServiceHealth,
    Stats,
    WorkloadPrediction,
)

__all__ = [
    "Client",
    "ArithmeticResult",
    "CompareResult",
    "RigidityResult",
    "ScheduleAnalysis",
    "ServiceHealth",
    "Stats",
    "WorkloadPrediction",
]
