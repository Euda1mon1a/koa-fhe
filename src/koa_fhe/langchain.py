"""LangChain BaseTool wrappers for Koa FHE operations.

Usage:
    from koa_fhe.langchain import koa_tools

    # Add to any LangChain agent
    agent = initialize_agent(
        tools=koa_tools(),
        llm=ChatOpenAI(),
        agent=AgentType.OPENAI_FUNCTIONS,
    )

Requires: pip install langchain-core
"""

from __future__ import annotations

import json
from typing import Optional

try:
    from langchain_core.tools import BaseTool
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain integration. "
        "Install it with: pip install langchain-core"
    )

import koa_fhe

# Shared lazy client
_client: Optional[koa_fhe.Client] = None


def _get_client(server: str = "http://127.0.0.1:3410") -> koa_fhe.Client:
    global _client
    if _client is None:
        _client = koa_fhe.Client(server)
    return _client


class KoaCompareTool(BaseTool):
    name: str = "koa_encrypted_compare"
    description: str = (
        "Compare two integers under FHE encryption. The server never sees "
        "the values. Returns whether a > b. Use for sealed bids, risk gates, "
        "or compliance checks. Cost: $0.002 USDC."
    )

    def _run(self, a: int, b: int) -> str:
        result = _get_client().compare(int(a), int(b))
        return json.dumps({"greater": result.greater, "ms": result.stats.roundtrip_ms})


class KoaAddTool(BaseTool):
    name: str = "koa_encrypted_add"
    description: str = (
        "Add two integers under FHE encryption. The server never sees "
        "the values. Use for blind aggregation, multi-agent consensus, "
        "or prediction markets. Cost: $0.001 USDC."
    )

    def _run(self, a: int, b: int) -> str:
        result = _get_client().add(int(a), int(b))
        return json.dumps({"value": result.value, "ms": result.stats.roundtrip_ms})


class KoaMultiplyTool(BaseTool):
    name: str = "koa_encrypted_multiply"
    description: str = (
        "Multiply two integers under FHE encryption. The server never sees "
        "the values. Use for portfolio math, encrypted scoring, or private "
        "arithmetic. Cost: $0.002 USDC."
    )

    def _run(self, a: int, b: int) -> str:
        result = _get_client().multiply(int(a), int(b))
        return json.dumps({"value": result.value, "ms": result.stats.roundtrip_ms})


class KoaHealthTool(BaseTool):
    name: str = "koa_fhe_health"
    description: str = (
        "Check Koa FHE service health and available circuits. "
        "Returns status, circuit list, and uptime. Free."
    )

    def _run(self, _input: str = "") -> str:
        h = _get_client().health()
        return json.dumps({
            "status": h.status,
            "circuits": h.circuits,
            "version": h.version,
        })


class KoaRigidityTool(BaseTool):
    name: str = "koa_encrypted_rigidity"
    description: str = (
        "Measure schedule rigidity (Hamming distance) under FHE encryption. "
        "Compares two schedule vectors without the server seeing assignments. "
        "Use for change detection, data integrity checks, or anti-churn. "
        "Cost: $0.05 USDC."
    )

    def _run(self, old_schedule: str, new_schedule: str) -> str:
        old = json.loads(old_schedule) if isinstance(old_schedule, str) else old_schedule
        new = json.loads(new_schedule) if isinstance(new_schedule, str) else new_schedule
        result = _get_client().measure_rigidity(old, new)
        return json.dumps({
            "hamming_distance": result.hamming_distance,
            "rigidity_score": result.rigidity_score,
            "severity": result.severity,
            "ms": result.stats.roundtrip_ms,
        })


class KoaScheduleAnalysisTool(BaseTool):
    name: str = "koa_encrypted_schedule_analysis"
    description: str = (
        "Full encrypted schedule analysis: rigidity + fairness + constraints "
        "under FHE. Server computes Hamming distance and per-resident churn "
        "on ciphertext. Cost: $0.10 USDC."
    )

    def _run(self, old_schedule: str, new_schedule: str) -> str:
        old = json.loads(old_schedule) if isinstance(old_schedule, str) else old_schedule
        new = json.loads(new_schedule) if isinstance(new_schedule, str) else new_schedule
        result = _get_client().analyze_schedule(old, new)
        return json.dumps({
            "objective": result.objective,
            "rigidity_score": result.rigidity_score,
            "fairness_score": result.fairness_score,
            "constraint_score": result.constraint_score,
            "severity": result.severity,
            "ms": result.stats.roundtrip_ms,
        })


def koa_tools(server: str = "http://127.0.0.1:3410") -> list[BaseTool]:
    """Return all Koa FHE tools for use with a LangChain agent.

    Args:
        server: Koa FHE service URL (default: localhost:3410).

    Returns:
        List of BaseTool instances ready to add to an agent.
    """
    global _client
    _client = koa_fhe.Client(server)
    return [
        KoaHealthTool(),
        KoaCompareTool(),
        KoaAddTool(),
        KoaMultiplyTool(),
        KoaRigidityTool(),
        KoaScheduleAnalysisTool(),
    ]
