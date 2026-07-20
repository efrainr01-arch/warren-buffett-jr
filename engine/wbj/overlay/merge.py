"""Judgment overlay merge — Task 20.

Collects `JudgmentRequest`s a specialist could not answer mechanically,
accepts `Judgment` answers for them, and re-runs the affected
specialist(s) with those answers folded into their `overlay` dict --
replacing the NOT_SCORABLE metric/dimension, recomputing coverage and
confidence (both fall straight out of re-running the specialist's own
`Category`/`Dimension` math), and assigning a new output hash.

Interface note: the Task-20 brief suggests a `specialist.rescore(output)`
method patched onto a frozen output object. This module instead
re-invokes the specialist's own `run(packet, overlay=..., ...)` --
Tasks 15/16/18/19 (business/market/risk/valuation) already take
`overlay` as their side-channel for judgment answers, so re-running from
the packet is a strict equivalent that reuses the full scoring machinery
instead of hand-patching a frozen output. Financial and Technical
(Tasks 14/17) never emit judgment requests, so they never need rescoring.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from wbj.schemas.overlay import Judgment
from wbj.specialists import business, market, risk, valuation
from wbj.specialists.common import JudgmentRequest, SpecialistOutput

_RUNNERS: dict[str, Callable[..., SpecialistOutput]] = {
    "business_analysis": business.run,
    "market_analysis": market.run,
    "risk_analysis": risk.run,
    "valuation_analysis": valuation.run,
}


def collect_requests(outputs: list[SpecialistOutput]) -> list[JudgmentRequest]:
    """Flatten every specialist output's outstanding judgment requests."""
    return [jr for out in outputs for jr in out.judgment_requests]


def compute_output_hash(output: SpecialistOutput) -> str:
    """sha256 of the output's canonical JSON, excluding its own hash field."""
    payload = output.model_dump(mode="json", exclude={"output_hash"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def merge_overlay(
    packet,
    outputs: list[SpecialistOutput],
    judgments: list[Judgment],
    run_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[SpecialistOutput]:
    """Apply `judgments` to `outputs`, re-running each affected specialist.

    `run_kwargs` optionally supplies extra positional context each
    specialist's `run()` needs beyond `packet`/`overlay` (e.g.
    `{"business_analysis": {"wacc": 0.10}}`) -- values a full pipeline
    would already have computed elsewhere (e.g. from the Financial or
    Valuation specialist) and hands to this call.

    Raises `ValueError` for an unknown `request_id`, or a judgment
    missing `evidence_class`/`source`.
    """
    run_kwargs = run_kwargs or {}
    known_requests = {jr.request_id: jr for jr in collect_requests(outputs)}

    overlay_by_agent: dict[str, dict[str, Any]] = {}
    for j in judgments:
        if j.request_id not in known_requests:
            raise ValueError(f"unknown judgment request_id: {j.request_id!r}")
        if not j.evidence_class or not j.source:
            raise ValueError(f"judgment for {j.request_id!r} rejected: evidence_class and source are required")
        request = known_requests[j.request_id]
        overlay_by_agent.setdefault(request.agent_id, {})[request.metric_id] = j.answer

    merged: list[SpecialistOutput] = []
    for out in outputs:
        agent_overlay = overlay_by_agent.get(out.agent_id)
        if not agent_overlay or out.agent_id not in _RUNNERS:
            merged.append(out)
            continue
        kwargs = dict(run_kwargs.get(out.agent_id, {}))
        kwargs["overlay"] = agent_overlay
        new_out = _RUNNERS[out.agent_id](packet, **kwargs)
        new_out = new_out.model_copy(update={"output_hash": compute_output_hash(new_out)})
        merged.append(new_out)
    return merged
