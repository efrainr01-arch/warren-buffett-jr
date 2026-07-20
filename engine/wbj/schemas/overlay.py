"""Pydantic model for a judgment answer — Task 20.

`JudgmentRequest` (the question side) lives in `wbj.specialists.common`,
created alongside the specialist output envelope in Task 14. This module
adds the answering side.
"""

from __future__ import annotations

from pydantic import BaseModel

from wbj.core.nullstates import EvidenceClass


class Judgment(BaseModel):
    """An agent-supplied answer to a `JudgmentRequest`.

    Per the Task-20 brief: a judgment without `evidence_class`/`source`
    is rejected by `merge_overlay`, and an unknown `request_id` is an
    error — a judgment must trace to a real, previously-issued request.
    """

    request_id: str
    answer: float | str | dict
    evidence_class: EvidenceClass
    source: str
    rationale: str
