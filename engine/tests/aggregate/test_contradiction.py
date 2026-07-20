"""Tests for wbj.aggregate.contradiction, per Cerebro/00_main_agent/
CONTRADICTION_RESOLUTION.md.
"""

from __future__ import annotations

from wbj.aggregate.contradiction import contradictions

_MAX = {"business": 20, "financial": 15, "market": 20, "technical": 20, "risk": 15, "valuation": 10}


def test_strong_business_weak_technical_flags_quality_watch():
    cats = {"business": 16, "financial": 10, "market": 10, "technical": 6, "risk": 8, "valuation": 5}
    found = contradictions(cats, _MAX)
    assert any(c["label"] == "Quality watch / wait for confirmation" for c in found)


def test_weak_business_strong_technical_flags_speculative_momentum():
    cats = {"business": 6, "financial": 8, "market": 10, "technical": 16, "risk": 8, "valuation": 5}
    found = contradictions(cats, _MAX)
    assert any(c["label"] == "Speculative momentum only" for c in found)


def test_no_contradiction_when_scores_agree():
    cats = {"business": 14, "financial": 10, "market": 14, "technical": 14, "risk": 10, "valuation": 7}
    found = contradictions(cats, _MAX)
    assert found == []


def test_contradictions_never_mutate_input_scores():
    cats = {"business": 16, "financial": 10, "market": 10, "technical": 6, "risk": 8, "valuation": 5}
    original = dict(cats)
    contradictions(cats, _MAX)
    assert cats == original
