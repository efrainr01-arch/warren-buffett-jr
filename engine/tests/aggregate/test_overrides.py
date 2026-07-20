"""Tests for wbj.aggregate.overrides.validate_handoff, per Cerebro/
shared/HANDOFF_CONTRACT.md's rejection conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

from wbj.aggregate.overrides import validate_handoff
from wbj.schemas.packet import Packet
from wbj.specialists import financial

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


def test_validate_handoff_accepts_a_real_specialist_output():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = financial.run(packet, wacc=0.10)
    assert validate_handoff(out) == []


def test_validate_handoff_rejects_missing_knowledge_timestamp():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = financial.run(packet, wacc=0.10)
    out.knowledge_timestamp = None
    reasons = validate_handoff(out)
    assert any("knowledge timestamp" in r for r in reasons)


def test_validate_handoff_rejects_missing_coverage():
    packet = Packet.model_validate(json.loads(_FIXTURE.read_text()))
    out = financial.run(packet, wacc=0.10)
    out.coverage = None
    reasons = validate_handoff(out)
    assert any("coverage" in r for r in reasons)
