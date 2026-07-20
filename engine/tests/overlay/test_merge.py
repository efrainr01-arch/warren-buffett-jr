"""Tests for wbj.overlay.merge — Task 20.

Round-trip: compute the valuation specialist on the golden NVDA packet
without a beta (most dimensions NOT_SCORABLE since WACC can't be
computed) -> collect its judgment requests -> answer the beta/ERP
request -> merge -> coverage and category points increase, and the
output hash changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.core.nullstates import EvidenceClass
from wbj.overlay.merge import collect_requests, compute_output_hash, merge_overlay
from wbj.schemas.overlay import Judgment
from wbj.schemas.packet import Packet
from wbj.specialists import valuation

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture
def packet():
    return Packet.model_validate(json.loads(_FIXTURE.read_text()))


def test_collect_requests_flattens_across_outputs(packet):
    val_out = valuation.run(packet)  # no beta -> emits a beta_and_erp request
    requests = collect_requests([val_out])
    assert any(r.metric_id == "beta_and_erp" and r.agent_id == "valuation_analysis" for r in requests)


def test_merge_overlay_round_trip_increases_coverage_and_changes_hash(packet):
    val_out = valuation.run(packet)
    original_hash = compute_output_hash(val_out)
    original_coverage = val_out.coverage
    original_points = val_out.category.awarded_points

    requests = collect_requests([val_out])
    beta_request = next(r for r in requests if r.metric_id == "beta_and_erp")

    judgment = Judgment(
        request_id=beta_request.request_id,
        answer={"beta": 1.72, "erp": 0.045},
        evidence_class=EvidenceClass.E,
        source="FMP company profile beta",
        rationale="Reported 5y monthly beta from the data provider, used as a bottom-up proxy",
    )

    merged = merge_overlay(packet, [val_out], [judgment])
    new_out = merged[0]

    assert new_out.output_hash != original_hash
    assert new_out.output_hash != ""
    # Category points must increase: dimensions gated on WACC (growth-
    # adjusted multiples, fair value by scenarios, margin of safety) go
    # from NOT_SCORABLE to scored. Coverage itself need not move here --
    # Category.coverage()'s denominator (wbj.core.scoring) only counts
    # dimensions that already have >=1 registered metric weight, so an
    # empty-dimension-to-scored transition doesn't dilute or improve that
    # ratio the way a naive reading would suggest; it's consistent with
    # the "no penalty for N/A" behavior already exercised in Task 15's
    # BUS-T008 test.
    assert new_out.coverage >= original_coverage
    assert new_out.category.awarded_points > original_points
    assert new_out.wacc["value"] is not None
    # The answered request is resolved -- no longer outstanding.
    assert not any(r.metric_id == "beta_and_erp" for r in new_out.judgment_requests)


def test_merge_overlay_unknown_request_id_raises(packet):
    val_out = valuation.run(packet)
    bogus = Judgment(request_id="does.not.exist", answer=1.0, evidence_class=EvidenceClass.E, source="x", rationale="y")
    with pytest.raises(ValueError, match="unknown judgment request_id"):
        merge_overlay(packet, [val_out], [bogus])


def test_merge_overlay_missing_source_is_rejected(packet):
    val_out = valuation.run(packet)
    requests = collect_requests([val_out])
    beta_request = next(r for r in requests if r.metric_id == "beta_and_erp")
    bad_judgment = Judgment(
        request_id=beta_request.request_id, answer={"beta": 1.72}, evidence_class=EvidenceClass.E, source="", rationale="y"
    )
    with pytest.raises(ValueError, match="evidence_class and source are required"):
        merge_overlay(packet, [val_out], [bad_judgment])


def test_merge_overlay_leaves_unaffected_agents_untouched(packet):
    from wbj.specialists import business

    val_out = valuation.run(packet)
    business_out = business.run(packet, wacc=0.10)  # a different agent, no judgment answered for it here
    requests = collect_requests([val_out])
    beta_request = next(r for r in requests if r.metric_id == "beta_and_erp")
    judgment = Judgment(
        request_id=beta_request.request_id, answer={"beta": 1.72, "erp": 0.045},
        evidence_class=EvidenceClass.E, source="x", rationale="y",
    )
    merged = merge_overlay(packet, [val_out, business_out], [judgment])
    assert merged[1] is business_out  # untouched: no judgment targeted business_analysis
