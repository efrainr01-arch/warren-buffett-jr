"""Tests for wbj.aggregate.synthesis, per Cerebro/00_main_agent/
PRICE_LEVEL_SYNTHESIS.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wbj.aggregate.synthesis import confluence_tolerance, distance_atr, distance_percent, find_confluences, synthesize_levels


def _technical(zones_support=None, zones_resistance=None, indicators=None):
    return SimpleNamespace(
        important_levels={"support": zones_support or [], "resistance": zones_resistance or []},
        indicators=indicators or {},
    )


def _valuation(reference_bands=None):
    return SimpleNamespace(reference_bands=reference_bands or {})


# --- confluence tolerance -----------------------------------------------


def test_confluence_tolerance():
    # atr=2, price=100 -> max(0.50*2, 0.0075*100) = max(1.0, 0.75) = 1.0
    assert confluence_tolerance(atr=2.0, price=100.0) == pytest.approx(1.0)


def test_confluence_tolerance_price_dominated():
    # atr=0.1, price=1000 -> max(0.05, 7.5) = 7.5
    assert confluence_tolerance(atr=0.1, price=1000.0) == pytest.approx(7.5)


# --- never average technical and intrinsic ------------------------------


def test_synthesis_never_averages_technical_and_intrinsic():
    zone = {"type": "resistance", "center": 105.0, "lower": 104.0, "upper": 106.0, "status": "confirmed", "strength_0_100": 60.0}
    tech = _technical(zones_resistance=[zone])
    val = _valuation(reference_bands={"base": 105.9})  # within the atr=2 tolerance (1.0) of the zone

    levels = synthesize_levels(tech, val, price=100.0, atr=2.0)

    values = [lvl["value"] for lvl in levels]
    assert 105.0 in values and 105.9 in values
    # No blended/averaged price like (105.0+105.9)/2 = 105.45 was invented.
    assert not any(abs(v - 105.45) < 1e-9 for v in values)

    resistance_level = next(lvl for lvl in levels if lvl["type"] == "resistance")
    base_level = next(lvl for lvl in levels if lvl.get("label") == "base")
    assert resistance_level["confluence"] is True
    assert base_level["confluence"] is True


def test_confluence_requires_at_least_one_technical_side():
    val1 = {"type": "intrinsic_value_reference", "source": "valuation", "value": 100.0}
    val2 = {"type": "intrinsic_value_reference", "source": "valuation", "value": 100.2}
    pairs = find_confluences([val1, val2], atr=2.0, price=100.0)
    assert pairs == []


def test_no_confluence_when_far_apart():
    zone = {"type": "support", "center": 90.0, "lower": 89.0, "upper": 91.0, "status": "confirmed", "strength_0_100": 50.0}
    tech = _technical(zones_support=[zone])
    val = _valuation(reference_bands={"bear": 60.0})  # far outside tolerance

    levels = synthesize_levels(tech, val, price=100.0, atr=2.0)
    assert all(not lvl["confluence"] for lvl in levels if lvl["type"] != "current_price")


# --- distances -----------------------------------------------------------


def test_distance_percent_and_atr():
    assert distance_percent(110.0, 100.0) == pytest.approx(0.10)
    assert distance_atr(110.0, 100.0, atr=5.0) == pytest.approx(2.0)


def test_moving_averages_and_reference_bands_included():
    tech = _technical(indicators={"sma50": 95.0, "sma200": 90.0})
    val = _valuation(reference_bands={"bear": 80.0, "base": 100.0, "bull": 130.0})
    levels = synthesize_levels(tech, val, price=100.0, atr=2.0)
    labels = {lvl.get("label") for lvl in levels}
    assert {"sma50", "sma200", "bear", "base", "bull"} <= labels
