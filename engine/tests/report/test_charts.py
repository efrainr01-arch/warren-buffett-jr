"""Tests for wbj.report.charts, per root CLAUDE.md's visualization rules."""

from __future__ import annotations

import pytest

from wbj.report.charts import (
    _scenario_label,
    football_field_chart,
    price_levels_chart,
    scenario_fan_chart,
    scorecard_chart,
)


def test_price_levels_chart_creates_nonempty_file(tmp_path):
    closes = [100.0 + i * 0.5 for i in range(60)]
    levels = [{"type": "resistance", "lower": 118.0, "center": 120.0, "upper": 122.0, "status": "confirmed"}]
    smas = {"sma20": [110.0] * 40}
    out = price_levels_chart(closes, levels, smas, tmp_path / "price_levels.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_scorecard_chart_creates_nonempty_file(tmp_path):
    points = {"business": 16, "financial": 10, "market": 14}
    maxes = {"business": 20, "financial": 15, "market": 20}
    out = scorecard_chart(points, maxes, tmp_path / "scorecard.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_football_field_chart_creates_nonempty_file(tmp_path):
    bands = {"Bear": (60.0, 70.0), "Base": (80.0, 100.0), "Bull": (110.0, 140.0)}
    out = football_field_chart(bands, current_price=90.0, out_path=tmp_path / "football.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_scenario_fan_chart_creates_nonempty_file(tmp_path):
    history = [80.0 + i for i in range(20)]
    scenarios = [
        {"name": "Bear", "growth": 0.03, "margin": 0.10, "low": 90.0, "high": 100.0, "years": 5},
        {"name": "Base", "growth": 0.06, "margin": 0.15, "low": 110.0, "high": 130.0, "years": 5},
        {"name": "Bull", "growth": 0.10, "margin": 0.20, "low": 140.0, "high": 170.0, "years": 5},
    ]
    out = scenario_fan_chart(history, scenarios, tmp_path / "fan.png")
    assert out.exists()
    assert out.stat().st_size > 0


# --- rule 1: never a single line -----------------------------------------


def test_scenario_fan_chart_rejects_zero_width_band(tmp_path):
    history = [100.0] * 10
    scenarios = [{"name": "Base", "growth": 0.05, "margin": 0.15, "low": 100.0, "high": 100.0}]
    with pytest.raises(ValueError, match="single-line projection prohibited"):
        scenario_fan_chart(history, scenarios, tmp_path / "fan.png")


# --- rule 2: label the assumptions --------------------------------------


def test_scenario_label_includes_growth_and_margin():
    label = _scenario_label({"name": "Base", "growth": 0.065, "margin": 0.153})
    assert "growth=" in label
    assert "margin=" in label
    assert "6%" in label
    assert "Base" in label
