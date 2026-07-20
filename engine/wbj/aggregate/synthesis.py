"""Price-level synthesis — Cerebro/00_main_agent/PRICE_LEVEL_SYNTHESIS.md.

Combines the Technical specialist's market-behavior zones with the
Valuation specialist's intrinsic-value references into one ranked table,
flagging confluence where they overlap -- but never averaging a
technical zone with an intrinsic-value reference into a single blended
price (the document's central rule).
"""

from __future__ import annotations

LANGUAGE_WHITELIST = {"reference", "zone", "confirmation", "invalidation", "scenario value"}
FORBIDDEN_PHRASES = {"guaranteed target", "must hold", "certain floor"}


def confluence_tolerance(atr: float, price: float) -> float:
    """max(0.50*ATR14, 0.75% of current price)."""
    return max(0.50 * atr, 0.0075 * price)


def distance_percent(level: float, current_price: float) -> float:
    return (level - current_price) / current_price


def distance_atr(level: float, current_price: float, atr: float) -> float:
    if atr == 0:
        return float("inf") if level != current_price else 0.0
    return (level - current_price) / atr


def find_confluences(levels: list[dict], atr: float, price: float) -> list[tuple[dict, dict]]:
    """Pairs of levels from independent sources -- at least one
    technical -- whose values overlap within the confluence tolerance.
    Never merges or averages: both original level dicts are returned
    unmodified as a pair; the caller only *flags* them."""
    tol = confluence_tolerance(atr, price)
    pairs = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            if a["source"] == "valuation" and b["source"] == "valuation":
                continue
            if abs(a["value"] - b["value"]) <= tol:
                pairs.append((a, b))
    return pairs


def synthesize_levels(technical_output, valuation_output, price: float, atr: float) -> list[dict]:
    """Builds the unified level table (PRICE_LEVEL_SYNTHESIS.md's twelve
    required level classes, as far as each source populates them) with
    distance_percent/distance_atr and confluence flags."""
    levels: list[dict] = []

    important = technical_output.important_levels or {}
    for zone in list(important.get("support", [])) + list(important.get("resistance", [])):
        levels.append(
            {
                "type": zone["type"],
                "source": "technical",
                "value": zone["center"],
                "lower": zone["lower"],
                "upper": zone["upper"],
                "timeframe": zone.get("timeframe"),
                "status": zone.get("status"),
                "strength_0_100": zone.get("strength_0_100"),
                "distance_percent": distance_percent(zone["center"], price),
                "distance_atr": distance_atr(zone["center"], price, atr),
                "confirmation_rule": zone.get("confirmation_rule", ""),
                "invalidation_rule": zone.get("invalidation_rule", ""),
                "confluence": False,
            }
        )

    indicators = technical_output.indicators or {}
    for ma_name in ("sma20", "sma50", "sma100", "sma200"):
        ma_val = indicators.get(ma_name)
        if ma_val is not None:
            levels.append(
                {
                    "type": "moving_average", "source": "technical", "value": ma_val, "label": ma_name,
                    "distance_percent": distance_percent(ma_val, price), "distance_atr": distance_atr(ma_val, price, atr),
                    "confluence": False,
                }
            )

    bands = valuation_output.reference_bands or {}
    for label in ("bear", "base", "bull", "margin_of_safety_15pct", "margin_of_safety_25pct"):
        v = bands.get(label)
        if v is not None:
            levels.append(
                {
                    "type": "intrinsic_value_reference", "source": "valuation", "value": v, "label": label,
                    "distance_percent": distance_percent(v, price), "distance_atr": distance_atr(v, price, atr),
                    "confluence": False,
                }
            )

    levels.append({"type": "current_price", "source": "market", "value": price, "distance_percent": 0.0, "distance_atr": 0.0, "confluence": False})

    non_price = [lvl for lvl in levels if lvl["type"] != "current_price"]
    for a, b in find_confluences(non_price, atr, price):
        a["confluence"] = True
        b["confluence"] = True

    return levels
