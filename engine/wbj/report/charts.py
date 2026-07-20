"""Report charts — root CLAUDE.md's visualization rules:

1. Never a single line -- always show a range, not one value.
2. Label the assumptions -- every scenario declares its growth/margin.
3. The past is never projected -- history solid, forecast dotted.
4. The agent decides, not the chart -- charts illustrate a decision
   already made by the aggregation logic; they never compute one.

All functions save a PNG at 150 dpi via matplotlib's non-interactive
Agg backend and return `out_path`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150


def _scenario_label(s: dict) -> str:
    """The on-chart assumption label for one scenario (rule 2: label the
    assumptions). Extracted as a pure function so it's testable without
    inspecting a live matplotlib figure."""
    return f"{s['name']}: growth={s['growth']:.0%}, margin={s['margin']:.0%}"


def price_levels_chart(closes: list[float], levels: list[dict], smas: dict[str, list[float]], out_path: Path) -> Path:
    """Price history + shaded support/resistance zone bands (never a
    single drawn line for a zone) + moving averages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(closes)))
    ax.plot(x, closes, color="black", linewidth=1.3, label="Close")

    for name, series in smas.items():
        ax.plot(x[-len(series):], series, linewidth=1.0, alpha=0.8, label=name)

    for lvl in levels:
        color = "tab:red" if lvl["type"] == "resistance" else "tab:green"
        ax.axhspan(lvl["lower"], lvl["upper"], color=color, alpha=0.15)
        ax.text(x[-1], lvl["center"], f"{lvl['type']} {lvl.get('status', '')}", fontsize=7, color=color)

    ax.set_title("Price with support/resistance zones")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def scenario_fan_chart(history: list[float], scenarios: list[dict], out_path: Path) -> Path:
    """History in a solid line; each scenario a *dotted, projected band*
    (low..high), never a single value. Every scenario is labeled on-chart
    with its growth/margin assumptions (rule 2).

    `scenarios`: `[{"name", "growth", "margin", "low", "high", "years"?}]`.
    Raises `ValueError("single-line projection prohibited")` if any
    scenario's low == high (a disguised single-line projection).
    """
    for s in scenarios:
        if s["low"] == s["high"]:
            raise ValueError("single-line projection prohibited")

    fig, ax = plt.subplots(figsize=(10, 6))
    hist_x = list(range(len(history)))
    ax.plot(hist_x, history, color="black", linewidth=1.5, label="History (actual)")

    last_x = len(history) - 1
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for i, s in enumerate(scenarios):
        years = s.get("years", 5)
        proj_x = [last_x, last_x + years]
        color = colors[i % len(colors)]
        ax.plot(proj_x, [history[-1], s["low"]], linestyle=":", color=color)
        ax.plot(proj_x, [history[-1], s["high"]], linestyle=":", color=color)
        ax.fill_between(proj_x, [history[-1], s["low"]], [history[-1], s["high"]], alpha=0.15, color=color)
        ax.text(proj_x[1], (s["low"] + s["high"]) / 2, _scenario_label(s), fontsize=8, color=color)

    ax.set_title("Scenario fan -- projected range, never a single line")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def scorecard_chart(category_points: dict[str, float], category_max: dict[str, float], out_path: Path) -> Path:
    """Horizontal bars: awarded points vs. max points, per category."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(category_points.keys())
    points = [category_points[n] for n in names]
    maxes = [category_max.get(n, 0) for n in names]
    y = range(len(names))

    ax.barh(y, maxes, color="lightgray", label="Max")
    ax.barh(y, points, color="tab:blue", label="Awarded")
    ax.set_yticks(list(y), names)
    ax.set_xlabel("Points")
    ax.set_title("Category scorecard")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def football_field_chart(reference_bands: dict[str, tuple[float, float]], current_price: float, out_path: Path) -> Path:
    """Valuation ranges (low, high) per model/scenario, as horizontal
    bars, with a vertical line marking the current price -- never a
    single point per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(reference_bands.keys())
    for i, name in enumerate(names):
        low, high = reference_bands[name]
        ax.plot([low, high], [i, i], linewidth=8, alpha=0.5, solid_capstyle="butt", color="tab:blue")
        ax.text(high, i, f" {name}: {low:.0f}-{high:.0f}", fontsize=8, va="center")

    ax.axvline(current_price, color="black", linestyle="--", label=f"Current price {current_price:.2f}")
    ax.set_yticks(list(range(len(names))), names)
    ax.set_title("Football field -- valuation reference ranges")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path
