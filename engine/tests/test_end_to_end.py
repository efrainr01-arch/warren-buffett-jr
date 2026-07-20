"""Offline end-to-end pipeline test + golden-report stability check —
Task 24.

Populates a real `Cache` directory with the same fixture JSON used by
`tests/fixtures/packet/make_packet_fixture.py` (Task 10's golden-packet
generator), under the exact (ticker, cache_key) pairs the real provider
classes use, then runs `run_all(..., offline=True)` -- Provider.offline
(added in Task 24) makes every provider cache-only, so no network call
is ever attempted.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FIXTURES_PACKET_DIR = _FIXTURES_DIR / "packet"
if str(_FIXTURES_PACKET_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_PACKET_DIR))

from make_packet_fixture import FIXED_NOW, generate_ohlcv_sessions  # noqa: E402

from wbj.config import Settings  # noqa: E402
from wbj.pipeline import run_all  # noqa: E402
from wbj.providers.cache import Cache  # noqa: E402

_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden" / "NVDA_report.json"


def _load_fixture(provider: str, name: str):
    return json.loads((_FIXTURES_DIR / provider / f"{name}.json").read_text())


def _populate_cache(cache: Cache, ticker: str) -> None:
    ohlcv = generate_ohlcv_sessions(end=(FIXED_NOW - timedelta(days=1)).date())
    cache.put(ticker, "profile", _load_fixture("fmp", "profile"))
    cache.put(ticker, "ohlcv_daily", {"historical": ohlcv})
    cache.put(ticker, "income_annual", _load_fixture("fmp", "income_annual"))
    cache.put(ticker, "income_quarterly", _load_fixture("fmp", "income_quarterly"))
    cache.put(ticker, "balance_annual", _load_fixture("fmp", "balance_annual"))
    cache.put(ticker, "balance_quarterly", _load_fixture("fmp", "balance_quarterly"))
    cache.put(ticker, "cashflow_annual", _load_fixture("fmp", "cashflow_annual"))
    cache.put(ticker, "cashflow_quarterly", _load_fixture("fmp", "cashflow_quarterly"))
    cache.put(ticker, "earnings_calendar", _load_fixture("fmp", "earnings_calendar"))
    cache.put(ticker, "institutional_holders", _load_fixture("fmp", "institutional_holders"))
    cache.put(ticker, "insider_trades", _load_fixture("fmp", "insider_trades"))
    cache.put(ticker, "analyst_estimates", _load_fixture("fmp", "analyst_estimates"))
    cache.put("_GLOBAL", "tickers", _load_fixture("edgar", "tickers_sample"))
    cache.put("CIK0001045810", "companyfacts", _load_fixture("edgar", "companyfacts_sample"))
    cache.put(ticker, "quote", _load_fixture("finnhub", "quote"))
    cache.put(ticker, "estimates", _load_fixture("finnhub", "eps_estimate"))
    cache.put(ticker, "revenue_estimates", _load_fixture("finnhub", "revenue_estimate"))
    cache.put("_macro", "fred_DGS10", _load_fixture("fred", "dgs10"))


def _settings_for(tmp_path: Path, cache_dir: Path) -> Settings:
    return Settings(
        fmp_api_key="test-fmp-key",
        finnhub_api_key="test-finnhub-key",
        fred_api_key="test-fred-key",
        repo_root=tmp_path,
        cache_dir=cache_dir,
        reports_dir=tmp_path / "Reportes",
    )


@pytest.fixture
def fixture_cache(tmp_path) -> Path:
    cache_dir = tmp_path / "cache"
    cache = Cache(cache_dir)
    _populate_cache(cache, "NVDA")
    return cache_dir


def _normalize(payload: dict) -> dict:
    """Strips fields that legitimately vary run-to-run (wall-clock
    analysis_timestamp, the packet hash) before comparing to the golden
    fixture."""
    payload = json.loads(json.dumps(payload))  # deep copy
    payload.get("security", {}).pop("analysis_timestamp", None)
    payload.get("audit", {}).pop("packet_hashes", None)
    return payload


def test_analyze_offline_end_to_end(tmp_path, fixture_cache):
    settings = _settings_for(tmp_path, fixture_cache)
    final = run_all("NVDA", settings, offline=True, beta=1.72, now=FIXED_NOW)

    assert final.report_version == "2.0.0"
    assert 0 <= final.profile.raw_score <= 100

    out = settings.reports_dir / "NVDA"
    day_dir = next(out.iterdir())
    assert (day_dir / "report.md").exists()
    assert (day_dir / "report.json").exists()
    assert len(list((day_dir / "charts").iterdir())) >= 3


def test_golden_report_stable(tmp_path, fixture_cache):
    settings = _settings_for(tmp_path, fixture_cache)
    final = run_all("NVDA", settings, offline=True, beta=1.72, now=FIXED_NOW)

    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert _normalize(final.model_dump(mode="json")) == _normalize(golden)
