"""Packet builder: fetches from FMP, EDGAR, FinnHub, and FRED, maps raw
statement rows to Cerebro canonical field names, reconciles a source-
hierarchy facts table, derives staleness, and hashes the result.

Sources: Cerebro/QUICK_START.md, Cerebro/examples/INPUT_PACKET_EXAMPLE.md,
Cerebro/shared/DATA_POLICY.md ("Staleness defaults"), Cerebro/shared/
DATA_DICTIONARY.md (canonical field names). Reconciliation itself lives in
`wbj.packet.reconcile` (Task 8); staleness classification in
`wbj.packet.staleness` (also Task 10).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.packet.reconcile import reconcile
from wbj.packet.staleness import staleness_state
from wbj.schemas.packet import AnalysisMeta, MarketData, OHLCVRow, Packet, Security

MIN_DAILY_SESSIONS = 252
DEFAULT_INDUSTRY_ADAPTER = "default_nonfinancial"
DEFAULT_SECURITY_TYPE = "operating_company"


class PacketRejected(Exception):
    """Raised when the input data does not meet Task 10's hard-reject bar:
    no currency, no derivable market timestamp, no diluted share count from
    any source, or fewer than 252 daily sessions."""


@dataclass
class Providers:
    """The four data sources the packet builder draws from. Each attribute
    matches the public method surface of its real `wbj.providers.*` class
    (or a test fake with the same surface)."""

    fmp: Any
    edgar: Any
    finnhub: Any
    fred: Any


# --- canonical field mapping (Cerebro/shared/DATA_DICTIONARY.md) -----------

_INCOME_FIELDS = {
    "revenue": "revenue",
    "cogs": "costOfRevenue",
    "gross_profit": "grossProfit",
    "ebit": "operatingIncome",
    "pretax_income": "incomeBeforeTax",
    "income_tax_expense": "incomeTaxExpense",
    "net_income": "netIncome",
    "eps_basic": "eps",
    "eps_diluted": "epsdiluted",
    "diluted_shares": "weightedAverageShsOutDil",
    "basic_shares": "weightedAverageShsOut",
}

_BALANCE_FIELDS = {
    "cash": "cashAndCashEquivalents",
    "receivables": "netReceivables",
    "inventory": "inventory",
    "total_current_assets": "totalCurrentAssets",
    "total_current_liabilities": "totalCurrentLiabilities",
    "short_term_debt": "shortTermDebt",
    "long_term_debt": "longTermDebt",
    "total_debt": "totalDebt",
    "total_assets": "totalAssets",
    "total_liabilities": "totalLiabilities",
    "total_equity": "totalStockholdersEquity",
}

_CASHFLOW_FIELDS = {
    "operating_cash_flow": "netCashProvidedByOperatingActivities",
    "capex": "capitalExpenditure",
    "fcf": "freeCashFlow",
    "acquisitions": "acquisitionsNet",
    "debt_repayment": "debtRepayment",
    "buybacks": "commonStockRepurchased",
    "dividends_paid": "dividendsPaid",
    "stock_based_comp": "stockBasedCompensation",
}

_PASSTHROUGH_KEYS = ("date", "symbol", "period", "calendarYear", "acceptedDate")


def _map_row(raw: dict, field_map: dict[str, str]) -> dict:
    out = {canonical: raw[fmp_key] for canonical, fmp_key in field_map.items() if fmp_key in raw}
    for key in _PASSTHROUGH_KEYS:
        if key in raw:
            out[key] = raw[key]
    return out


def _merge_statements(income: list[dict], balance: list[dict], cashflow: list[dict]) -> list[dict]:
    """Merge income/balance/cashflow rows for the same fiscal `date` into
    one canonical-name record per period, newest-first."""
    by_date: dict[str, dict] = {}
    for raw in income or []:
        by_date.setdefault(raw["date"], {}).update(_map_row(raw, _INCOME_FIELDS))
    for raw in balance or []:
        by_date.setdefault(raw["date"], {}).update(_map_row(raw, _BALANCE_FIELDS))
    for raw in cashflow or []:
        by_date.setdefault(raw["date"], {}).update(_map_row(raw, _CASHFLOW_FIELDS))
    return [by_date[d] for d in sorted(by_date, reverse=True)]


# --- EDGAR XBRL fact lookup --------------------------------------------------


def _edgar_units(companyfacts: dict, taxonomy: str, concept: str, unit: str) -> list[dict]:
    try:
        return companyfacts["facts"][taxonomy][concept]["units"][unit]
    except (KeyError, TypeError):
        return []


def _edgar_fact_at(companyfacts: dict, taxonomy: str, concept: str, unit: str, end: str) -> float | None:
    """The value of `concept` whose `end` date matches `end` exactly, or
    None if the concept or a matching period is absent."""
    for entry in _edgar_units(companyfacts, taxonomy, concept, unit):
        if entry.get("end") == end:
            return entry["val"]
    return None


def _edgar_fact_latest(companyfacts: dict, taxonomy: str, concept: str, unit: str) -> float | None:
    """The most recent (by `end` date) value of `concept`, regardless of
    period alignment — used as a last-resort fallback, e.g. entity shares
    outstanding when no period-matched weighted-diluted tag exists."""
    entries = _edgar_units(companyfacts, taxonomy, concept, unit)
    if not entries:
        return None
    return max(entries, key=lambda e: e["end"])["val"]


# --- facts table reconciliation ---------------------------------------------


def _fmp_value(x: float | None, unit: str) -> Value:
    if x is None:
        return Value.null(NullState.MISSING, unit=unit, source_name="FMP")
    return Value.of(x, unit=unit, source_name="FMP", evidence_class=EvidenceClass.R)


def _edgar_value(x: float | None, unit: str) -> Value:
    if x is None:
        return Value.null(NullState.MISSING, unit=unit, source_name="EDGAR")
    return Value.of(x, unit=unit, source_name="EDGAR", evidence_class=EvidenceClass.R)


def build_packet(ticker: str, providers: Providers, now: datetime) -> Packet:
    """Build the frozen analysis `Packet` for `ticker` as of `now`.

    `now` is the sole clock: never `datetime.now()`, so builds are
    reproducible and hash-stable given identical provider responses.
    """
    ticker = ticker.upper()

    profile_raw = providers.fmp.profile(ticker)
    profile = profile_raw[0] if isinstance(profile_raw, list) and profile_raw else (profile_raw or {})

    currency = profile.get("currency")
    if not currency:
        raise PacketRejected(f"packet rejected for {ticker}: missing reporting currency")

    ohlcv_raw = providers.fmp.ohlcv_daily(ticker) or []
    daily = [
        OHLCVRow(
            date=row["date"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            adj_close=row.get("adjClose", row["close"]),
            volume=row["volume"],
        )
        for row in ohlcv_raw
    ]

    quote = providers.finnhub.quote(ticker) if providers.finnhub else None

    market_timestamp: str | None = None
    if daily:
        market_timestamp = daily[0].date
    elif quote and quote.get("t") is not None:
        market_timestamp = datetime.fromtimestamp(quote["t"], tz=now.tzinfo).isoformat()
    if market_timestamp is None:
        raise PacketRejected(
            f"packet rejected for {ticker}: no market timestamp available from OHLCV or quote"
        )

    if len(daily) < MIN_DAILY_SESSIONS:
        raise PacketRejected(
            f"packet rejected for {ticker}: fewer than {MIN_DAILY_SESSIONS} daily sessions ({len(daily)})"
        )

    annual = _merge_statements(
        providers.fmp.income_annual(ticker) or [],
        providers.fmp.balance_annual(ticker) or [],
        providers.fmp.cashflow_annual(ticker) or [],
    )
    quarterly = _merge_statements(
        providers.fmp.income_quarterly(ticker) or [],
        providers.fmp.balance_quarterly(ticker) or [],
        providers.fmp.cashflow_quarterly(ticker) or [],
    )

    cik = providers.edgar.cik_for(ticker)
    companyfacts = (providers.edgar.companyfacts(cik) if cik is not None else None) or {}

    latest_annual_date = annual[0]["date"] if annual else None

    fmp_diluted = annual[0].get("diluted_shares") if annual else None
    fmp_diluted_q = quarterly[0].get("diluted_shares") if quarterly else None
    edgar_diluted = (
        _edgar_fact_at(
            companyfacts, "us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", latest_annual_date
        )
        if latest_annual_date
        else None
    )
    edgar_basic = _edgar_fact_latest(companyfacts, "dei", "EntityCommonStockSharesOutstanding", "shares")

    if all(x is None for x in (fmp_diluted, fmp_diluted_q, edgar_diluted, edgar_basic)):
        raise PacketRejected(f"packet rejected for {ticker}: no diluted share count from any source")

    fmp_shares_for_facts = fmp_diluted if fmp_diluted is not None else fmp_diluted_q
    edgar_shares_for_facts = edgar_diluted if edgar_diluted is not None else edgar_basic

    edgar_revenue = (
        _edgar_fact_at(companyfacts, "us-gaap", "Revenues", "USD", latest_annual_date) if latest_annual_date else None
    )
    edgar_cash = (
        _edgar_fact_at(companyfacts, "us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD", latest_annual_date)
        if latest_annual_date
        else None
    )
    edgar_lt_debt = (
        _edgar_fact_at(companyfacts, "us-gaap", "LongTermDebtNoncurrent", "USD", latest_annual_date)
        if latest_annual_date
        else None
    )
    edgar_st_debt = (
        _edgar_fact_at(companyfacts, "us-gaap", "DebtCurrent", "USD", latest_annual_date)
        if latest_annual_date
        else None
    )
    edgar_total_debt = (
        edgar_lt_debt + edgar_st_debt if edgar_lt_debt is not None and edgar_st_debt is not None else None
    )

    fmp_revenue = annual[0].get("revenue") if annual else None
    fmp_cash = annual[0].get("cash") if annual else None
    fmp_total_debt = annual[0].get("total_debt") if annual else None
    fmp_price = profile.get("price")

    facts_table = {
        "revenue": reconcile("revenue", _fmp_value(fmp_revenue, "usd"), _edgar_value(edgar_revenue, "usd")),
        "diluted_shares": reconcile(
            "diluted_shares",
            _fmp_value(fmp_shares_for_facts, "shares"),
            _edgar_value(edgar_shares_for_facts, "shares"),
        ),
        "cash": reconcile("cash", _fmp_value(fmp_cash, "usd"), _edgar_value(edgar_cash, "usd")),
        "total_debt": reconcile(
            "total_debt", _fmp_value(fmp_total_debt, "usd"), _edgar_value(edgar_total_debt, "usd")
        ),
        "price": _fmp_value(fmp_price, "usd_per_share"),
    }

    staleness: dict[str, str] = {}
    today = now.date()

    daily_age = (today - date.fromisoformat(daily[0].date)).days
    staleness["daily_market"] = staleness_state("daily_market", daily_age)

    if quarterly:
        q_age = (today - date.fromisoformat(quarterly[0]["date"])).days
        staleness["quarterly_fundamentals"] = staleness_state("quarterly_fundamentals", q_age)

    earnings_calendar = providers.fmp.earnings_calendar(ticker) or []
    actual_prints = [row for row in earnings_calendar if row.get("eps") is not None]
    if actual_prints:
        latest_print_date = max(row["date"] for row in actual_prints)
        consensus_age = (today - date.fromisoformat(latest_print_date)).days
        staleness["consensus"] = staleness_state("consensus", consensus_age)

    holders = providers.fmp.institutional_holders(ticker) or []
    if holders:
        latest_13f_date = max(row["dateReported"] for row in holders)
        peer_age = (today - date.fromisoformat(latest_13f_date)).days
        staleness["peer_set"] = staleness_state("peer_set", peer_age)

    security = Security(
        ticker=ticker,
        exchange=profile.get("exchangeShortName") or "UNKNOWN",
        security_type=DEFAULT_SECURITY_TYPE,
        reporting_currency=currency,
        valuation_currency=currency,
    )
    analysis = AnalysisMeta(
        knowledge_timestamp=now.isoformat(),
        market_timestamp=market_timestamp,
        industry_adapter=DEFAULT_INDUSTRY_ADAPTER,
    )
    market_data = MarketData(daily=daily)

    latest = annual[0] if annual else {}
    capital_structure = {
        "total_debt": latest.get("total_debt"),
        "short_term_debt": latest.get("short_term_debt"),
        "long_term_debt": latest.get("long_term_debt"),
        "total_equity": latest.get("total_equity"),
        "diluted_shares": latest.get("diluted_shares"),
    }

    risk_free = providers.fred.risk_free_rate() if providers.fred else None
    estimates = {
        "finnhub_eps": providers.finnhub.estimates(ticker) if providers.finnhub else None,
        "finnhub_revenue": providers.finnhub.revenue_estimates(ticker) if providers.finnhub else None,
        "fmp_analyst_estimates": providers.fmp.analyst_estimates(ticker),
        "risk_free_rate": risk_free.value if risk_free is not None and risk_free.is_valid else None,
    }

    packet = Packet(
        security=security,
        analysis=analysis,
        fundamentals={"annual": annual, "quarterly": quarterly},
        market_data=market_data,
        estimates=estimates,
        capital_structure=capital_structure,
        insiders=providers.fmp.insider_trades(ticker) or [],
        institutional_holders=holders,
        facts_table=facts_table,
        staleness=staleness,
        packet_hash="",
    )

    payload = packet.model_dump(mode="json", exclude={"packet_hash"})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    packet.packet_hash = digest
    return packet
