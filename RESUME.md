# Resume Point — wbj Compute Engine Build

**Status: COMPLETE.** All 25 tasks of `docs/superpowers/plans/2026-07-16-wbj-engine.md`
are implemented, tested, and committed to `main`. Last build session: 2026-07-20.

## What's here

- **Phase 1 (Foundations):** package scaffold, `Value` null-state type, formula
  registry, scoring engine, confidence engine.
- **Phase 2 (Data layer):** response cache, resilient provider base (with an
  `offline` cache-only mode), FMP (`/stable` API), SEC EDGAR, FinnHub, and FRED
  providers, plus the packet builder (canonical field mapping, source-hierarchy
  reconciliation, staleness, hashing).
- **Phase 3 (Math engines):** technical indicators (Wilder ATR/RSI/ADX, MACD,
  composite RS), the important-levels engine (pivots, zones, strength,
  breakouts, AVWAP, volume profile, earnings gaps), and the institutional
  valuation engine (DCF, WACC, reverse DCF, scenarios, Monte Carlo, ensemble).
- **Phase 4 (Specialists):** all six — Financial, Business, Market & Growth,
  Technical & Momentum, Risk & Resilience, Valuation — each scoring against the
  Cerebro formula registry with a shared output envelope.
- **Phase 5 (Assembly):** the judgment overlay (answers qualitative
  `judgment_requests` and re-scores), aggregation (7 mandatory overrides, the
  three profile gates, contradiction detection, price-level synthesis), report
  charts, the final-report renderer (md + json), and the staged CLI pipeline
  (`wbj engine fetch|packet|compute|aggregate|report|analyze`).

**381 tests, all offline/deterministic.** Golden fixtures:
`engine/tests/fixtures/packet/NVDA_packet.json` (packet builder) and
`engine/tests/fixtures/golden/NVDA_report.json` (full pipeline).

**Live-verified:** `wbj engine analyze NVDA` run live against real SEC EDGAR +
FMP data (2026-07-20). See `engine/README.md` for install/usage and the
Task-25 commit for the live-API fixes that came out of that run (FMP's
`/api/v3` → `/stable` migration, tier-driven `limit` caps, a couple of
renamed fields).

## Documented deviations from the original plan/Cerebro text

Each is noted inline in its module's docstring and its commit message; the
short version:

- `relative_strength()` uses Cerebro's literal excess-return definition, not
  the plan's parenthetical "(ratio of n-day returns)".
- Several Financial/Business/Market/Risk formulas that Cerebro describes only
  qualitatively (no numeric band) use a disclosed, documented threshold
  instead — flagged in each module's `assumptions` output.
- The judgment overlay re-runs `specialist.run(packet, overlay=...)` rather
  than patching a frozen output via a `rescore()` method.
- The full engine lives under `wbj engine <stage>`, not the top-level
  `wbj analyze` — that name was already the zero-API-key MVP's entry point
  (`scripts/webapp.py` imports from it directly, and README.md documents it
  as the no-keys quick start). Keeping both avoided breaking either.
- Industry adapters (bank/insurer/REIT/etc.) are out of scope, per the plan's
  own exclusions list — the engine covers the "mature non-financial company"
  path only; other security types get `ADAPTER_UNSUPPORTED` in Valuation's
  model selection.

## If you pick this up again

Natural next steps, none of them blocking:

1. Get FinnHub + FRED API keys (both free) into `API/.env` for full
   consensus-estimate and WACC coverage on live runs.
2. `wbj/providers/fmp.py`'s `institutional_holders`/`insider_trades` endpoint
   paths are best-effort guesses at the `/stable` API shape (this FMP account
   tier returns 402 for both, so they were never confirmed against a real
   response) — worth double-checking against FMP's docs if a higher-tier key
   becomes available.
3. `market_data.benchmark`/`market_data.sector` are never populated by the
   packet builder, so Relative Strength and sector-breadth scoring are
   permanently `NOT_SCORABLE` until that's wired up.
