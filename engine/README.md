# wbj — Ruta 2030 Compute Engine

The `wbj` Python package implements every deterministic piece of the
Cerebro v2.0.0 methodology: data fetching, packet validation, ~200
formulas, scoring/gates/overrides, the technical-levels and institutional
valuation engines, and report rendering — behind a staged CLI.

This is the rigorous engine. It's distinct from the zero-API-key MVP
(`wbj analyze`, `wbj scorecard`, `wbj screen`, `python scripts/webapp.py`)
documented in the [root README](../README.md) — see
["Two engines in this repo"](#two-engines-in-this-repo) below.

## Install

```bash
cd engine
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

On Windows, a fresh Python install sometimes fails to verify TLS
certificates against SEC EDGAR/FMP with
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`. Fix:

```bash
pip install pip-system-certs
```

## API keys

Copy your keys into `API/.env` (repo root, already gitignored):

```
FMP_API_KEY=...
FINNHUB_API_KEY=...
FRED_API_KEY=...
```

| Key | Free tier | Used for |
|---|---|---|
| `FMP_API_KEY` | [financialmodelingprep.com](https://financialmodelingprep.com) | Financial statements, adjusted OHLCV, analyst estimates, earnings calendar, insider trades, 13F holders |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io) | Consensus EPS/revenue estimates, quote |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | 10-year Treasury rate (risk-free rate input to WACC) |

SEC EDGAR needs no key. Every provider degrades gracefully when its key
is absent: affected metrics come back `MISSING`/`NOT_SCORABLE` rather
than the pipeline failing, with a note in the report's "Missing /
Conflicted Data" section — never a silently fabricated number.

**Known live-API constraint (found during the Task-25 live smoke
test):** FMP's free/basic tier caps the `limit` query parameter at 5 for
financial-statement endpoints (this package's default), and does not
include institutional-ownership or insider-trading data (those come
back `None` → empty lists in the packet, handled gracefully).

## Stage-by-stage usage

The full engine lives under `wbj engine <stage>` (not the top-level
`wbj analyze` — see below):

```bash
wbj engine fetch NVDA                    # warm the provider cache
wbj engine packet NVDA                   # build + cache the analysis packet
wbj engine compute NVDA --beta 1.3       # build the packet, run all 6 specialists
wbj engine aggregate NVDA --beta 1.3     # + overrides/gates/contradictions
wbj engine report NVDA --beta 1.3        # + charts, report.md, report.json
wbj engine analyze NVDA --beta 1.3       # all of the above in one call
```

Each stage writes a JSON artifact under `engine/cache/<TICKER>/artifacts/`
(the packet, each specialist's output, and `judgment_requests.json` for
anything a specialist couldn't answer mechanically). `report` and
`analyze` also write `Reportes/<TICKER>/<YYYY-MM-DD>/report.md`,
`report.json`, and `charts/*.png`.

`--beta` supplies the bottom-up beta for WACC (Cost of equity =
risk-free + beta × ERP); without it, WACC-dependent Valuation dimensions
come back `NOT_SCORABLE` with a judgment request instead of a fabricated
number, since this packet has no benchmark series to compute beta from
directly. Pass `--offline` to force cache-only, no-network mode (what
the test suite uses).

## The judgment overlay

Some Cerebro metrics are inherently qualitative (moat classification,
catalyst probability/impact, market-size source tier). Specialists
surface these as `judgment_requests` instead of guessing. To answer
them:

1. Run `wbj engine compute <TICKER>` and inspect
   `engine/cache/<TICKER>/artifacts/judgment_requests.json`.
2. Write your answers as a JSON array of judgments:
   ```json
   [
     {
       "request_id": "business.moat_effects_count",
       "answer": 3,
       "evidence_class": "Q",
       "source": "10-K MD&A, competitive strengths section",
       "rationale": "Switching costs, scale cost advantage, and a regulated-protection moat are each independently cited."
     }
   ]
   ```
3. Re-run with `--overlay path/to/judgments.json`:
   ```bash
   wbj engine aggregate NVDA --overlay judgments.json
   ```

`merge_overlay` (`wbj/overlay/merge.py`) validates each judgment (unknown
`request_id`, or a judgment missing `evidence_class`/`source`, is
rejected), re-runs only the affected specialist with the answer folded
into its `overlay` dict, and recomputes that specialist's coverage/score
from scratch.

## Offline mode (tests)

`Provider(..., offline=True)` never touches the network: a cache hit is
served regardless of freshness, and a cache miss returns `None`
immediately. `engine/tests/test_end_to_end.py` populates a real cache
directory from the same fixture JSON the packet-builder tests use, then
runs the full pipeline against it — see that file for the exact
(ticker, cache_key) pairs each provider caches under.

## Two engines in this repo

| | Top-level `wbj <cmd>` | `wbj engine <stage>` |
|---|---|---|
| Data source | SEC EDGAR only (no key needed) | FMP + EDGAR + FinnHub + FRED |
| Scoring | A single quick heuristic scorecard | The full 6-specialist, ~200-formula Cerebro methodology with gates/overrides |
| Entry point | `wbj analyze`, `wbj scorecard`, `wbj screen`, `python scripts/webapp.py` | `wbj engine analyze` |
| Intended use | Zero-setup quick look, the web dashboard | Full audit-trail research report |

Both are real and maintained; they're kept as separate command
namespaces specifically so neither one breaks the other (see
`wbj/cli.py`'s module docstring and `RESUME.md`).

## Tests

```bash
python -m pytest -v
```

220+ tests, offline and deterministic (no network, no API keys
required). The golden fixture `tests/fixtures/golden/NVDA_report.json`
locks the full pipeline's output shape; `tests/fixtures/packet/
NVDA_packet.json` locks the packet builder's.
