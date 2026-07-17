"""WBJ mini web app: search any US-listed company, analyze on demand.

Usage: .venv/bin/python scripts/webapp.py  ->  http://localhost:8765
Stdlib http.server only — no extra dependencies.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from wbj.cli import _build_packet, _compute
from wbj.config import load_settings
from wbj.providers.cache import Cache
from wbj.providers.edgar import (
    _EDGAR_HEADERS,
    _GLOBAL_CACHE_TICKER,
    _MAX_AGE_TICKERS,
    TICKERS_URL,
    EdgarProvider,
)

PORT = 8765
_lock = threading.Lock()

settings = load_settings()
edgar = EdgarProvider(settings, Cache(settings.cache_dir))


def ticker_map() -> list[dict]:
    payload = edgar.get_json(
        TICKERS_URL, {}, "tickers", _GLOBAL_CACHE_TICKER,
        max_age_days=_MAX_AGE_TICKERS, headers=_EDGAR_HEADERS,
    )
    if not isinstance(payload, dict):
        return []
    return [e for e in payload.values() if isinstance(e, dict)]


def search(q: str, limit: int = 8) -> list[dict]:
    q = q.strip().upper()
    if not q:
        return []
    exact, prefix, name = [], [], []
    for e in ticker_map():
        t = str(e.get("ticker", "")).upper()
        n = str(e.get("title", "")).upper()
        row = {"ticker": t, "name": e.get("title", "")}
        if t == q:
            exact.append(row)
        elif t.startswith(q):
            prefix.append(row)
        elif q in n:
            name.append(row)
    return (exact + prefix + name)[:limit]


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WBJ — Company Search</title>
<style>
  :root { color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --border:rgba(11,11,11,.10); --blue:#2a78d6; --blue2:#86b6ef;
    --track:#f0efec; --good:#006300; }
  @media (prefers-color-scheme: dark) { :root { color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --border:rgba(255,255,255,.10); --blue:#3987e5; --blue2:#1c5cab;
    --track:#262624; --good:#0ca30c; } }
  * { margin:0; box-sizing:border-box; }
  body { background:var(--page); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; padding:48px 24px; }
  .wrap { max-width:640px; margin:0 auto; }
  .kicker { font-size:12px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--muted); font-weight:600; margin-bottom:6px; }
  h1 { font-size:24px; margin-bottom:18px; }
  .searchbox { position:relative; }
  input { width:100%; font:inherit; font-size:16px; padding:13px 16px;
    border-radius:10px; border:1px solid var(--border); background:var(--surface);
    color:var(--ink); outline:none; }
  input:focus { border-color:var(--blue); }
  .sugg { position:absolute; top:calc(100% + 6px); left:0; right:0; z-index:5;
    background:var(--surface); border:1px solid var(--border); border-radius:10px;
    overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.12); display:none; }
  .sugg button { display:flex; gap:10px; width:100%; text-align:left; font:inherit;
    font-size:14px; padding:11px 16px; border:0; background:none; color:var(--ink);
    cursor:pointer; border-bottom:1px solid var(--grid); }
  .sugg button:last-child { border-bottom:none; }
  .sugg button:hover, .sugg button.active { background:var(--track); }
  .sugg b { min-width:56px; }
  .sugg span { color:var(--ink2); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #status { margin:22px 2px; color:var(--ink2); font-size:14px; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px;
    padding:22px; margin-top:20px; display:none; }
  .card h2 { font-size:17px; } .fy { color:var(--muted); font-size:12px; margin-top:2px; }
  .hero { display:flex; align-items:baseline; gap:10px; margin:16px 0 2px; }
  .hero .num { font-size:32px; font-weight:700; letter-spacing:-.02em; }
  .hero .unit { font-size:13px; color:var(--ink2); }
  .delta { font-size:13px; font-weight:600; color:var(--good); }
  table { width:100%; border-collapse:collapse; margin:14px 0 18px; font-size:13.5px; }
  td { padding:7px 0; border-bottom:1px solid var(--grid); }
  td:first-child { color:var(--ink2); }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
  tr:last-child td { border-bottom:none; }
  .meter { margin-bottom:13px; }
  .meter .row { display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; }
  .meter .name { color:var(--ink2); } .meter .val { font-weight:700; }
  .track { height:10px; background:var(--track); border-radius:4px; overflow:hidden; }
  .fill { height:100%; border-radius:4px; background:var(--blue); transition:width .5s ease; }
  .fill.b { background:var(--blue2); }
  .total { display:flex; justify-content:space-between; align-items:baseline;
    margin-top:14px; padding-top:13px; border-top:1px solid var(--grid); }
  .total .label { font-size:13px; color:var(--ink2); }
  .total .pts { font-size:20px; font-weight:700; }
  .total .pts small { font-size:13px; font-weight:500; color:var(--muted); }
  .foot { margin-top:20px; color:var(--muted); font-size:12px; line-height:1.6; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid var(--grid);
    border-top-color:var(--blue); border-radius:50%; animation:r .7s linear infinite;
    vertical-align:-2px; margin-right:7px; }
  @keyframes r { to { transform:rotate(360deg); } }
</style></head><body><div class="wrap">
  <div class="kicker">Warren Buffett Jr · Compute Engine · Live SEC EDGAR</div>
  <h1>Search a company</h1>
  <div class="searchbox">
    <input id="q" placeholder="Type a ticker or company name — e.g. NFLX, Disney, Coca-Cola…"
      autocomplete="off" autofocus>
    <div class="sugg" id="sugg"></div>
  </div>
  <div id="status"></div>
  <div class="card" id="card"></div>
  <div class="foot"><b>MVP:</b> Financial category only (15/100 pts), anchor bands are engine
  defaults, missing data never imputed. Research classification — not investment advice.</div>
</div>
<script>
const q = document.getElementById('q'), sugg = document.getElementById('sugg'),
      status = document.getElementById('status'), card = document.getElementById('card');
let timer = null;

q.addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const v = q.value.trim();
    if (!v) { sugg.style.display = 'none'; return; }
    const r = await fetch('/api/search?q=' + encodeURIComponent(v));
    const items = await r.json();
    sugg.innerHTML = items.map(i =>
      `<button data-t="${i.ticker}"><b>${i.ticker}</b><span>${i.name}</span></button>`).join('');
    sugg.style.display = items.length ? 'block' : 'none';
  }, 180);
});
q.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    const first = sugg.querySelector('button');
    analyze(first ? first.dataset.t : q.value.trim().toUpperCase());
  }
});
sugg.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b) analyze(b.dataset.t);
});

const pct = x => typeof x === 'number' ? (x * 100).toFixed(1) + '%' : x;
const n10 = x => typeof x === 'number' ? x.toFixed(1) : x;

async function analyze(t) {
  if (!t) return;
  sugg.style.display = 'none'; card.style.display = 'none';
  q.value = t;
  status.innerHTML = `<span class="spin"></span>Fetching SEC filings + scoring <b>${t}</b>…`;
  try {
    const r = await fetch('/api/analyze?ticker=' + encodeURIComponent(t));
    if (!r.ok) throw new Error((await r.json()).error || r.status);
    const d = await r.json();
    const m = d.metrics, c = d.scores.category, dims = d.scores.dimensions;
    const prof = dims['Profitability'], grow = dims['Growth & Balance Sheet'];
    const rev = m.revenue_usd ? '$' + (m.revenue_usd / 1e9).toFixed(1) + 'B' : '—';
    const yoyNum = typeof m.revenue_yoy === 'number';
    card.innerHTML = `
      <h2>${d.entity} · ${d.ticker}</h2>
      <div class="fy">Fiscal year ended ${d.fiscal_year_end} · Form 10-K · SEC EDGAR</div>
      <div class="hero"><span class="num">${rev}</span><span class="unit">revenue</span>
        ${yoyNum ? `<span class="delta">${m.revenue_yoy >= 0 ? '▲' : '▼'} ${pct(Math.abs(m.revenue_yoy))} YoY</span>` : ''}</div>
      <table>
        <tr><td>Net margin</td><td>${pct(m.net_margin)}</td></tr>
        <tr><td>FCF margin</td><td>${pct(m.fcf_margin)}</td></tr>
        <tr><td>Debt / Equity</td><td>${typeof m.debt_to_equity === 'number' ? m.debt_to_equity.toFixed(2) + '×' : m.debt_to_equity}</td></tr>
      </table>
      <div class="meter"><div class="row"><span class="name">Profitability</span>
        <span class="val">${n10(prof)} / 10</span></div>
        <div class="track"><div class="fill" style="width:0%"></div></div></div>
      <div class="meter"><div class="row"><span class="name">Growth &amp; Balance Sheet</span>
        <span class="val">${n10(grow)} / 10</span></div>
        <div class="track"><div class="fill b" style="width:0%"></div></div></div>
      <div class="total"><span class="label">Financial category · coverage ${(c.coverage * 100).toFixed(0)}%</span>
        <span class="pts">${c.points.toFixed(2)} <small>/ ${c.max_points} pts</small></span></div>`;
    card.style.display = 'block';
    status.textContent = '';
    requestAnimationFrame(() => {
      const fills = card.querySelectorAll('.fill');
      if (typeof prof === 'number') fills[0].style.width = (prof * 10) + '%';
      if (typeof grow === 'number') fills[1].style.width = (grow * 10) + '%';
    });
  } catch (err) {
    status.innerHTML = `Could not analyze <b>${t}</b>: ${err.message}`;
  }
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj: dict | list, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/search":
            self._json(search(qs.get("q", [""])[0]))
        elif url.path == "/api/analyze":
            ticker = qs.get("ticker", [""])[0].strip().upper()
            if not ticker:
                self._json({"error": "missing ticker"}, 400)
                return
            try:
                # One analysis at a time: providers share one httpx client/cache.
                with _lock:
                    result = _compute(_build_packet(ticker))
                self._json(result)
            except Exception as e:  # surface as JSON, keep server alive
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[wbj] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    print(f"WBJ web app -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
