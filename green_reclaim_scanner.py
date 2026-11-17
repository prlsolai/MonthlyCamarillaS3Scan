#!/usr/bin/env python3
"""Monthly Green Candle Reclaims S3 (NSE).

Rule (monthly candles, yearly Camarilla levels from the previous calendar year):
  1. Price dipped into the S3-S4 zone: lowest low of the signal month and the
     previous LB months is at/below S3 (+TOL) and not far below S4 (-TOL).
  2. Signal month is a GREEN candle that closes ABOVE S3.
  3. Previous month closed at/below S3 (so this is a fresh reclaim).
Filtered version also needs: price below 20M average two months earlier and
monthly RSI(14) under 40 at some point in the last 3 months.

Run on the 1st of each month. Usage: python green_reclaim_scanner.py [symbols.txt] [docs/green_reclaim.html]
"""
import json, sys
from pathlib import Path
import pandas as pd
import yfinance as yf

TOL = 0.015
LB = 2                 # touch may be in the signal month or the 2 months before
RISK, RR = 500, 3
HORIZONS = (3, 6, 12)
SMA_N, RSI_N, RSI_MAX = 20, 14, 40
DEFAULT = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "LT", "ITC",
           "BHARTIARTL", "AXISBANK", "MARUTI", "TATASTEEL", "SUNPHARMA", "NTPC", "ONGC"]


def load_symbols(path):
    p = Path(path)
    syms = p.read_text().split() if p.exists() else DEFAULT
    return [s if s.endswith(".NS") else s + ".NS" for s in syms]


def yearly_camarilla(m):
    y = m.groupby(m.index.year).agg(High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    prev = y.shift(1).reindex(m.index.year)
    prev.index = m.index
    rng = prev.High - prev.Low
    m["S3"] = prev.Close - rng * 1.1 / 4
    m["S4"] = prev.Close - rng * 1.1 / 2
    return m


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def process(sym, cur):
    m = yf.download(sym, period="max", interval="1mo", auto_adjust=True, progress=False)
    if m is None or m.empty:
        return None
    if isinstance(m.columns, pd.MultiIndex):
        m.columns = m.columns.get_level_values(0)
    m = m[["Open", "High", "Low", "Close"]].dropna()
    m = m[m.index.to_period("M") < cur]
    if len(m) < 30:
        return None
    m = yearly_camarilla(m)
    o, l, c = m.Open, m.Low, m.Close
    m["RSI"] = rsi(c, RSI_N)
    sma = c.rolling(SMA_N).mean()
    m["tlow"] = l.rolling(LB + 1).min()
    zone = (m.tlow <= m.S3 * (1 + TOL)) & (m.tlow >= m.S4 * (1 - TOL))
    reclaim = (c > o) & (c > m.S3) & (c.shift(1) <= m.S3)
    m["base"] = (reclaim & zone).fillna(False)
    filt = (m.RSI.rolling(3).min() < RSI_MAX) & (c.shift(2) < sma.shift(2))
    m["filt"] = (m.base & filt).fillna(False)
    for h in HORIZONS:
        m[f"f{h}"] = c.shift(-h) / c - 1
    return m


def stats(df):
    row = {"n": int(len(df))}
    for k in HORIZONS:
        v = df[f"f{k}"].dropna() if len(df) else pd.Series(dtype=float)
        row[f"w{k}"] = round(float((v > 0).mean() * 100), 1) if len(v) else None
        row[f"a{k}"] = round(float(v.mean() * 100), 1) if len(v) else None
    return row


def main():
    sym_file = sys.argv[1] if len(sys.argv) > 1 else "symbols.txt"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/green_reclaim.html")
    today = pd.Timestamp.today()
    cur = today.to_period("M")
    target = cur - 1
    hist, base, latest, scanned = [], [], [], 0
    for s in load_symbols(sym_file):
        try:
            m = process(s, cur)
        except Exception as e:
            print("skip", s, e)
            continue
        if m is None:
            continue
        scanned += 1
        cols = [f"f{h}" for h in HORIZONS]
        base.append(m[cols])
        sig = m[m.base]
        hist.append(sig[cols + ["filt"]])
        for d, r in sig[sig.index.to_period("M") == target].iterrows():
            entry, stop = float(r.High), float(r.tlow)
            risk = entry - stop
            if risk <= 0:
                continue
            latest.append({
                "symbol": s[:-3], "month": d.strftime("%b %Y"), "filt": bool(r.filt),
                "close": round(float(r.Close), 2), "S3": round(float(r.S3), 2),
                "S4": round(float(r.S4), 2), "reclaim": round((float(r.Close) / float(r.S3) - 1) * 100, 1),
                "rsi": round(float(r.RSI), 1), "entry": round(entry, 2), "stop": round(stop, 2),
                "target": round(entry + RR * risk, 2), "risk_pct": round(risk / entry * 100, 1),
                "qty": int(RISK // risk)})
    h = pd.concat(hist) if hist else pd.DataFrame(columns=[f"f{x}" for x in HORIZONS] + ["filt"])
    b = pd.concat(base) if base else pd.DataFrame(columns=[f"f{x}" for x in HORIZONS])
    latest.sort(key=lambda x: (not x["filt"], x["symbol"]))
    bt = [{"name": "Green candle reclaims S3", **stats(h)},
          {"name": "+ RSI & downtrend filters", **stats(h[h.filt.astype(bool)])},
          {"name": "Any month (baseline)", **stats(b)}]
    data = {"generated": today.strftime("%d %b %Y"), "signal_month": target.strftime("%b %Y"),
            "scanned": scanned, "risk": RISK, "rr": RR, "tol": TOL * 100,
            "horizons": list(HORIZONS), "signals": latest, "backtest": bt}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"{scanned} scanned, {len(latest)} signals for {data['signal_month']} -> {out}")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monthly Green Candle Reclaims S3</title>
<style>
:root{--bg:#0f1419;--card:#1a2129;--tx:#e6edf3;--mu:#8b98a5;--ac:#3fb950;--rd:#f85149;--bd:#2b3640}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 system-ui,sans-serif;padding:12px;max-width:760px;margin:auto}
h1{font-size:19px;margin:4px 0}h2{font-size:16px;margin:18px 0 8px}.mu{color:var(--mu);font-size:13px}a{color:#58a6ff}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin:10px 0}
.row{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}.g{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}
.g div{background:#121820;border-radius:6px;padding:6px;text-align:center}.g b{display:block;font-size:14px}.g span{font-size:11px;color:var(--mu)}
.t{color:var(--ac)}.s{color:var(--rd)}.badge{background:#1f3a2a;color:var(--ac);border-radius:6px;padding:2px 8px;font-size:12px}.badge.n{background:#2b3640;color:var(--mu)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 4px;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}.sc{overflow-x:auto}
input[type=text]{width:100%;background:#121820;border:1px solid var(--bd);color:var(--tx);border-radius:6px;padding:6px;margin-top:8px}
label{display:flex;gap:8px;align-items:center;margin-top:8px}
</style></head><body>
<h1>Monthly Green Candle Reclaims S3</h1>
<div class="mu" id="meta"></div>
<div class="mu">Green monthly candle closes back above yearly Camarilla S3 after a dip into the S3-S4 zone. <a href="index.html">Pattern scan</a></div>
<h2>Signals for <span id="sm"></span></h2><div id="sig"></div>
<h2>Backtest</h2>
<div class="card sc"><table id="bt"></table></div>
<div class="mu">Win % / average return after N months from the signal close, compared with any month. Past results do not guarantee future returns.</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const f=n=>Number(n).toLocaleString('en-IN',{maximumFractionDigits:2});
let st={};try{st=JSON.parse(localStorage.getItem('gr3')||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem('gr3',JSON.stringify(st))}catch(e){}};
document.getElementById('meta').textContent=`Updated ${D.generated} | ${D.scanned} stocks scanned | zone \u00b1${D.tol}% | risk \u20b9${D.risk}/trade | target ${D.rr}R`;
document.getElementById('sm').textContent=D.signal_month;
const box=document.getElementById('sig');
if(!D.signals.length)box.innerHTML='<div class="card mu">No signals this month.</div>';
D.signals.forEach(x=>{
 const k=x.symbol+x.month,s=st[k]||{};
 const c=document.createElement('div');c.className='card';
 c.innerHTML=`<div class="row"><b>${x.symbol}</b><span class="badge ${x.filt?'':'n'}">${x.filt?'RSI + trend filters \u2713':'Base reclaim'}</span></div>
 <div class="mu">${x.month} close \u20b9${f(x.close)} (${x.reclaim}% above S3) | S3 \u20b9${f(x.S3)} | S4 \u20b9${f(x.S4)} | RSI ${x.rsi}</div>
 <div class="g"><div><b>\u20b9${f(x.entry)}</b><span>Entry (high)</span></div><div><b class="s">\u20b9${f(x.stop)}</b><span>Stop (${x.risk_pct}%)</span></div><div><b class="t">\u20b9${f(x.target)}</b><span>Target</span></div></div>
 <div style="margin-top:8px">Qty for \u20b9${D.risk} risk: <b>${x.qty}</b></div>
 <label><input type="checkbox" ${s.taken?'checked':''}> Trade taken</label>
 <input type="text" placeholder="Notes / skip reason" value="${(s.note||'').replace(/"/g,'&quot;')}">`;
 const [cb,tx]=c.querySelectorAll('input');
 cb.onchange=()=>{st[k]={...st[k],taken:cb.checked};save()};
 tx.oninput=()=>{st[k]={...st[k],note:tx.value};save()};
 box.appendChild(c);
});
let h='<tr><th>Rule</th><th>N</th>'+D.horizons.map(m=>`<th>${m}M win/avg</th>`).join('')+'</tr>';
D.backtest.forEach(r=>{h+=`<tr><td>${r.name}</td><td>${r.n}</td>`+D.horizons.map(m=>r['w'+m]==null?'<td>-</td>':`<td>${r['w'+m]}% / ${r['a'+m]}%</td>`).join('')+'</tr>'});
document.getElementById('bt').innerHTML=h;
</script></body></html>
"""

if __name__ == "__main__":
    main()
