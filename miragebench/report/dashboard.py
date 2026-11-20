"""Etkileşimli, tek-dosya HTML dashboard üreteci.

results/ içindeki kol×tohum JSON'larını toplar ve kendi kendine yeten (harici
bağımlılık yok, çevrimdışı ve file:// ile açılır) bir HTML dashboard yazar:
veri-verimliliği eğrisi (metrik değiştirilebilir), domain gap, ablasyon, boyut
katkısı, tam sonuç tablosu ve hipotez kararları. Grafikler vanilla JS + SVG ile
çizilir (CDN yok), veri HTML'e gömülür.

Kullanım:
    python -m miragebench.report.dashboard --results results/study --out docs/report/dashboard.html

mb-report'tan bağımsız çalışır; aynı toplama fonksiyonlarını kullanır, böylece
deneyi yeniden koşup bunu tekrar çalıştırınca dashboard güncel veriyle yenilenir.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .aggregate import (
    aggregate_by_arm,
    data_efficiency_curve,
    hypothesis_decisions,
    load_results,
)


def _real_n(arm: str) -> int | None:
    m = re.search(r"_(\d+)$", arm)
    if arm == "real_only_all":
        return None
    return int(m.group(1)) if m else None


def build_data(results_dir: str | Path) -> dict:
    """Dashboard için gereken tüm veriyi tek sözlükte toplar."""
    records = load_results(results_dir)
    agg = aggregate_by_arm(records)

    def m(arm, key):
        e = agg.get(arm)
        return e["metrics"][key]["mean"] if e else None

    # Sonuç tablosu satırları
    rows = []
    for arm in sorted(agg):
        e = agg[arm]
        rows.append({
            "arm": arm,
            "n_real": e.get("n_real_train"),
            "n_synth": e.get("n_synth_train"),
            "map50": m(arm, "map50"),
            "map50_95": m(arm, "map50_95"),
            "small": m(arm, "map50_small"),
            "medium": m(arm, "map50_medium"),
            "large": m(arm, "map50_large"),
            "seeds": e.get("n_seeds"),
        })

    # Veri-verimliliği eğrisi: her N için real_only ve synth_pretrain metrikleri
    def curve(prefix):
        pts = []
        for arm in agg:
            if prefix == "real_only" and arm.startswith("real_only_"):
                n = _real_n(arm)
                if arm == "real_only_all":
                    n = None
            elif prefix == "synth_pretrain" and arm.startswith("synth_pretrain_real_"):
                n = _real_n(arm)
            else:
                continue
            if n is None:
                continue
            pts.append({"n": n, "map50": m(arm, "map50"),
                        "map50_95": m(arm, "map50_95"), "small": m(arm, "map50_small")})
        pts.sort(key=lambda p: p["n"])
        return pts

    real_curve = curve("real_only")
    synth_curve = curve("synth_pretrain")

    # Ablasyon: referans synth_pretrain_real_250
    ref = m("synth_pretrain_real_250", "map50")
    ablation = []
    for arm in sorted(agg):
        if arm.startswith("ablation_"):
            v = m(arm, "map50")
            ablation.append({"factor": arm[len("ablation_"):], "map50": v,
                             "drop": round((ref - v), 4) if ref is not None and v is not None else None})

    # Boyut katkısı: synth_pretrain vs real_only, N=50 ve N=250
    def size_gain(n):
        sp = f"synth_pretrain_real_{n}"
        ro = f"real_only_{n}"
        out = {}
        for sz in ("small", "medium", "large"):
            a = m(sp, f"map50_{sz}")
            b = m(ro, f"map50_{sz}")
            out[sz] = round(a - b, 4) if a is not None and b is not None else None
        return out

    # Öne çıkan metrikler
    best_real = max((r["map50"] for r in rows if r["n_synth"] == 0 and r["map50"] is not None), default=None)
    synth_only = m("synth_only", "map50")
    key = {
        "domain_gap": round(best_real - synth_only, 4) if best_real and synth_only else None,
        "synth_only": synth_only,
        "best_real": best_real,
        "gain_at_50": (round(m("synth_pretrain_real_50", "map50") - m("real_only_50", "map50"), 4)
                       if m("synth_pretrain_real_50", "map50") and m("real_only_50", "map50") is not None else None),
        "small_gain_at_50": size_gain(50).get("small"),
        "n_arms": len(agg),
        "n_runs": len(records),
    }

    return {
        "rows": rows,
        "real_curve": real_curve,
        "synth_curve": synth_curve,
        "ablation": ablation,
        "size_gain_50": size_gain(50),
        "size_gain_250": size_gain(250),
        "hypotheses": hypothesis_decisions(agg),
        "key": key,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return _TEMPLATE.replace("/*__DATA__*/", payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MirageBench etkileşimli dashboard üreteci")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="docs/report/dashboard.html")
    args = ap.parse_args(argv)
    data = build_data(args.results)
    if not data["rows"]:
        print("sonuç bulunamadı:", args.results)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    print(f"dashboard -> {out}  ({data['key']['n_runs']} koşu, {data['key']['n_arms']} kol)")
    return 0


# ---------------------------------------------------------------------------
# Tek-dosya HTML/JS/CSS şablonu. Veri /*__DATA__*/ yerine gömülür.
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MirageBench — Sonuç Panosu</title>
<style>
  :root{
    --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --border:#2d3748;
    --fg:#e6edf3; --muted:#9aa7b4; --accent:#58a6ff; --accent2:#f78166;
    --good:#3fb950; --bad:#f85149; --grid:#21262d;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1100px;margin:0 auto;padding:28px 20px 60px}
  h1{font-size:26px;margin:0 0 4px}
  .sub{color:var(--muted);margin:0 0 24px;font-size:14px}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:28px}
  .tile{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}
  .tile .v{font-size:28px;font-weight:700}
  .tile .l{color:var(--muted);font-size:12.5px;margin-top:4px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:22px}
  .panel h2{font-size:16px;margin:0 0 4px}
  .panel .note{color:var(--muted);font-size:13px;margin:0 0 14px}
  .toggle{display:inline-flex;gap:6px;margin-bottom:10px}
  .toggle button{background:var(--panel2);color:var(--muted);border:1px solid var(--border);
    border-radius:8px;padding:5px 11px;cursor:pointer;font-size:13px}
  .toggle button.on{background:var(--accent);color:#06101f;border-color:var(--accent);font-weight:600}
  svg{width:100%;height:auto;display:block}
  .legend{display:flex;gap:18px;margin-top:8px;font-size:13px;color:var(--muted);flex-wrap:wrap}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:600;cursor:pointer;user-select:none}
  tbody tr:hover{background:var(--panel2)}
  .bar{height:9px;border-radius:5px;background:var(--accent);display:inline-block;vertical-align:middle}
  .hyp{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
  .hcard{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px}
  .hcard .t{font-weight:700;margin-bottom:4px}
  .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600}
  .ok{background:rgba(63,185,80,.15);color:var(--good)}
  .no{background:rgba(248,81,73,.15);color:var(--bad)}
  .hcard .d{color:var(--muted);font-size:12.5px;margin-top:6px}
  .tt{position:fixed;pointer-events:none;background:#000d;border:1px solid var(--border);
    border-radius:8px;padding:7px 10px;font-size:12.5px;opacity:0;transition:opacity .08s;z-index:9}
  footer{color:var(--muted);font-size:12.5px;margin-top:30px;border-top:1px solid var(--border);padding-top:16px}
  a{color:var(--accent)}
</style>
</head>
<body>
<div class="wrap">
  <h1>MirageBench — Sonuç Panosu</h1>
  <p class="sub">Küçük drone tespitinde sentetik verinin gerçek veri açığını ne kadar kapattığını ölçen benchmark · DUT Anti-UAV test seti · YOLOv11s</p>

  <div class="tiles" id="tiles"></div>

  <div class="panel">
    <h2>Veri verimliliği</h2>
    <p class="note">Gerçek görüntü sayısı (N) arttıkça başarı. Sentetik ön-eğitimin katkısı az-veri rejiminde en yüksek.</p>
    <div class="toggle" id="metricToggle"></div>
    <div id="lineChart"></div>
    <div class="legend">
      <span><span class="dot" style="background:#58a6ff"></span>Gerçek-only</span>
      <span><span class="dot" style="background:#f78166"></span>Sentetik ön-eğitim + gerçek</span>
    </div>
  </div>

  <div class="panel">
    <h2>Boyut katmanına göre sentetik katkısı (N=50)</h2>
    <p class="note">Sentetik ön-eğitimin, sentetiksiz-50'ye göre puan kazancı. En kıt rejimde küçük nesnede kazanç çok büyük.</p>
    <div id="sizeChart"></div>
  </div>

  <div class="panel">
    <h2>Ablasyon — domain randomization faktör önemi</h2>
    <p class="note">Her faktör kapatıldığında referansa (synth_pretrain_real_250) göre mAP@50 düşüşü. N=250'de sentetik zaten doyduğu için farklar küçük (bkz. FINDINGS §4).</p>
    <div id="ablChart"></div>
  </div>

  <div class="panel">
    <h2>Tüm kollar</h2>
    <p class="note">Başlığa tıklayarak sırala. Bar = mAP@50.</p>
    <table id="tbl"><thead></thead><tbody></tbody></table>
  </div>

  <div class="panel">
    <h2>Hipotez kararları</h2>
    <div class="hyp" id="hyp"></div>
  </div>

  <footer>
    Otomatik üretildi · <span id="meta"></span> · Kaynak: <a href="https://github.com/mrpmoffical/sim2real-drone-detection">sim2real-drone-detection</a>
    · Tek tohum sonuçları; varyans için 3 tohumla tekrar önerilir (FINDINGS §7).
  </footer>
</div>

<div class="tt" id="tt"></div>
<script>
const DATA = /*__DATA__*/;
const $ = s => document.querySelector(s);
const tt = $('#tt');
function showTip(e,html){tt.innerHTML=html;tt.style.opacity=1;tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY+14)+'px';}
function hideTip(){tt.style.opacity=0;}
const fmt = v => (v==null?'–':(+v).toFixed(3));
const pts = v => (v>=0?'+':'')+(v*100).toFixed(1);

/* ---- tiles ---- */
(function(){
  const k=DATA.key;
  const tiles=[
    ['Domain gap', k.domain_gap!=null?pts(k.domain_gap)+' p':'–','sadece-sentetik vs en iyi gerçek (mAP@50)'],
    ['N=50 kazancı', k.gain_at_50!=null?pts(k.gain_at_50)+' p':'–','sentetik ön-eğitim vs gerçek-only'],
    ['N=50 küçük nesne', k.small_gain_at_50!=null?pts(k.small_gain_at_50)+' p':'–','küçük-nesne mAP@50 kazancı'],
    ['En iyi gerçek', fmt(k.best_real),'real_only (tüm veri) mAP@50'],
    ['Koşu / kol', k.n_runs+' / '+k.n_arms,'tamamlanan deney'],
  ];
  $('#tiles').innerHTML=tiles.map(t=>`<div class="tile"><div class="v">${t[1]}</div><div class="l"><b>${t[0]}</b><br>${t[2]}</div></div>`).join('');
})();

/* ---- line chart (data efficiency) ---- */
let metric='map50';
const W=900,H=380,P={l:56,r:20,t:18,b:46};
function xlog(n,min,max){return P.l+(Math.log10(n)-Math.log10(min))/(Math.log10(max)-Math.log10(min))*(W-P.l-P.r);}
function ylin(v){return H-P.b-(v-0)/(1-0)*(H-P.t-P.b);}
function drawLine(){
  const all=DATA.real_curve.concat(DATA.synth_curve).map(p=>p.n);
  const nmin=Math.min(...all), nmax=Math.max(...all);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
  // grid + y labels
  for(let i=0;i<=5;i++){const v=i/5;const y=ylin(v);
    s+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="var(--grid)"/>`;
    s+=`<text x="${P.l-8}" y="${y+4}" fill="var(--muted)" font-size="12" text-anchor="end">${v.toFixed(1)}</text>`;}
  // x ticks at actual N
  [...new Set(all)].sort((a,b)=>a-b).forEach(n=>{const x=xlog(n,nmin,nmax);
    s+=`<text x="${x}" y="${H-P.b+18}" fill="var(--muted)" font-size="12" text-anchor="middle">${n}</text>`;});
  s+=`<text x="${(P.l+W-P.r)/2}" y="${H-6}" fill="var(--muted)" font-size="12.5" text-anchor="middle">Gerçek eğitim görüntüsü (log ölçek)</text>`;
  s+=`<text transform="translate(16 ${(P.t+H-P.b)/2}) rotate(-90)" fill="var(--muted)" font-size="12.5" text-anchor="middle">${metric}</text>`;
  function series(arr,color){
    if(!arr.length)return'';
    let d='',dots='';
    arr.forEach((p,i)=>{const x=xlog(p.n,nmin,nmax),y=ylin(p[metric]);
      d+=(i?'L':'M')+x+' '+y+' ';
      dots+=`<circle cx="${x}" cy="${y}" r="4.5" fill="${color}" stroke="#0d1117" stroke-width="1.5"
        onmousemove="showTip(event,'N=${p.n}<br>${metric}: <b>${fmt(p[metric])}</b>')" onmouseleave="hideTip()"/>`;});
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5"/>`+dots;
  }
  s+=series(DATA.real_curve,'#58a6ff');
  s+=series(DATA.synth_curve,'#f78166');
  s+=`</svg>`;
  $('#lineChart').innerHTML=s;
}
(function(){
  const metrics=[['map50','mAP@50'],['map50_95','mAP@50-95'],['small','küçük nesne mAP@50']];
  $('#metricToggle').innerHTML=metrics.map(m=>`<button data-m="${m[0]}" class="${m[0]==metric?'on':''}">${m[1]}</button>`).join('');
  $('#metricToggle').querySelectorAll('button').forEach(b=>b.onclick=()=>{
    metric=b.dataset.m;
    $('#metricToggle').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.m==metric));
    drawLine();});
  drawLine();
})();

/* ---- horizontal bar helper ---- */
function hbars(el,items,opt){
  const w=900,bh=30,gap=10,pl=180,pr=70,h=items.length*(bh+gap)+10;
  const vals=items.map(i=>i.v);
  const lo=Math.min(0,...vals), hi=Math.max(0,...vals);
  const span=(hi-lo)||1;
  const zx=pl+(0-lo)/span*(w-pl-pr);
  let s=`<svg viewBox="0 0 ${w} ${h}">`;
  s+=`<line x1="${zx}" y1="0" x2="${zx}" y2="${h-10}" stroke="var(--border)"/>`;
  items.forEach((it,i)=>{const y=i*(bh+gap)+6;
    const x=pl+(Math.min(0,it.v)-lo)/span*(w-pl-pr);
    const bw=Math.abs(it.v)/span*(w-pl-pr);
    const col=it.color||(it.v>=0?'var(--accent)':'var(--bad)');
    s+=`<text x="${pl-10}" y="${y+bh/2+4}" fill="var(--fg)" font-size="13" text-anchor="end">${it.label}</text>`;
    s+=`<rect x="${x}" y="${y}" width="${Math.max(1,bw)}" height="${bh}" rx="5" fill="${col}"
        onmousemove="showTip(event,'${it.label}<br><b>${it.disp}</b>')" onmouseleave="hideTip()"/>`;
    s+=`<text x="${x+ (it.v>=0?bw+8:-8)}" y="${y+bh/2+4}" fill="var(--muted)" font-size="12.5"
        text-anchor="${it.v>=0?'start':'end'}">${it.disp}</text>`;});
  s+=`</svg>`;el.innerHTML=s;
}

/* size gain N=50 */
(function(){
  const g=DATA.size_gain_50||{};
  const order=['small','medium','large'], names={small:'küçük',medium:'orta',large:'büyük'};
  const cols={small:'#3fb950',medium:'#58a6ff',large:'#8957e5'};
  const items=order.filter(k=>g[k]!=null).map(k=>({label:names[k],v:g[k],disp:pts(g[k])+' p',color:cols[k]}));
  if(items.length)hbars($('#sizeChart'),items); else $('#sizeChart').innerHTML='<p class="note">veri yok</p>';
})();

/* ablation */
(function(){
  const a=(DATA.ablation||[]).filter(x=>x.drop!=null).sort((x,y)=>y.drop-x.drop);
  const items=a.map(x=>({label:x.factor,v:x.drop,disp:pts(x.drop)+' p',color:'#f78166'}));
  if(items.length)hbars($('#ablChart'),items); else $('#ablChart').innerHTML='<p class="note">ablasyon kolu yok</p>';
})();

/* table */
(function(){
  const cols=[['arm','Kol'],['n_real','Gerçek'],['n_synth','Sentetik'],['map50','mAP@50'],
    ['map50_95','mAP@50-95'],['small','küçük'],['medium','orta'],['large','büyük']];
  $('#tbl thead').innerHTML='<tr>'+cols.map(c=>`<th data-k="${c[0]}">${c[1]}</th>`).join('')+'</tr>';
  let rows=DATA.rows.slice(), sortK='map50', dir=-1;
  const maxMap=Math.max(...rows.map(r=>r.map50||0));
  function render(){
    rows.sort((a,b)=>{const x=a[sortK],y=b[sortK];
      if(typeof x==='string')return dir*x.localeCompare(y);
      return dir*(((x==null?-1:x))-((y==null?-1:y)));});
    $('#tbl tbody').innerHTML=rows.map(r=>{
      const barw=(r.map50||0)/maxMap*70;
      return `<tr>
        <td>${r.arm}</td><td>${r.n_real==null?'tümü':r.n_real}</td><td>${r.n_synth||0}</td>
        <td><span class="bar" style="width:${barw}px"></span> ${fmt(r.map50)}</td>
        <td>${fmt(r.map50_95)}</td><td>${fmt(r.small)}</td><td>${fmt(r.medium)}</td><td>${fmt(r.large)}</td></tr>`;}).join('');
  }
  $('#tbl thead').querySelectorAll('th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(k===sortK)dir*=-1; else {sortK=k;dir=-1;} render();});
  render();
})();

/* hypotheses */
(function(){
  $('#hyp').innerHTML=(DATA.hypotheses||[]).map(h=>{
    const ok=h.verdict==='desteklendi';
    return `<div class="hcard"><div class="t">${h.hypothesis}
      <span class="badge ${ok?'ok':'no'}">${h.verdict}</span></div>
      <div class="d">${h.detail||''}</div></div>`;}).join('');
})();

$('#meta').textContent = DATA.key.n_runs+' koşu · '+DATA.key.n_arms+' kol';
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
