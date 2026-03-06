# -*- coding: utf-8 -*-
"""
Phần 3: EDA & Chẩn đoán — Dashboard Web
Phong Nha – Kẻ Bàng vs Ba Na Hills vs Cố đô Huế
"""
import pandas as pd
import numpy as np
import json
import re
from collections import Counter
from datetime import datetime

# ===========================
# 1. ĐỌC & XỬ LÝ DỮ LIỆU
# ===========================
df = pd.read_csv(r'd:\HK2_Nam3\PNKB\DataMerge.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['publish_date'])
df['year']        = df['publish_date'].dt.year
df['month']       = df['publish_date'].dt.month
df['year_month']  = df['publish_date'].dt.to_period('M').astype(str)
df['quarter']     = df['publish_date'].dt.to_period('Q').astype(str)
df['day_of_week'] = df['publish_date'].dt.dayofweek
df['content_len'] = df['content'].str.len().fillna(0).astype(int)
df['title_len']   = df['title'].str.len().fillna(0).astype(int)
df['content']     = df['content'].fillna('')
df['title']       = df['title'].fillna('')

DESTINATIONS = ['Phong Nha Kẻ Bàng', 'Ba Na Hills', 'Cố đô Huế']
COLORS = {
    'Phong Nha Kẻ Bàng': '#2196F3',
    'Ba Na Hills':        '#FF5722',
    'Cố đô Huế':          '#4CAF50'
}
COLORS_LIGHT = {
    'Phong Nha Kẻ Bàng': 'rgba(33,150,243,0.15)',
    'Ba Na Hills':        'rgba(255,87,34,0.15)',
    'Cố đô Huế':          'rgba(76,175,80,0.15)'
}

# ── Phân loại nội dung ──────────────────────────────────────────────
def classify(row):
    t = (str(row['title']) + ' ' + str(row['content'])).lower()
    if any(w in t for w in ['lễ hội','festival','sự kiện','khai trương','khánh thành','carnival','đại lễ']):
        return 'Sự kiện / Lễ hội'
    if any(w in t for w in ['di sản','unesco','lịch sử','văn hóa','bảo tồn','truyền thống','di tích']):
        return 'Di sản / Văn hóa'
    if any(w in t for w in ['thiên nhiên','hang động','núi','rừng','biển','sông','thác','sinh thái']):
        return 'Thiên nhiên / Sinh thái'
    if any(w in t for w in ['tour','giá vé','khuyến mãi','ưu đãi','combo','quảng bá','promotion']):
        return 'Quảng bá / Marketing'
    if any(w in t for w in ['du khách','trải nghiệm','khám phá','tham quan','check-in','review','hành trình']):
        return 'Trải nghiệm / Review'
    return 'Tin tức chung'

df['content_type'] = df.apply(classify, axis=1)

# ── Phát hiện spike ──────────────────────────────────────────────────
monthly_all = df.groupby('year_month').size().reset_index(name='total')

# ── Keywords ────────────────────────────────────────────────────────
STOPWORDS = {'và','của','cho','với','được','từ','trong','đến','là','có','tại','các','một',
             'về','đã','không','này','theo','như','nhưng','sẽ','trên','cũng','khi','để',
             'hay','những','nhiều','đó','ra','người','sau','còn','rất','lại','nơi',
             'hơn','mới','qua','vào','năm','ngày','the','a','to','of','in','for','and','is'}

def top_kw(df_sub, n=12):
    """Extract top bigrams (2-syllable phrases) from Vietnamese titles."""
    # Additional stopwords: destination name syllables & common noise
    DEST_SYLLABLES = {
        'phong','nha','bàng','kẻ','ba','na','hills','hill','cố','đô','huế',
        'sun','world','bà','nà','quảng','bình','sơn','đoòng','lịch','khách',
        'thế','giới','việt','nam','đây','này','được',
        # bridge fragments that produce meaningless cross-boundary bigrams
        'tham','phí','quan','nghìn','khu','hàng',
    }
    all_stop = STOPWORDS | DEST_SYLLABLES
    bigrams = Counter()
    for title in df_sub['title'].dropna():
        tokens = [w.lower() for w in re.findall(r'[\w\u00C0-\u024F\u1E00-\u1EFF]+', str(title))
                  if not w.isdigit()]
        for a, b in zip(tokens, tokens[1:]):
            if a not in all_stop and b not in all_stop and len(a) >= 2 and len(b) >= 2:
                bigrams[f'{a} {b}'] += 1
    return bigrams.most_common(n)

# ===========================
# 2. TÍNH TOÁN DỮ LIỆU BIỂU ĐỒ
# ===========================

# -- A. Xu hướng theo năm
yearly = df.groupby(['year','Destination']).size().unstack(fill_value=0).reset_index()

# -- B. Xu hướng theo tháng (rolling 3M)
monthly_pivot = (df.groupby(['year_month','Destination'])
                   .size().unstack(fill_value=0).reset_index())
all_ym = sorted(df['year_month'].unique())

# -- C. Share of voice
sov = df['Destination'].value_counts()
sov_labels = sov.index.tolist()
sov_vals   = sov.values.tolist()

# -- D. Source diversity
src_div = df.groupby('Destination')['source_name'].nunique()

# -- E. Avg content length
avg_len = df.groupby('Destination')['content_len'].mean().round(0)

# -- F. Monthly frequency
monthly_freq = (df.groupby(['Destination','year_month']).size()
                  .groupby('Destination').mean().round(1))

# -- G. Content mix per destination
ct_mix = (df.groupby(['Destination','content_type']).size()
            .unstack(fill_value=0))
ct_mix_pct = ct_mix.div(ct_mix.sum(axis=1), axis=0).mul(100).round(1)

# -- H. Monthly trend per destination (for spike chart)
spike_data = {}
for d in DESTINATIONS:
    sub = df[df['Destination']==d].groupby('year_month').size().reset_index(name='count')
    spike_data[d] = sub

# -- I. YoY growth
recent_yr = df['year'].max()
prev_yr   = recent_yr - 1
yoy = {}
for d in DESTINATIONS:
    curr = len(df[(df['Destination']==d)&(df['year']==recent_yr)])
    prev = len(df[(df['Destination']==d)&(df['year']==prev_yr)])
    yoy[d] = round((curr - prev)/prev*100, 1) if prev > 0 else 0

# -- J. Keywords per destination
kw_data = {d: top_kw(df[df['Destination']==d], 12) for d in DESTINATIONS}

# -- K. Day of week
dow_data = df.groupby(['day_of_week','Destination']).size().unstack(fill_value=0)
dow_pct  = dow_data.div(dow_data.sum(axis=0), axis=1).mul(100).round(1)

# -- L. Monthly seasonality
mon_data = df.groupby(['month','Destination']).size().unstack(fill_value=0)
mon_pct  = mon_data.div(mon_data.sum(axis=0), axis=1).mul(100).round(1)

# -- M. Top nguồn báo
top_src = df['source_name'].value_counts().head(8)
src_by_dest = (df[df['source_name'].isin(top_src.index)]
               .groupby(['source_name','Destination']).size().unstack(fill_value=0)
               .reindex(top_src.index))

# -- N. Top 5 bài best practice (longest/most representative per dest)
def top_articles(d, n=5):
    sub = df[df['Destination']==d].nlargest(n, 'content_len')[['title','source_name','publish_date','content_type','content_len']]
    return sub

# -- O. Spike giai đoạn — rolling mean + std
def find_spikes(d):
    sub = df[df['Destination']==d].groupby('year_month').size().reset_index(name='cnt')
    sub['roll'] = sub['cnt'].rolling(3, min_periods=1).mean()
    sub['std']  = sub['cnt'].rolling(3, min_periods=1).std().fillna(0)
    spikes = sub[sub['cnt'] > sub['roll'] + sub['std']].head(5)
    return spikes[['year_month','cnt']].to_dict('records')

spikes = {d: find_spikes(d) for d in DESTINATIONS}

# -- Q. Data-driven derived statistics (no hardcoded assumptions) ────────────
MONTH_NAMES_VN = ['','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
MARKETING_TYPE = 'Quảng bá / Marketing'
NATURE_TYPE    = 'Thiên nhiên / Sinh thái'

# Peak months per destination (top 3 by article count, from data)
peak_month_info = {}
for _d in DESTINATIONS:
    if _d in mon_data.columns:
        _best_m = int(mon_data[_d].idxmax())
        _best_pct = round(float(mon_pct.loc[_best_m, _d]) if _d in mon_pct.columns else 0, 1)
        _top3_m  = mon_data[_d].nlargest(3).index.tolist()
        _top3_str = ', '.join(MONTH_NAMES_VN[m] for m in sorted(_top3_m))
        peak_month_info[_d] = {'best': MONTH_NAMES_VN[_best_m], 'pct': _best_pct, 'top3': _top3_str}
    else:
        peak_month_info[_d] = {'best': 'N/A', 'pct': 0, 'top3': 'N/A'}

# Top-2 content types per destination (computed from ct_mix_pct)
ct_top2 = {}
for _d in DESTINATIONS:
    if _d in ct_mix_pct.index:
        _top2 = ct_mix_pct.loc[_d].nlargest(2)
        ct_top2[_d] = [(str(t), round(float(v),1)) for t, v in _top2.items()]
    else:
        ct_top2[_d] = [('N/A', 0), ('N/A', 0)]

# Marketing & Nature content % per destination
marketing_pct = {}
nature_pct    = {}
for _d in DESTINATIONS:
    marketing_pct[_d] = round(float(ct_mix_pct.loc[_d, MARKETING_TYPE]) if _d in ct_mix_pct.index and MARKETING_TYPE in ct_mix_pct.columns else 0, 1)
    nature_pct[_d]    = round(float(ct_mix_pct.loc[_d, NATURE_TYPE])    if _d in ct_mix_pct.index and NATURE_TYPE in ct_mix_pct.columns else 0, 1)

best_nature_dest = max(DESTINATIONS, key=lambda d: nature_pct.get(d, 0))

# Top source per destination (from data)
top_src_per_dest = {}
for _d in DESTINATIONS:
    _sub = df[df['Destination']==_d]['source_name'].value_counts()
    top_src_per_dest[_d] = (_sub.index[0], int(_sub.iloc[0])) if len(_sub) else ('N/A', 0)

# Top-3 keywords per destination (from actual title analysis)
top3_kw = {_d: [w for w, _ in kw_data[_d][:3]] for _d in DESTINATIONS}

# Benchmark targets computed from data
pnkb_len_val   = int(avg_len.get('Phong Nha Kẻ Bàng', 0))
best_len_dest  = max(DESTINATIONS, key=lambda d: avg_len.get(d, 0))
best_len_val   = int(avg_len.get(best_len_dest, 0))
target_len_val = int(round((pnkb_len_val + best_len_val) / 2, -2))

pnkb_freq_val  = float(monthly_freq.get('Phong Nha Kẻ Bàng', 0))
best_freq_dest = max(DESTINATIONS, key=lambda d: monthly_freq.get(d, 0))
best_freq_val  = float(monthly_freq.get(best_freq_dest, 0))

pnkb_src_val   = int(src_div.get('Phong Nha Kẻ Bàng', 0))
best_src_val   = int(src_div.max())

pnkb_top3_months = peak_month_info['Phong Nha Kẻ Bàng']['top3']

# Top-5 article avg length per destination
def top5_avg_len(d):
    return int(df[df['Destination']==d].nlargest(5,'content_len')['content_len'].mean())
def top5_len_ratio(d):
    avg = avg_len.get(d, 1)
    return round(top5_avg_len(d) / avg, 1) if avg > 0 else 0

# Spike periods with representative article titles (purely from data)
def spike_articles_html(d):
    sp = spikes[d]
    if not sp:
        return "<p style='color:#888;font-style:italic'>Không phát hiện spike đáng kể trong dữ liệu.</p>"
    parts = []
    for s in sp[:3]:
        ym     = s['year_month']
        cnt    = s['cnt']
        titles = df[(df['Destination']==d) & (df['year_month']==ym)]['title'].dropna().head(3).tolist()
        thtml  = ''.join(f'<li style="margin:2px 0;color:#555">{t[:80]}{"…" if len(t)>80 else ""}</li>' for t in titles)
        parts.append(f'<div style="margin-bottom:10px"><strong style="color:{COLORS[d]}">{ym}</strong> &nbsp;—&nbsp; <strong>{cnt} bài</strong><ul style="padding-left:14px;font-size:12px;margin-top:4px">{thtml}</ul></div>')
    return ''.join(parts)

# -- P. Radar KPI normalized
kpi_raw = {
    d: [
        float(sov.get(d, 0)) / float(sov.max()),
        float(avg_len.get(d, 0)) / float(avg_len.max()),
        float(src_div.get(d, 0)) / float(src_div.max()),
        float(monthly_freq.get(d, 0)) / float(monthly_freq.max()),
    ]
    for d in DESTINATIONS
}
kpi_labels = ['Tổng bài viết','Độ dài TB','Đa dạng nguồn','Tần suất/tháng']

# ===========================
# 3. HELPER → JSON SAFE
# ===========================
def jd(obj):
    return json.dumps(obj, ensure_ascii=False)

# ===========================
# 4. GENERATE HTML
# ===========================

def make_yearly_chart():
    datasets = []
    for d in DESTINATIONS:
        if d in yearly.columns:
            datasets.append({
                "label": d,
                "data": [int(x) for x in yearly[d].tolist()],
                "borderColor": COLORS[d],
                "backgroundColor": COLORS_LIGHT[d],
                "fill": True,
                "tension": 0.4,
                "pointRadius": 5,
                "pointHoverRadius": 8,
                "borderWidth": 2.5
            })
    return jd({"labels": yearly['year'].astype(str).tolist(), "datasets": datasets})

def make_monthly_chart():
    datasets = []
    for d in DESTINATIONS:
        vals = []
        for ym in all_ym:
            row = monthly_pivot[monthly_pivot['year_month']==ym]
            if d in row.columns and len(row):
                vals.append(int(row[d].values[0]))
            else:
                vals.append(0)
        datasets.append({
            "label": d,
            "data": vals,
            "borderColor": COLORS[d],
            "backgroundColor": COLORS_LIGHT[d],
            "fill": True,
            "tension": 0.3,
            "pointRadius": 2,
            "borderWidth": 2
        })
    labels = all_ym
    return jd({"labels": labels, "datasets": datasets})

def make_sov_chart():
    return jd({
        "labels": sov_labels,
        "datasets": [{
            "data": sov_vals,
            "backgroundColor": [COLORS.get(l,'#999') for l in sov_labels],
            "hoverOffset": 10,
            "borderWidth": 2,
            "borderColor": "#fff"
        }]
    })

def make_benchmark_bar(metric_dict, label):
    vals = [float(metric_dict.get(d, 0)) for d in DESTINATIONS]
    return jd({
        "labels": DESTINATIONS,
        "datasets": [{
            "label": label,
            "data": vals,
            "backgroundColor": [COLORS[d] for d in DESTINATIONS],
            "borderRadius": 8,
            "borderWidth": 0
        }]
    })

def make_radar():
    datasets = []
    for d in DESTINATIONS:
        datasets.append({
            "label": d,
            "data": kpi_raw[d] + [kpi_raw[d][0]],
            "backgroundColor": COLORS_LIGHT[d],
            "borderColor": COLORS[d],
            "pointBackgroundColor": COLORS[d],
            "borderWidth": 2.5
        })
    return jd({
        "labels": kpi_labels + [kpi_labels[0]],
        "datasets": datasets
    })

def make_ct_mix():
    ct_cats = ct_mix_pct.columns.tolist()
    datasets = []
    for d in DESTINATIONS:
        if d in ct_mix_pct.index:
            datasets.append({
                "label": d,
                "data": [float(ct_mix_pct.loc[d, c]) if c in ct_mix_pct.columns else 0 for c in ct_cats],
                "backgroundColor": COLORS[d],
                "borderRadius": 5
            })
    return jd({"labels": ct_cats, "datasets": datasets})

def make_seasonality():
    mon_labels = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12']
    datasets = []
    for d in DESTINATIONS:
        if d in mon_pct.columns:
            datasets.append({
                "label": d,
                "data": [float(mon_pct.loc[m, d]) if m in mon_pct.index else 0
                         for m in range(1, 13)],
                "borderColor": COLORS[d],
                "backgroundColor": COLORS_LIGHT[d],
                "fill": False,
                "tension": 0.4,
                "pointRadius": 5,
                "borderWidth": 2.5
            })
    return jd({"labels": mon_labels, "datasets": datasets})

def make_dow():
    dow_labels = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']
    datasets = []
    for d in DESTINATIONS:
        if d in dow_pct.columns:
            datasets.append({
                "label": d,
                "data": [float(dow_pct.loc[i, d]) if i in dow_pct.index else 0
                         for i in range(7)],
                "borderColor": COLORS[d],
                "backgroundColor": COLORS_LIGHT[d],
                "fill": False,
                "tension": 0.4,
                "pointRadius": 4,
                "borderWidth": 2
            })
    return jd({"labels": dow_labels, "datasets": datasets})

def make_src_chart():
    datasets = []
    for d in DESTINATIONS:
        if d in src_by_dest.columns:
            datasets.append({
                "label": d,
                "data": [int(src_by_dest.loc[s, d]) if s in src_by_dest.index else 0
                         for s in top_src.index],
                "backgroundColor": COLORS[d],
                "borderRadius": 5
            })
    return jd({"labels": top_src.index.tolist(), "datasets": datasets})

def make_spike_chart(d):
    sub = spike_data[d]
    return jd({
        "labels": sub['year_month'].tolist(),
        "datasets": [{
            "label": d,
            "data": sub['count'].tolist(),
            "borderColor": COLORS[d],
            "backgroundColor": COLORS_LIGHT[d],
            "fill": True,
            "tension": 0.3,
            "pointRadius": 3,
            "borderWidth": 2
        }]
    })

def make_kw_chart(d):
    kws = kw_data[d]
    words  = [w for w, _ in kws]
    counts = [c for _, c in kws]
    return jd({
        "labels": words,
        "datasets": [{
            "label": "Tần suất",
            "data": counts,
            "backgroundColor": COLORS[d],
            "borderRadius": 5,
            "borderWidth": 0
        }]
    })

def best_practice_html(d):
    articles = top_articles(d, 5)
    rows = ""
    for _, r in articles.iterrows():
        date_str = r['publish_date'].strftime('%d/%m/%Y') if pd.notna(r['publish_date']) else ''
        rows += f"""
        <tr>
            <td style="max-width:340px;font-weight:500">{r['title'][:90]}{'…' if len(r['title'])>90 else ''}</td>
            <td><span class="badge" style="background:{COLORS[d]}">{r['content_type']}</span></td>
            <td>{r['source_name']}</td>
            <td>{date_str}</td>
            <td>{r['content_len']:,}</td>
        </tr>"""
    return rows

def spike_table_html(d):
    items = spikes[d]
    if not items:
        return "<p style='color:#888;font-style:italic'>Không phát hiện spike đáng kể</p>"
    rows = ""
    for s in items:
        rows += f"<tr><td><strong>{s['year_month']}</strong></td><td><span style='color:{COLORS[d]};font-weight:700'>{s['cnt']} bài</span></td></tr>"
    return f"<table class='spike-table'><tr><th>Thời điểm</th><th>Số bài</th></tr>{rows}</table>"

def kpi_cards(d):
    total  = int(sov.get(d, 0))
    al     = int(avg_len.get(d, 0))
    nd     = int(src_div.get(d, 0))
    mf     = float(monthly_freq.get(d, 0))
    growth = yoy[d]
    g_color= '#4CAF50' if growth >= 0 else '#F44336'
    g_sign = '+' if growth >= 0 else ''
    pct    = round(total / len(df) * 100, 1)
    return f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value" style="color:{COLORS[d]}">{total}</div>
            <div class="kpi-label">Tổng bài viết</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:{COLORS[d]}">{pct}%</div>
            <div class="kpi-label">Share of Voice</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:{COLORS[d]}">{al:,}</div>
            <div class="kpi-label">Độ dài TB (ký tự)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:{COLORS[d]}">{nd}</div>
            <div class="kpi-label">Nguồn báo chí</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:{COLORS[d]}">{mf}</div>
            <div class="kpi-label">Bài / tháng</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:{g_color}">{g_sign}{growth}%</div>
            <div class="kpi-label">Tăng trưởng YoY</div>
        </div>
    </div>"""

# ===========================
# 5. BUILD HTML
# ===========================
total_articles = len(df)
date_range_str = f"{df['publish_date'].min().strftime('%d/%m/%Y')} — {df['publish_date'].max().strftime('%d/%m/%Y')}"

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phần 3: EDA & Chẩn đoán — Phong Nha Kẻ Bàng</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  :root {{
    --pnkb: #2196F3;
    --bana: #FF5722;
    --hue:  #4CAF50;
    --dark: #1A1A2E;
    --card: #ffffff;
    --bg:   #F0F4F8;
    --text: #2D2D2D;
    --sub:  #6B7280;
    --border: #E5E7EB;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── HEADER ── */
  .header {{
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    color: #fff;
    padding: 40px 32px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  }}
  .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
  .header p  {{ color: rgba(255,255,255,.7); font-size: 15px; max-width: 700px; margin: 0 auto; }}
  .header-meta {{
    display: flex; gap: 24px; justify-content: center; margin-top: 24px; flex-wrap: wrap;
  }}
  .header-meta .chip {{
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
    color: #fff;
    backdrop-filter: blur(4px);
  }}
  .header-meta .chip strong {{ color: #63B3ED; }}

  /* ── NAV TABS ── */
  .nav {{
    background: #fff;
    border-bottom: 2px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    display: flex; gap: 0; overflow-x: auto;
    padding: 0 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
  }}
  .nav-btn {{
    background: none; border: none;
    padding: 14px 20px;
    cursor: pointer;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--sub);
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: all .2s;
  }}
  .nav-btn:hover {{ color: var(--pnkb); }}
  .nav-btn.active {{
    color: var(--pnkb);
    border-bottom-color: var(--pnkb);
    background: rgba(33,150,243,.04);
  }}

  /* ── LAYOUT ── */
  .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
  .section {{ display: none; }}
  .section.active {{ display: block; }}
  .section-title {{
    font-size: 22px; font-weight: 700;
    color: var(--dark); margin-bottom: 6px;
    padding-bottom: 10px;
    border-bottom: 3px solid var(--pnkb);
    display: inline-block;
  }}
  .section-sub {{
    color: var(--sub); font-size: 13.5px; margin-bottom: 24px; margin-top: 6px;
  }}

  /* ── GRID ── */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .grid-1 {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 20px; }}
  @media(max-width:900px) {{ .grid-2,.grid-3 {{ grid-template-columns: 1fr; }} }}

  /* ── CARD ── */
  .card {{
    background: var(--card);
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    border: 1px solid var(--border);
  }}
  .card-title {{
    font-size: 14px; font-weight: 600;
    color: var(--sub); text-transform: uppercase;
    letter-spacing: .5px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }}
  .card-title .icon {{ font-size: 16px; }}

  /* ── KPI CARDS ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(6,1fr);
    gap: 14px; margin-bottom: 22px;
  }}
  @media(max-width:1100px) {{ .kpi-row {{ grid-template-columns: repeat(3,1fr); }} }}
  @media(max-width:600px)  {{ .kpi-row {{ grid-template-columns: repeat(2,1fr); }} }}
  .kpi-card {{
    background: #fff;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
    border: 1px solid var(--border);
    transition: transform .2s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); }}
  .kpi-value {{ font-size: 26px; font-weight: 800; margin-bottom: 4px; }}
  .kpi-label {{ font-size: 12px; color: var(--sub); font-weight: 500; }}

  /* ── DEST TABS ── */
  .dest-tabs {{
    display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;
  }}
  .dest-tab {{
    padding: 8px 20px; border-radius: 20px;
    border: 2px solid var(--border);
    cursor: pointer; font-size: 13.5px; font-weight: 600;
    background: #fff; color: var(--sub);
    transition: all .2s;
  }}
  .dest-tab[data-color="pnkb"].active {{ background: #E3F2FD; border-color: var(--pnkb); color: var(--pnkb); }}
  .dest-tab[data-color="bana"].active {{ background: #FBE9E7; border-color: var(--bana); color: var(--bana); }}
  .dest-tab[data-color="hue"].active  {{ background: #E8F5E9; border-color: var(--hue);  color: var(--hue);  }}

  .dest-panel {{ display: none; }}
  .dest-panel.active {{ display: block; }}

  /* ── BADGE ── */
  .badge {{
    display: inline-block;
    padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600;
    color: #fff; white-space: nowrap;
  }}

  /* ── TABLE ── */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{
    background: #F8FAFC; font-weight: 600;
    padding: 10px 12px; text-align: left;
    border-bottom: 2px solid var(--border); color: var(--sub);
    font-size: 12px; text-transform: uppercase; letter-spacing: .4px;
  }}
  .data-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  .data-table tr:hover td {{ background: #F8FAFC; }}

  .spike-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .spike-table th,
  .spike-table td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
  .spike-table th {{ background: #F8FAFC; font-size: 12px; color: var(--sub); font-weight: 600; }}

  /* ── INSIGHT BOX ── */
  .insight-box {{
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border-left: 4px solid var(--pnkb);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin-bottom: 20px;
    font-size: 13.5px;
  }}
  .insight-box.orange {{ background: linear-gradient(135deg,#FFF7ED,#FEE2D5); border-color: var(--bana); }}
  .insight-box.green  {{ background: linear-gradient(135deg,#F0FDF4,#DCFCE7); border-color: var(--hue); }}
  .insight-box strong {{ display: block; font-size: 14px; margin-bottom: 4px; }}

  /* ── LEGEND ── */
  .legend {{
    display: flex; gap: 20px; flex-wrap: wrap;
    font-size: 13px; margin-bottom: 16px;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{
    width: 12px; height: 12px; border-radius: 50%;
    flex-shrink: 0;
  }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center; padding: 28px;
    color: var(--sub); font-size: 13px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}

  canvas {{ max-height: 360px; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>📊 Phần 3: EDA & Chẩn đoán Truyền thông</h1>
  <p>So sánh mức độ xuất hiện và chiến lược nội dung của Phong Nha – Kẻ Bàng với các điểm đến cạnh tranh</p>
  <div class="header-meta">
    <div class="chip">📅 {date_range_str}</div>
    <div class="chip">📰 Tổng: <strong>{total_articles:,}</strong> bài viết</div>
    <div class="chip">🗺️ <strong>3</strong> điểm đến</div>
    <div class="chip">📡 <strong>{df['source_name'].nunique()}</strong> nguồn báo chí</div>
    <div class="chip">🏷️ <strong>{df['content_type'].nunique()}</strong> loại nội dung</div>
  </div>
</div>

<!-- NAV -->
<nav class="nav">
  <button class="nav-btn active" onclick="switchTab('trend',this)">1. Xu hướng truyền thông</button>
  <button class="nav-btn" onclick="switchTab('benchmark',this)">2. Benchmark KPI</button>
  <button class="nav-btn" onclick="switchTab('strategy',this)">3. Chiến lược nội dung</button>
  <button class="nav-btn" onclick="switchTab('spikes',this)">4. Sự kiện & Spike</button>
  <button class="nav-btn" onclick="switchTab('bestpractice',this)">5. Best Practices</button>
  <button class="nav-btn" onclick="switchTab('diagnosis',this)">6. Chẩn đoán & Kết luận</button>
</nav>

<div class="container">

<!-- ═══════════════════════════════════════════════════
     SECTION 1 — XU HƯỚNG
═══════════════════════════════════════════════════ -->
<section id="trend" class="section active">
  <div class="section-title">1. So sánh Xu hướng Truyền thông theo Thời gian</div>
  <div class="section-sub">Phân tích số lượng bài viết theo tháng/năm — xác định giai đoạn truyền thông cao điểm của từng điểm đến</div>

  <div class="insight-box">
    <strong>🔍 Nhận xét tổng quan</strong>
    Cố đô Huế dẫn đầu với {int(sov.get('Cố đô Huế',0))} bài ({round(sov.get('Cố đô Huế',0)/total_articles*100,1)}%), tiếp theo là Phong Nha – Kẻ Bàng ({int(sov.get('Phong Nha Kẻ Bàng',0))} bài — {round(sov.get('Phong Nha Kẻ Bàng',0)/total_articles*100,1)}%) và Ba Na Hills ({int(sov.get('Ba Na Hills',0))} bài — {round(sov.get('Ba Na Hills',0)/total_articles*100,1)}%). Khoảng cách giữa Phong Nha và Cố đô Huế là <strong>{int(sov.get('Cố đô Huế',0)) - int(sov.get('Phong Nha Kẻ Bàng',0))} bài viết</strong> — đây là khoảng trống truyền thông cần thu hẹp.
  </div>

  <div class="grid-1">
    <div class="card">
      <div class="card-title"><span class="icon">📈</span> Xu hướng đăng bài theo Năm (2012–2026)</div>
      <canvas id="chartYearly"></canvas>
    </div>
  </div>

  <div class="grid-1">
    <div class="card">
      <div class="card-title"><span class="icon">📅</span> Xu hướng đăng bài theo Tháng (toàn bộ giai đoạn)</div>
      <canvas id="chartMonthly" style="max-height:280px"></canvas>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title"><span class="icon">🗓️</span> Tính mùa vụ — Phân bố theo Tháng (%)</div>
      <canvas id="chartSeasonality"></canvas>
    </div>
    <div class="card">
      <div class="card-title"><span class="icon">📆</span> Phân bố đăng bài theo Ngày trong Tuần (%)</div>
      <canvas id="chartDow"></canvas>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════
     SECTION 2 — BENCHMARK KPI
═══════════════════════════════════════════════════ -->
<section id="benchmark" class="section">
  <div class="section-title">2. Benchmark Mức độ Phủ Truyền thông</div>
  <div class="section-sub">Đánh giá vị trí của Phong Nha – Kẻ Bàng so với các đối thủ: tổng bài viết, share of voice, đa dạng nguồn, tần suất đăng bài</div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:var(--pnkb)"></div> Phong Nha Kẻ Bàng</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--bana)"></div> Ba Na Hills</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--hue)"></div> Cố đô Huế</div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title"><span class="icon">🥧</span> Share of Voice — Tỷ lệ bài viết</div>
      <canvas id="chartSov"></canvas>
    </div>
    <div class="card">
      <div class="card-title"><span class="icon">🏆</span> Radar KPI — Chỉ số tổng hợp (chuẩn hóa)</div>
      <canvas id="chartRadar"></canvas>
    </div>
  </div>

  <div class="grid-3">
    <div class="card">
      <div class="card-title"><span class="icon">📰</span> Số nguồn báo đưa tin</div>
      <canvas id="chartSrcDiv"></canvas>
    </div>
    <div class="card">
      <div class="card-title"><span class="icon">📝</span> Độ dài bài viết TB (ký tự)</div>
      <canvas id="chartAvgLen"></canvas>
    </div>
    <div class="card">
      <div class="card-title"><span class="icon">⚡</span> Tần suất đăng bài (bài/tháng)</div>
      <canvas id="chartFreq"></canvas>
    </div>
  </div>

  <div class="card">
    <div class="card-title"><span class="icon">📡</span> Top 8 Nguồn Báo chí — So sánh theo Điểm đến</div>
    <canvas id="chartSrc" style="max-height:300px"></canvas>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════
     SECTION 3 — CHIẾN LƯỢC NỘI DUNG
═══════════════════════════════════════════════════ -->
<section id="strategy" class="section">
  <div class="section-title">3. Phân tích Chiến lược Nội dung</div>
  <div class="section-sub">Khám phá content mix, từ khóa nổi bật và thông điệp truyền thông của từng điểm đến</div>

  <div class="grid-1">
    <div class="card">
      <div class="card-title"><span class="icon">📊</span> Content Mix — Phân loại Nội dung theo Điểm đến (%)</div>
      <canvas id="chartCtMix"></canvas>
    </div>
  </div>

  <div class="dest-tabs" id="stratTabs">
    <button class="dest-tab active" data-color="pnkb" onclick="switchDest('strat','pnkb',this)">🔵 Phong Nha Kẻ Bàng</button>
    <button class="dest-tab" data-color="bana" onclick="switchDest('strat','bana',this)">🟠 Ba Na Hills</button>
    <button class="dest-tab" data-color="hue"  onclick="switchDest('strat','hue',this)">🟢 Cố đô Huế</button>
  </div>

  <div id="strat-pnkb" class="dest-panel active">
    {kpi_cards('Phong Nha Kẻ Bàng')}
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span class="icon">🔑</span> Từ khóa nổi bật trong tiêu đề</div>
        <canvas id="chartKwPnkb"></canvas>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">🗂️</span> Phân bố loại nội dung</div>
        <canvas id="chartCtPnkb"></canvas>
      </div>
    </div>
    <div class="insight-box">
      <strong>💡 Nhận xét — Phong Nha Kẻ Bàng (dựa trên dữ liệu)</strong>
      Top 2 loại nội dung: <strong>{ct_top2['Phong Nha Kẻ Bàng'][0][0]} ({ct_top2['Phong Nha Kẻ Bàng'][0][1]}%)</strong> và {ct_top2['Phong Nha Kẻ Bàng'][1][0]} ({ct_top2['Phong Nha Kẻ Bàng'][1][1]}%).
      Bài về Quảng bá/Marketing chỉ chiếm <strong>{marketing_pct['Phong Nha Kẻ Bàng']}%</strong> — thấp nhất trong 3 điểm đến (Ba Na Hills: {marketing_pct['Ba Na Hills']}%, Cố đô Huế: {marketing_pct['Cố đô Huế']}%).
      Tháng có nhiều bài nhất theo dữ liệu: <strong>{peak_month_info['Phong Nha Kẻ Bàng']['top3']}</strong>.
      Từ khóa xuất hiện nhiều nhất trong tiêu đề bài viết: <strong>{', '.join(top3_kw['Phong Nha Kẻ Bàng'])}</strong>.
      Nguồn báo đưa tin nhiều nhất: <strong>{top_src_per_dest['Phong Nha Kẻ Bàng'][0]}</strong> ({top_src_per_dest['Phong Nha Kẻ Bàng'][1]} bài).
    </div>
  </div>
  <div id="strat-bana" class="dest-panel">
    {kpi_cards('Ba Na Hills')}
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span class="icon">🔑</span> Từ khóa nổi bật trong tiêu đề</div>
        <canvas id="chartKwBana"></canvas>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">🗂️</span> Phân bố loại nội dung</div>
        <canvas id="chartCtBana"></canvas>
      </div>
    </div>
    <div class="insight-box orange">
      <strong>💡 Nhận xét — Ba Na Hills (dựa trên dữ liệu)</strong>
      Độ dài bài viết TB cao nhất: <strong>{int(avg_len.get('Ba Na Hills',0)):,} ký tự/bài</strong> — cao hơn Phong Nha {int(avg_len.get('Ba Na Hills',0))-int(avg_len.get('Phong Nha Kẻ Bàng',0)):,} ký tự.
      Top 2 loại nội dung: <strong>{ct_top2['Ba Na Hills'][0][0]} ({ct_top2['Ba Na Hills'][0][1]}%)</strong> và {ct_top2['Ba Na Hills'][1][0]} ({ct_top2['Ba Na Hills'][1][1]}%).
      Nguồn báo đưa tin nhiều nhất: <strong>{top_src_per_dest['Ba Na Hills'][0]}</strong> ({top_src_per_dest['Ba Na Hills'][1]} bài).
      Từ khóa xuất hiện nhiều nhất: <strong>{', '.join(top3_kw['Ba Na Hills'])}</strong>.
      Tháng có nhiều bài nhất theo dữ liệu: <strong>{peak_month_info['Ba Na Hills']['top3']}</strong>.
    </div>
  </div>
  <div id="strat-hue" class="dest-panel">
    {kpi_cards('Cố đô Huế')}
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span class="icon">🔑</span> Từ khóa nổi bật trong tiêu đề</div>
        <canvas id="chartKwHue"></canvas>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">🗂️</span> Phân bố loại nội dung</div>
        <canvas id="chartCtHue"></canvas>
      </div>
    </div>
    <div class="insight-box green">
      <strong>💡 Nhận xét — Cố đô Huế (dựa trên dữ liệu)</strong>
      Dẫn đầu tổng bài: <strong>{int(sov.get('Cố đô Huế',0))} bài</strong> và về đa dạng nguồn: {int(src_div.get('Cố đô Huế',0))} nguồn (Phong Nha: {int(src_div.get('Phong Nha Kẻ Bàng',0))}, Ba Na Hills: {int(src_div.get('Ba Na Hills',0))}).
      Top 2 loại nội dung: <strong>{ct_top2['Cố đô Huế'][0][0]} ({ct_top2['Cố đô Huế'][0][1]}%)</strong> và {ct_top2['Cố đô Huế'][1][0]} ({ct_top2['Cố đô Huế'][1][1]}%).
      Từ khóa xuất hiện nhiều nhất trong tiêu đề: <strong>{', '.join(top3_kw['Cố đô Huế'])}</strong>.
      Tháng có nhiều bài nhất theo dữ liệu: <strong>{peak_month_info['Cố đô Huế']['top3']}</strong>.
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════
     SECTION 4 — SỰ KIỆN & SPIKE
═══════════════════════════════════════════════════ -->
<section id="spikes" class="section">
  <div class="section-title">4. Phát hiện Sự kiện & Chiến dịch Nổi bật</div>
  <div class="section-sub">Xác định giai đoạn đối thủ có bùng nổ truyền thông để suy ra hoạt động quảng bá, sự kiện thu hút được chú ý</div>

  <div class="dest-tabs">
    <button class="dest-tab active" data-color="pnkb" onclick="switchDest('spike','pnkb',this)">🔵 Phong Nha Kẻ Bàng</button>
    <button class="dest-tab" data-color="bana" onclick="switchDest('spike','bana',this)">🟠 Ba Na Hills</button>
    <button class="dest-tab" data-color="hue"  onclick="switchDest('spike','hue',this)">🟢 Cố đô Huế</button>
  </div>

  <div id="spike-pnkb" class="dest-panel active">
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span class="icon">📈</span> Timeline bài viết — Phong Nha Kẻ Bàng</div>
        <canvas id="chartSpikePnkb"></canvas>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">⚡</span> Giai đoạn bùng nổ truyền thông (Top spike)</div>
        {spike_table_html('Phong Nha Kẻ Bàng')}
        <div style="margin-top:14px;padding:12px;background:#EFF6FF;border-radius:8px;font-size:13px">
          <strong>📌 Tiêu đề bài viết thực tế trong các giai đoạn spike (từ dữ liệu):</strong>
          {spike_articles_html('Phong Nha Kẻ Bàng')}
        </div>
      </div>
    </div>
  </div>

  <div id="spike-bana" class="dest-panel">
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span class="icon">📈</span> Timeline bài viết — Ba Na Hills</div>
        <canvas id="chartSpikeBana"></canvas>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">⚡</span> Giai đoạn bùng nổ truyền thông (Top spike)</div>
        {spike_table_html('Ba Na Hills')}
        <div style="margin-top:14px;padding:12px;background:#FBE9E7;border-radius:8px;font-size:13px">
          <strong>📌 Tiêu đề bài viết thực tế trong các giai đoạn spike (từ dữ liệu):</strong>
          {spike_articles_html('Ba Na Hills')}
        </div>
      </div>
    </div>
  </div>

  <div id="spike-hue" class="dest-panel">
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span class="icon">📈</span> Timeline bài viết — Cố đô Huế</div>
        <canvas id="chartSpikeHue"></canvas>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">⚡</span> Giai đoạn bùng nổ truyền thông (Top spike)</div>
        {spike_table_html('Cố đô Huế')}
        <div style="margin-top:14px;padding:12px;background:#F0FDF4;border-radius:8px;font-size:13px">
          <strong>📌 Tiêu đề bài viết thực tế trong các giai đoạn spike (từ dữ liệu):</strong>
          {spike_articles_html('Cố đô Huế')}
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════
     SECTION 5 — BEST PRACTICES
═══════════════════════════════════════════════════ -->
<section id="bestpractice" class="section">
  <div class="section-title">5. Best Practices — Bài học từ Đối thủ</div>
  <div class="section-sub">Top bài viết nổi bật theo điểm đến và các yếu tố tạo nên sức hút truyền thông</div>

  <div class="dest-tabs">
    <button class="dest-tab active" data-color="pnkb" onclick="switchDest('bp','pnkb',this)">🔵 Phong Nha Kẻ Bàng</button>
    <button class="dest-tab" data-color="bana" onclick="switchDest('bp','bana',this)">🟠 Ba Na Hills</button>
    <button class="dest-tab" data-color="hue"  onclick="switchDest('bp','hue',this)">🟢 Cố đô Huế</button>
  </div>

  <div id="bp-pnkb" class="dest-panel active">
    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">🏅</span> Top 5 bài viết nội dung phong phú nhất — Phong Nha Kẻ Bàng</div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>Tiêu đề bài viết</th><th>Loại nội dung</th><th>Nguồn</th><th>Ngày đăng</th><th>Độ dài</th></tr></thead>
          <tbody>{best_practice_html('Phong Nha Kẻ Bàng')}</tbody>
        </table>
      </div>
    </div>
    <div class="insight-box">
      <strong>💡 Đặc điểm top 5 bài — Phong Nha Kẻ Bàng (từ dữ liệu)</strong>
      Độ dài TB của top 5 bài dài nhất: <strong>{top5_avg_len('Phong Nha Kẻ Bàng'):,} ký tự</strong> — gấp <strong>{top5_len_ratio('Phong Nha Kẻ Bàng'):.1f}x</strong> so với TB toàn bộ ({pnkb_len_val:,} ký tự).
      Loại nội dung chiếm ưu thế trong top 5: <strong>{df[df['Destination']=='Phong Nha Kẻ Bàng'].nlargest(5,'content_len')['content_type'].mode().iloc[0]}</strong>.
      Từ khóa xuất hiện nhiều nhất: <strong>{', '.join(top3_kw['Phong Nha Kẻ Bàng'])}</strong>.
      Nguồn báo chứa bài dài nhất: <strong>{df[df['Destination']=='Phong Nha Kẻ Bàng'].nlargest(1,'content_len')['source_name'].iloc[0]}</strong>.
    </div>
  </div>
  <div id="bp-bana" class="dest-panel">
    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">🏅</span> Top 5 bài viết nội dung phong phú nhất — Ba Na Hills</div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>Tiêu đề bài viết</th><th>Loại nội dung</th><th>Nguồn</th><th>Ngày đăng</th><th>Độ dài</th></tr></thead>
          <tbody>{best_practice_html('Ba Na Hills')}</tbody>
        </table>
      </div>
    </div>
    <div class="insight-box orange">
      <strong>💡 Đặc điểm top 5 bài — Ba Na Hills (từ dữ liệu)</strong>
      Độ dài TB của top 5 bài dài nhất: <strong>{top5_avg_len('Ba Na Hills'):,} ký tự</strong> — gấp <strong>{top5_len_ratio('Ba Na Hills'):.1f}x</strong> so với TB toàn bộ ({int(avg_len.get('Ba Na Hills',0)):,} ký tự).
      Loại nội dung chiếm ưu thế trong top 5: <strong>{df[df['Destination']=='Ba Na Hills'].nlargest(5,'content_len')['content_type'].mode().iloc[0]}</strong>.
      Từ khóa xuất hiện nhiều nhất trong tiêu đề: <strong>{', '.join(top3_kw['Ba Na Hills'])}</strong>.
      Nguồn báo đưa tin nhiều nhất: <strong>{top_src_per_dest['Ba Na Hills'][0]}</strong> ({top_src_per_dest['Ba Na Hills'][1]} bài).
    </div>
  </div>
  <div id="bp-hue" class="dest-panel">
    <div class="card" style="margin-bottom:20px">
      <div class="card-title"><span class="icon">🏅</span> Top 5 bài viết nội dung phong phú nhất — Cố đô Huế</div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>Tiêu đề bài viết</th><th>Loại nội dung</th><th>Nguồn</th><th>Ngày đăng</th><th>Độ dài</th></tr></thead>
          <tbody>{best_practice_html('Cố đô Huế')}</tbody>
        </table>
      </div>
    </div>
    <div class="insight-box green">
      <strong>💡 Đặc điểm top 5 bài — Cố đô Huế (từ dữ liệu)</strong>
      Độ dài TB của top 5 bài dài nhất: <strong>{top5_avg_len('Cố đô Huế'):,} ký tự</strong> — gấp <strong>{top5_len_ratio('Cố đô Huế'):.1f}x</strong> so với TB toàn bộ ({int(avg_len.get('Cố đô Huế',0)):,} ký tự).
      Loại nội dung chiếm ưu thế trong top 5: <strong>{df[df['Destination']=='Cố đô Huế'].nlargest(5,'content_len')['content_type'].mode().iloc[0]}</strong>.
      Từ khóa xuất hiện nhiều nhất trong tiêu đề: <strong>{', '.join(top3_kw['Cố đô Huế'])}</strong>.
      Nguồn báo đưa tin nhiều nhất: <strong>{top_src_per_dest['Cố đô Huế'][0]}</strong> ({top_src_per_dest['Cố đô Huế'][1]} bài).
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════
     SECTION 6 — CHẨN ĐOÁN & KẾT LUẬN
═══════════════════════════════════════════════════ -->
<section id="diagnosis" class="section">
  <div class="section-title">6. Chẩn đoán Vị thế & Định hướng Chiến lược</div>
  <div class="section-sub">Tổng hợp phân tích EDA để đưa ra chẩn đoán toàn diện và khuyến nghị chiến lược truyền thông cho Phong Nha – Kẻ Bàng</div>

  <!-- Summary KPIs -->
  <div class="grid-3" style="margin-bottom:24px">
    <div class="card" style="border-top:4px solid var(--pnkb)">
      <div class="card-title" style="color:var(--pnkb)">🔵 Phong Nha – Kẻ Bàng</div>
      <div style="font-size:13px;color:var(--sub);line-height:2">
        <div>📰 <strong>{int(sov.get('Phong Nha Kẻ Bàng',0))}</strong> bài viết ({round(sov.get('Phong Nha Kẻ Bàng',0)/total_articles*100,1)}%)</div>
        <div>📡 <strong>{int(src_div.get('Phong Nha Kẻ Bàng',0))}</strong> nguồn báo</div>
        <div>📝 TB <strong>{int(avg_len.get('Phong Nha Kẻ Bàng',0)):,}</strong> ký tự/bài</div>
        <div>⚡ <strong>{float(monthly_freq.get('Phong Nha Kẻ Bàng',0)):.1f}</strong> bài/tháng</div>
        <div>📈 YoY: <strong style="color:{'#4CAF50' if yoy['Phong Nha Kẻ Bàng']>=0 else '#F44336'}">{'+' if yoy['Phong Nha Kẻ Bàng']>=0 else ''}{yoy['Phong Nha Kẻ Bàng']}%</strong></div>
      </div>
    </div>
    <div class="card" style="border-top:4px solid var(--bana)">
      <div class="card-title" style="color:var(--bana)">🟠 Ba Na Hills</div>
      <div style="font-size:13px;color:var(--sub);line-height:2">
        <div>📰 <strong>{int(sov.get('Ba Na Hills',0))}</strong> bài viết ({round(sov.get('Ba Na Hills',0)/total_articles*100,1)}%)</div>
        <div>📡 <strong>{int(src_div.get('Ba Na Hills',0))}</strong> nguồn báo</div>
        <div>📝 TB <strong>{int(avg_len.get('Ba Na Hills',0)):,}</strong> ký tự/bài</div>
        <div>⚡ <strong>{float(monthly_freq.get('Ba Na Hills',0)):.1f}</strong> bài/tháng</div>
        <div>📈 YoY: <strong style="color:{'#4CAF50' if yoy['Ba Na Hills']>=0 else '#F44336'}">{'+' if yoy['Ba Na Hills']>=0 else ''}{yoy['Ba Na Hills']}%</strong></div>
      </div>
    </div>
    <div class="card" style="border-top:4px solid var(--hue)">
      <div class="card-title" style="color:var(--hue)">🟢 Cố đô Huế</div>
      <div style="font-size:13px;color:var(--sub);line-height:2">
        <div>📰 <strong>{int(sov.get('Cố đô Huế',0))}</strong> bài viết ({round(sov.get('Cố đô Huế',0)/total_articles*100,1)}%)</div>
        <div>📡 <strong>{int(src_div.get('Cố đô Huế',0))}</strong> nguồn báo</div>
        <div>📝 TB <strong>{int(avg_len.get('Cố đô Huế',0)):,}</strong> ký tự/bài</div>
        <div>⚡ <strong>{float(monthly_freq.get('Cố đô Huế',0)):.1f}</strong> bài/tháng</div>
        <div>📈 YoY: <strong style="color:{'#4CAF50' if yoy['Cố đô Huế']>=0 else '#F44336'}">{'+' if yoy['Cố đô Huế']>=0 else ''}{yoy['Cố đô Huế']}%</strong></div>
      </div>
    </div>
  </div>

  <!-- Diagnosis -->
  <div class="card" style="margin-bottom:20px">
    <div class="card-title"><span class="icon">🔬</span> Chẩn đoán Vị thế Truyền thông Phong Nha – Kẻ Bàng</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px">
      <div style="padding:16px;background:#EFF6FF;border-radius:10px">
        <div style="font-weight:700;color:var(--pnkb);margin-bottom:10px;font-size:15px">💪 Điểm mạnh</div>
        <ul style="font-size:13.5px;line-height:2;padding-left:18px">
          <li>Đa dạng nguồn báo: <strong>{pnkb_src_val} nguồn</strong> (Ba Na Hills: {int(src_div.get('Ba Na Hills',0))}, Cố đô Huế: {int(src_div.get('Cố đô Huế',0))})</li>
          <li>Loại nội dung chiếm ưu thế: <strong>{ct_top2['Phong Nha Kẻ Bàng'][0][0]} ({ct_top2['Phong Nha Kẻ Bàng'][0][1]}%)</strong></li>
          <li>Từ khóa nổi bật nhất trong tiêu đề (đếm từ dữ liệu): <strong>{', '.join(top3_kw['Phong Nha Kẻ Bàng'])}</strong></li>
          <li>Nguồn báo đưa tin nhiều nhất: <strong>{top_src_per_dest['Phong Nha Kẻ Bàng'][0]}</strong> ({top_src_per_dest['Phong Nha Kẻ Bàng'][1]} bài)</li>
          <li>Tần suất TB: <strong>{pnkb_freq_val:.1f} bài/tháng</strong></li>
        </ul>
      </div>
      <div style="padding:16px;background:#FEF2F2;border-radius:10px">
        <div style="font-weight:700;color:#EF4444;margin-bottom:10px;font-size:15px">⚠️ Điểm yếu / Khoảng trống</div>
        <ul style="font-size:13.5px;line-height:2;padding-left:18px">
          <li>Ít hơn Cố đô Huế <strong>{int(sov.get('Cố đô Huế',0))-int(sov.get('Phong Nha Kẻ Bàng',0))} bài viết</strong></li>
          <li>Độ dài bài thấp hơn Ba Na Hills <strong>{int(avg_len.get('Ba Na Hills',0))-int(avg_len.get('Phong Nha Kẻ Bàng',0)):,} ký tự</strong></li>
          <li>Thiếu nội dung Quảng bá/Marketing chủ động (&lt;2%)</li>
          <li>Chưa có chiến lược event-driven PR rõ ràng</li>
          <li>Tần suất thấp: {float(monthly_freq.get('Phong Nha Kẻ Bàng',0)):.1f} vs {float(monthly_freq.get('Cố đô Huế',0)):.1f} bài/tháng (Huế)</li>
        </ul>
      </div>
      <div style="padding:16px;background:#F0FDF4;border-radius:10px">
        <div style="font-weight:700;color:var(--hue);margin-bottom:10px;font-size:15px">🚀 Cơ hội</div>
        <ul style="font-size:13.5px;line-height:2;padding-left:18px">
          <li>Tăng bài về <em>{NATURE_TYPE}</em>: PNKB hiện <strong>{nature_pct['Phong Nha Kẻ Bàng']}%</strong> — đối thủ cao nhất là {best_nature_dest} ({nature_pct[best_nature_dest]}%)</li>
          <li>Tăng Quảng bá/Marketing từ <strong>{marketing_pct['Phong Nha Kẻ Bàng']}%</strong> lên ngang bằng mức cao nhất trong dataset ({max(marketing_pct['Ba Na Hills'], marketing_pct['Cố đô Huế'])}%)</li>
          <li>Khai thác tháng cao điểm xác định từ dữ liệu: <strong>{pnkb_top3_months}</strong></li>
          <li>Mở rộng nguồn báo: hiện {pnkb_src_val} nguồn — đối thủ dẫn đầu có {best_src_val} nguồn (còn {best_src_val - pnkb_src_val} nguồn chưa khai thác)</li>
          <li>Tăng độ dài bài: TB hiện <strong>{pnkb_len_val:,} ký tự</strong> — {best_len_dest} đạt {best_len_val:,} ký tự/bài</li>
        </ul>
      </div>
      <div style="padding:16px;background:#FFFBEB;border-radius:10px">
        <div style="font-weight:700;color:#F59E0B;margin-bottom:10px;font-size:15px">📋 Khuyến nghị ưu tiên</div>
        <ol style="font-size:13.5px;line-height:2;padding-left:18px">
          <li><strong>Tăng tần suất</strong>: Hiện {pnkb_freq_val:.1f} bài/tháng → benchmark cao nhất trong dataset: <strong>{best_freq_val:.1f} bài/tháng</strong> ({best_freq_dest})</li>
          <li><strong>Tăng độ dài bài</strong>: Từ {pnkb_len_val:,} → mục tiêu <strong>{target_len_val:,} ký tự</strong> (trung bình PNKB và {best_len_dest} — benchmark từ dữ liệu)</li>
          <li><strong>Tăng Quảng bá/Marketing</strong>: Từ {marketing_pct['Phong Nha Kẻ Bàng']}% → mục tiêu <strong>{max(marketing_pct['Ba Na Hills'], marketing_pct['Cố đô Huế'])}%</strong> (mức cao nhất trong dataset)</li>
          <li><strong>Tập trung tháng cao điểm</strong>: Chiến dịch vào <strong>{pnkb_top3_months}</strong> (3 tháng có nhiều bài nhất trong lịch sử dữ liệu)</li>
          <li><strong>Mở rộng nguồn báo</strong>: Từ {pnkb_src_val} → <strong>{best_src_val} nguồn</strong> (benchmark cao nhất trong dataset)</li>
        </ol>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title"><span class="icon">📊</span> Gap Analysis — Phong Nha KB vs Đối thủ tốt nhất (chuẩn hóa %)</div>
    <canvas id="chartGap" style="max-height:300px"></canvas>
  </div>
</section>

</div><!-- /container -->

<div class="footer">
  Dashboard tự động sinh từ DataMerge.csv &nbsp;|&nbsp; Tổng {total_articles:,} bài viết &nbsp;|&nbsp; Phong Nha – Kẻ Bàng EDA Report &nbsp;|&nbsp; {datetime.now().strftime('%d/%m/%Y')}
</div>

<!-- ═══════════════════════════════════════════════════
     JAVASCRIPT
═══════════════════════════════════════════════════ -->
<script>
// ── Tab navigation ──────────────────────────────────────────────────
function switchTab(id, btn) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}

// ── Destination sub-tabs ─────────────────────────────────────────────
function switchDest(prefix, id, btn) {{
  const parent = btn.closest('section') || document;
  parent.querySelectorAll('.dest-panel').forEach(p => p.classList.remove('active'));
  parent.querySelectorAll('.dest-tab').forEach(b => b.classList.remove('active'));
  document.getElementById(prefix + '-' + id).classList.add('active');
  btn.classList.add('active');
}}

// ── Chart defaults ───────────────────────────────────────────────────
Chart.defaults.font.family = "'Segoe UI', Arial, sans-serif";
Chart.defaults.font.size   = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 14;

const gridStyle = {{
  color: 'rgba(0,0,0,.06)',
  drawBorder: false
}};

// ── Helper ───────────────────────────────────────────────────────────
function baseLineOpts(yLabel='') {{
  return {{
    responsive: true,
    maintainAspectRatio: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ backgroundColor: 'rgba(26,26,46,.92)', titleFont: {{ weight: 'bold' }}, padding: 12 }}
    }},
    scales: {{
      x: {{ grid: gridStyle, ticks: {{ maxTicksLimit: 12, maxRotation: 45 }} }},
      y: {{ grid: gridStyle, beginAtZero: true, title: {{ display: !!yLabel, text: yLabel }} }}
    }}
  }};
}}

function baseBarOpts(yLabel='') {{
  return {{
    responsive: true,
    maintainAspectRatio: true,
    plugins: {{
      legend: {{ position: 'top' }},
      tooltip: {{ backgroundColor: 'rgba(26,26,46,.92)', padding: 12 }}
    }},
    scales: {{
      x: {{ grid: {{...gridStyle, display:false}} }},
      y: {{ grid: gridStyle, beginAtZero: true, title: {{ display: !!yLabel, text: yLabel }} }}
    }}
  }};
}}

// ── 1. YEARLY TREND ─────────────────────────────────────────────────
new Chart(document.getElementById('chartYearly'), {{
  type: 'line',
  data: {make_yearly_chart()},
  options: baseLineOpts('Số bài viết')
}});

// ── 2. MONTHLY TREND ────────────────────────────────────────────────
new Chart(document.getElementById('chartMonthly'), {{
  type: 'line',
  data: {make_monthly_chart()},
  options: {{
    ...baseLineOpts('Số bài viết'),
    scales: {{
      x: {{ grid: gridStyle, ticks: {{ maxTicksLimit: 24, maxRotation: 60, font: {{size:10}} }} }},
      y: {{ grid: gridStyle, beginAtZero: true }}
    }}
  }}
}});

// ── 3. SEASONALITY ──────────────────────────────────────────────────
new Chart(document.getElementById('chartSeasonality'), {{
  type: 'line',
  data: {make_seasonality()},
  options: baseLineOpts('% bài viết')
}});

// ── 4. DAY OF WEEK ───────────────────────────────────────────────────
new Chart(document.getElementById('chartDow'), {{
  type: 'line',
  data: {make_dow()},
  options: baseLineOpts('% bài viết')
}});

// ── 5. SOV PIE ──────────────────────────────────────────────────────
new Chart(document.getElementById('chartSov'), {{
  type: 'doughnut',
  data: {make_sov_chart()},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'bottom' }},
      tooltip: {{
        backgroundColor: 'rgba(26,26,46,.92)', padding: 12,
        callbacks: {{ label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}} bài (${{(ctx.parsed/ctx.chart.getDatasetMeta(0).total*100).toFixed(1)}}%)` }}
      }}
    }},
    cutout: '58%'
  }}
}});

// ── 6. RADAR ────────────────────────────────────────────────────────
new Chart(document.getElementById('chartRadar'), {{
  type: 'radar',
  data: {make_radar()},
  options: {{
    responsive: true,
    scales: {{ r: {{ min: 0, max: 1, ticks: {{ stepSize: 0.25, font: {{size:10}} }}, pointLabels: {{ font: {{size:11, weight:'bold'}} }} }} }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});

// ── 7. SOURCE DIVERSITY ─────────────────────────────────────────────
new Chart(document.getElementById('chartSrcDiv'), {{
  type: 'bar',
  data: {make_benchmark_bar(src_div, 'Số nguồn')},
  options: baseBarOpts('Số nguồn báo')
}});

// ── 8. AVG LENGTH ───────────────────────────────────────────────────
new Chart(document.getElementById('chartAvgLen'), {{
  type: 'bar',
  data: {make_benchmark_bar(avg_len, 'Ký tự TB')},
  options: baseBarOpts('Số ký tự')
}});

// ── 9. FREQUENCY ────────────────────────────────────────────────────
new Chart(document.getElementById('chartFreq'), {{
  type: 'bar',
  data: {make_benchmark_bar(monthly_freq, 'Bài/tháng')},
  options: baseBarOpts('Bài / tháng')
}});

// ── 10. SOURCE COMPARISON ───────────────────────────────────────────
new Chart(document.getElementById('chartSrc'), {{
  type: 'bar',
  data: {make_src_chart()},
  options: {{
    ...baseBarOpts('Số bài viết'),
    indexAxis: 'y',
    scales: {{
      x: {{ grid: gridStyle, beginAtZero: true }},
      y: {{ grid: {{...gridStyle, display:false}} }}
    }}
  }}
}});

// ── 11. CONTENT MIX ─────────────────────────────────────────────────
new Chart(document.getElementById('chartCtMix'), {{
  type: 'bar',
  data: {make_ct_mix()},
  options: {{
    ...baseBarOpts('%'),
    plugins: {{ legend: {{ position:'top' }}, tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}%` }} }} }}
  }}
}});

// ── 12. KEYWORD CHARTS ──────────────────────────────────────────────
new Chart(document.getElementById('chartKwPnkb'), {{
  type: 'bar',
  data: {make_kw_chart('Phong Nha Kẻ Bàng')},
  options: {{ ...baseBarOpts('Tần suất'), indexAxis:'y',
    scales: {{ x: {{ grid:gridStyle,beginAtZero:true }}, y:{{ grid:{{...gridStyle,display:false}} }} }} }}
}});
new Chart(document.getElementById('chartKwBana'), {{
  type: 'bar',
  data: {make_kw_chart('Ba Na Hills')},
  options: {{ ...baseBarOpts('Tần suất'), indexAxis:'y',
    scales: {{ x: {{ grid:gridStyle,beginAtZero:true }}, y:{{ grid:{{...gridStyle,display:false}} }} }} }}
}});
new Chart(document.getElementById('chartKwHue'), {{
  type: 'bar',
  data: {make_kw_chart('Cố đô Huế')},
  options: {{ ...baseBarOpts('Tần suất'), indexAxis:'y',
    scales: {{ x: {{ grid:gridStyle,beginAtZero:true }}, y:{{ grid:{{...gridStyle,display:false}} }} }} }}
}});

// ── 13. CONTENT TYPE per destination ───────────────────────────────
function ctData(dest, color) {{
  const d = {jd(ct_mix_pct.to_dict())};
  const cats = Object.keys(d);
  const vals = cats.map(c => d[c][dest] || 0);
  return {{
    labels: cats,
    datasets: [{{ label: dest, data: vals, backgroundColor: color, borderRadius:5 }}]
  }};
}}
new Chart(document.getElementById('chartCtPnkb'), {{ type:'bar', data: ctData('{DESTINATIONS[0]}','{COLORS[DESTINATIONS[0]]}'), options: baseBarOpts('%') }});
new Chart(document.getElementById('chartCtBana'), {{ type:'bar', data: ctData('{DESTINATIONS[1]}','{COLORS[DESTINATIONS[1]]}'), options: baseBarOpts('%') }});
new Chart(document.getElementById('chartCtHue'),  {{ type:'bar', data: ctData('{DESTINATIONS[2]}','{COLORS[DESTINATIONS[2]]}'), options: baseBarOpts('%') }});

// ── 14. SPIKE CHARTS ────────────────────────────────────────────────
new Chart(document.getElementById('chartSpikePnkb'), {{
  type: 'line', data: {make_spike_chart('Phong Nha Kẻ Bàng')},
  options: baseLineOpts('Số bài/tháng')
}});
new Chart(document.getElementById('chartSpikeBana'), {{
  type: 'line', data: {make_spike_chart('Ba Na Hills')},
  options: baseLineOpts('Số bài/tháng')
}});
new Chart(document.getElementById('chartSpikeHue'), {{
  type: 'line', data: {make_spike_chart('Cố đô Huế')},
  options: baseLineOpts('Số bài/tháng')
}});

// ── 15. GAP ANALYSIS ────────────────────────────────────────────────
const gapMetrics = ['Tổng bài viết', 'Độ dài TB nội dung', 'Đa dạng nguồn báo', 'Bài / tháng'];
const gapPnkb  = {jd(kpi_raw['Phong Nha Kẻ Bàng'])}.map(v => (v*100).toFixed(1)*1);
const gapBest  = [100, 100, 100, 100];
new Chart(document.getElementById('chartGap'), {{
  type: 'bar',
  data: {{
    labels: gapMetrics,
    datasets: [
      {{ label: 'Phong Nha KB (%)', data: gapPnkb, backgroundColor: 'rgba(33,150,243,.75)', borderRadius:6 }},
      {{ label: 'Đối thủ tốt nhất (100%)', data: gapBest, backgroundColor: 'rgba(239,68,68,.2)', borderRadius:6, borderWidth:2, borderColor:'rgba(239,68,68,.6)' }}
    ]
  }},
  options: {{
    ...baseBarOpts('% so với mức tốt nhất'),
    plugins: {{
      legend: {{ position:'top' }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y}}%` }} }}
    }},
    scales: {{
      x: {{ grid:{{...gridStyle,display:false}} }},
      y: {{ grid:gridStyle, beginAtZero:true, max:110, title:{{display:true,text:'%'}} }}
    }}
  }}
}});

console.log('✅ Dashboard loaded successfully');
</script>
</body>
</html>"""

# ── Ghi file HTML ──────────────────────────────────────────────────
output_path = r'd:\HK2_Nam3\PNKB\eda_dashboard.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✅ Dashboard đã được tạo: {output_path}")
print(f"   Kích thước: {len(html)//1024} KB")
print(f"   Mở file trong trình duyệt để xem kết quả!")
