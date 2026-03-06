# -*- coding: utf-8 -*-
"""
Phần 3: EDA & Chẩn đoán — Phong Nha Kẻ Bàng vs Đối thủ
So sánh xu hướng, Benchmark KPI, Phân rã chiến lược, Best practices
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ===== Cấu hình font tiếng Việt =====
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.unicode_minus'] = False

# Màu sắc cho 3 điểm đến
COLORS = {
    'Phong Nha Kẻ Bàng': '#2196F3',  # Xanh dương
    'Ba Na Hills': '#FF5722',          # Cam
    'Cố đô Huế': '#4CAF50'            # Xanh lá
}

# ===== 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU =====
df = pd.read_csv(r'd:\HK2_Nam3\PNKB\DataMerge.csv')
df['publish_date'] = pd.to_datetime(df['publish_date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['publish_date'])
df['year'] = df['publish_date'].dt.year
df['month'] = df['publish_date'].dt.month
df['year_month'] = df['publish_date'].dt.to_period('M')
df['quarter'] = df['publish_date'].dt.to_period('Q')
df['content_length'] = df['content'].str.len().fillna(0).astype(int)
df['title_length'] = df['title'].str.len().fillna(0).astype(int)
df['day_of_week'] = df['publish_date'].dt.dayofweek  # 0=Mon
df['hour'] = df['publish_date'].dt.hour

# ===== Filter chỉ giữ các năm có đủ dữ liệu =====
year_counts = df.groupby('year').size()
valid_years = year_counts[year_counts >= 5].index
df = df[df['year'].isin(valid_years)]

print(f"Tổng số bài viết: {len(df)}")
print(f"Khoảng thời gian: {df['publish_date'].min().strftime('%d/%m/%Y')} -> {df['publish_date'].max().strftime('%d/%m/%Y')}")
print(f"\nSố bài theo điểm đến:")
print(df['Destination'].value_counts())

# =====================================================
# FIGURE 1: SO SÁNH XU HƯỚNG — Posting Frequency
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('PHẦN 3: EDA & CHẨN ĐOÁN\nPhong Nha Kẻ Bàng vs Đối thủ (Ba Na Hills, Cố đô Huế)',
             fontsize=16, fontweight='bold', y=1.02)

# --- 1a. Xu hướng đăng bài theo năm ---
ax1 = axes[0, 0]
yearly = df.groupby(['year', 'Destination']).size().unstack(fill_value=0)
for dest in COLORS:
    if dest in yearly.columns:
        ax1.plot(yearly.index, yearly[dest], marker='o', linewidth=2.5,
                 color=COLORS[dest], label=dest, markersize=6)
ax1.set_title('1a. Tần suất đăng bài theo năm\n(Posting Frequency Trend)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Năm')
ax1.set_ylabel('Số bài viết')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(yearly.index)

# --- 1b. Xu hướng đăng bài theo quý ---
ax2 = axes[0, 1]
quarterly = df.groupby(['quarter', 'Destination']).size().unstack(fill_value=0)
# Chỉ plot các quý gần nhất (2 năm gần nhất)
recent_quarters = quarterly.tail(12)
x_labels = [str(q) for q in recent_quarters.index]
x_pos = range(len(x_labels))
for dest in COLORS:
    if dest in recent_quarters.columns:
        ax2.plot(x_pos, recent_quarters[dest], marker='s', linewidth=2,
                 color=COLORS[dest], label=dest, markersize=5)
ax2.set_title('1b. Tần suất đăng bài theo quý (gần đây)\n(Quarterly Posting Trend)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Quý')
ax2.set_ylabel('Số bài viết')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, rotation=45, fontsize=8)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- 1c. Phân phối nguồn đăng bài (top 7) ---
ax3 = axes[1, 0]
source_dest = df.groupby(['source_name', 'Destination']).size().unstack(fill_value=0)
top_sources = df['source_name'].value_counts().head(7).index
source_dest_top = source_dest.loc[top_sources]
bar_width = 0.25
x = np.arange(len(top_sources))
for i, dest in enumerate(COLORS):
    if dest in source_dest_top.columns:
        ax3.bar(x + i * bar_width, source_dest_top[dest], bar_width,
                color=COLORS[dest], label=dest, edgecolor='white')
ax3.set_title('1c. Phân phối nguồn báo chí (Top 7)\n(Media Source Distribution)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Nguồn báo chí')
ax3.set_ylabel('Số bài viết')
ax3.set_xticks(x + bar_width)
ax3.set_xticklabels(top_sources, rotation=30, fontsize=8, ha='right')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# --- 1d. Tỷ lệ phân phối bài viết (Pie chart) ---
ax4 = axes[1, 1]
dest_counts = df['Destination'].value_counts()
explode = [0.05 if d != 'Phong Nha Kẻ Bàng' else 0.12 for d in dest_counts.index]
colors_pie = [COLORS[d] for d in dest_counts.index]
wedges, texts, autotexts = ax4.pie(dest_counts.values, labels=dest_counts.index,
                                     autopct='%1.1f%%', colors=colors_pie,
                                     explode=explode, startangle=90,
                                     textprops={'fontsize': 10})
for autotext in autotexts:
    autotext.set_fontweight('bold')
ax4.set_title('1d. Tỷ lệ bài viết theo điểm đến\n(Article Share)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(r'd:\HK2_Nam3\PNKB\fig1_trend_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
print("✔ Đã lưu: fig1_trend_comparison.png")

# =====================================================
# FIGURE 2: BENCHMARK KPI
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('BENCHMARK KPI — So sánh chỉ số giữa các điểm đến',
             fontsize=16, fontweight='bold', y=1.02)

# --- 2a. Độ dài bài viết trung bình (proxy cho mức độ đầu tư nội dung) ---
ax1 = axes[0, 0]
content_stats = df.groupby('Destination')['content_length'].agg(['mean', 'median', 'std'])
destinations = list(COLORS.keys())
x = np.arange(len(destinations))
means = [content_stats.loc[d, 'mean'] if d in content_stats.index else 0 for d in destinations]
medians = [content_stats.loc[d, 'median'] if d in content_stats.index else 0 for d in destinations]
bar_width = 0.35
bars1 = ax1.bar(x - bar_width/2, means, bar_width, color=[COLORS[d] for d in destinations],
                label='Trung bình', edgecolor='white', alpha=0.85)
bars2 = ax1.bar(x + bar_width/2, medians, bar_width, color=[COLORS[d] for d in destinations],
                label='Trung vị', edgecolor='white', alpha=0.5, hatch='//')
for bar, val in zip(bars1, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
ax1.set_title('2a. Độ dài bài viết (ký tự)\n(Content Length — Content Investment)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(destinations, fontsize=10)
ax1.set_ylabel('Số ký tự')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# --- 2b. Số nguồn báo chí đưa tin (proxy cho Reach) ---
ax2 = axes[0, 1]
source_diversity = df.groupby('Destination')['source_name'].nunique()
bars = ax2.bar(destinations, [source_diversity.get(d, 0) for d in destinations],
               color=[COLORS[d] for d in destinations], edgecolor='white', width=0.5)
for bar, val in zip(bars, [source_diversity.get(d, 0) for d in destinations]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val}', ha='center', fontsize=12, fontweight='bold')
ax2.set_title('2b. Số nguồn báo chí đưa tin\n(Media Reach — Source Diversity)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Số nguồn')
ax2.grid(True, alpha=0.3, axis='y')

# --- 2c. Tần suất đăng bài TB / tháng (proxy cho Posting Frequency) ---
ax3 = axes[1, 0]
monthly_avg = df.groupby(['Destination', 'year_month']).size().groupby('Destination').mean()
bars = ax3.bar(destinations, [monthly_avg.get(d, 0) for d in destinations],
               color=[COLORS[d] for d in destinations], edgecolor='white', width=0.5)
for bar, val in zip(bars, [monthly_avg.get(d, 0) for d in destinations]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
ax3.set_title('2c. Tần suất bài viết TB/tháng\n(Average Posting Frequency/Month)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Bài/tháng')
ax3.grid(True, alpha=0.3, axis='y')

# --- 2d. Radar chart: Tổng hợp KPI ---
ax4 = axes[1, 1]
ax4.remove()
ax4 = fig.add_subplot(2, 2, 4, projection='polar')

categories = ['Số bài viết', 'Độ dài TB', 'Đa dạng nguồn', 'Bài/tháng', 'Độ dài title TB']
# Normalize to 0-1
total_articles = df['Destination'].value_counts()
avg_content = df.groupby('Destination')['content_length'].mean()
avg_title = df.groupby('Destination')['title_length'].mean()

kpi_data = {}
for d in destinations:
    kpi_data[d] = [
        total_articles.get(d, 0),
        avg_content.get(d, 0),
        source_diversity.get(d, 0),
        monthly_avg.get(d, 0),
        avg_title.get(d, 0)
    ]

# Normalize
max_vals = [max(kpi_data[d][i] for d in destinations) for i in range(len(categories))]
for d in destinations:
    kpi_data[d] = [kpi_data[d][i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(len(categories))]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for d in destinations:
    values = kpi_data[d] + kpi_data[d][:1]
    ax4.plot(angles, values, 'o-', linewidth=2, color=COLORS[d], label=d, markersize=5)
    ax4.fill(angles, values, alpha=0.1, color=COLORS[d])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=8)
ax4.set_title('2d. Radar KPI tổng hợp\n(Normalized)', fontsize=12, fontweight='bold', y=1.1)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout()
plt.savefig(r'd:\HK2_Nam3\PNKB\fig2_benchmark_kpi.png', bbox_inches='tight', dpi=150)
plt.show()
print("✔ Đã lưu: fig2_benchmark_kpi.png")

# =====================================================
# FIGURE 3: PHÂN RÃ CHIẾN LƯỢC
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('PHÂN RÃ CHIẾN LƯỢC — Content Mix, Posting Time, Format',
             fontsize=16, fontweight='bold', y=1.02)

# --- 3a. Posting time — Ngày trong tuần ---
ax1 = axes[0, 0]
day_names = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
dow_data = df.groupby(['day_of_week', 'Destination']).size().unstack(fill_value=0)
dow_pct = dow_data.div(dow_data.sum(axis=0), axis=1) * 100
for dest in destinations:
    if dest in dow_pct.columns:
        ax1.plot(range(7), dow_pct[dest], marker='o', linewidth=2.5,
                 color=COLORS[dest], label=dest, markersize=6)
ax1.set_title('3a. Phân bố ngày đăng bài trong tuần\n(Posting Day Distribution %)', fontsize=12, fontweight='bold')
ax1.set_xticks(range(7))
ax1.set_xticklabels(day_names)
ax1.set_ylabel('% bài viết')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- 3b. Posting time — Tháng trong năm ---
ax2 = axes[0, 1]
month_data = df.groupby(['month', 'Destination']).size().unstack(fill_value=0)
month_pct = month_data.div(month_data.sum(axis=0), axis=1) * 100
month_names = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
for dest in destinations:
    if dest in month_pct.columns:
        ax2.plot(range(1, 13), month_pct[dest], marker='s', linewidth=2.5,
                 color=COLORS[dest], label=dest, markersize=5)
ax2.set_title('3b. Phân bố bài viết theo tháng\n(Monthly Posting Pattern %)', fontsize=12, fontweight='bold')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names)
ax2.set_ylabel('% bài viết')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- 3c. Content Mix — Phân loại nội dung dựa trên keywords ---
ax3 = axes[1, 0]

def classify_content(row):
    text = str(row.get('title', '')) + ' ' + str(row.get('content', ''))
    text = text.lower()
    if any(w in text for w in ['lễ hội', 'festival', 'sự kiện', 'khai trương', 'khánh thành', 'lễ kỷ niệm']):
        return 'Sự kiện/Lễ hội'
    elif any(w in text for w in ['du khách', 'trải nghiệm', 'khám phá', 'tham quan', 'check-in', 'review']):
        return 'Trải nghiệm/Review'
    elif any(w in text for w in ['di sản', 'lịch sử', 'văn hóa', 'bảo tồn', 'truyền thống', 'di tích']):
        return 'Di sản/Văn hóa'
    elif any(w in text for w in ['tour', 'giá vé', 'khuyến mãi', 'ưu đãi', 'combo', 'quảng bá']):
        return 'Quảng bá/Marketing'
    elif any(w in text for w in ['thiên nhiên', 'hang động', 'núi', 'rừng', 'biển', 'sông', 'thác']):
        return 'Thiên nhiên/Cảnh quan'
    else:
        return 'Tin tức chung'

df['content_type'] = df.apply(classify_content, axis=1)
ct_data = df.groupby(['Destination', 'content_type']).size().unstack(fill_value=0)
ct_pct = ct_data.div(ct_data.sum(axis=1), axis=0) * 100

content_types = ct_pct.columns.tolist()
x = np.arange(len(content_types))
bar_width = 0.25
for i, dest in enumerate(destinations):
    if dest in ct_pct.index:
        vals = [ct_pct.loc[dest, ct] if ct in ct_pct.columns else 0 for ct in content_types]
        ax3.bar(x + i * bar_width, vals, bar_width,
                color=COLORS[dest], label=dest, edgecolor='white')
ax3.set_title('3c. Content Mix — Phân loại nội dung\n(Content Type Distribution %)', fontsize=12, fontweight='bold')
ax3.set_xticks(x + bar_width)
ax3.set_xticklabels(content_types, rotation=25, fontsize=8, ha='right')
ax3.set_ylabel('% bài viết')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# --- 3d. Content Length Distribution (Box plot) ---
ax4 = axes[1, 1]
data_box = [df[df['Destination'] == d]['content_length'].values for d in destinations]
bp = ax4.boxplot(data_box, labels=destinations, patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2))
for patch, d in zip(bp['boxes'], destinations):
    patch.set_facecolor(COLORS[d])
    patch.set_alpha(0.7)
ax4.set_title('3d. Phân phối độ dài bài viết\n(Content Length Distribution)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Số ký tự')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r'd:\HK2_Nam3\PNKB\fig3_strategy_breakdown.png', bbox_inches='tight', dpi=150)
plt.show()
print("✔ Đã lưu: fig3_strategy_breakdown.png")

# =====================================================
# FIGURE 4: BEST PRACTICES — Top Content & Success Factors
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('BEST PRACTICES — Bài học từ đối thủ & Yếu tố thành công',
             fontsize=16, fontweight='bold', y=1.02)

# --- 4a. Top nguồn báo chí quan tâm nhất cho từng destination ---
ax1 = axes[0, 0]
# Stacked bar
top5_sources = df['source_name'].value_counts().head(6).index.tolist()
src_matrix = df[df['source_name'].isin(top5_sources)].groupby(['source_name', 'Destination']).size().unstack(fill_value=0)
src_matrix = src_matrix.reindex(top5_sources)
bottom = np.zeros(len(top5_sources))
for dest in destinations:
    if dest in src_matrix.columns:
        ax1.barh(range(len(top5_sources)), src_matrix[dest], left=bottom,
                 color=COLORS[dest], label=dest, edgecolor='white', height=0.6)
        bottom += src_matrix[dest].values
ax1.set_yticks(range(len(top5_sources)))
ax1.set_yticklabels(top5_sources, fontsize=9)
ax1.set_title('4a. Top nguồn báo chí theo điểm đến\n(Top Media Sources — Stacked)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Số bài viết')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')

# --- 4b. Keyword phổ biến trong tiêu đề (Word frequency) ---
ax2 = axes[0, 1]

def get_top_keywords(df_sub, n=10):
    stop_words = {'và', 'của', 'cho', 'với', 'được', 'từ', 'trong', 'đến', 'là', 'có',
                  'tại', 'các', 'một', 'về', 'đã', 'không', 'này', 'theo', 'như', 'nhưng',
                  'sẽ', 'trên', 'cũng', 'khi', 'để', 'hay', 'những', 'nhiều', 'đó',
                  'ra', 'người', 'sau', 'còn', 'rất', 'lại', 'bài', 'nơi', 'nào',
                  'hơn', 'mới', 'qua', 'vào', 'năm', 'ngày', 'The', 'the'}
    words = Counter()
    for title in df_sub['title'].dropna():
        for word in re.findall(r'\b\w+\b', str(title).lower()):
            if len(word) > 2 and word not in stop_words and not word.isdigit():
                words[word] += 1
    return words.most_common(n)

# So sánh keywords
for i, dest in enumerate(destinations):
    kws = get_top_keywords(df[df['Destination'] == dest], 8)
    if kws:
        words, counts = zip(*kws)
        y_pos = np.arange(len(words))
        ax2.barh(y_pos + i * 0.25, counts, height=0.22, color=COLORS[dest], label=dest, alpha=0.85)
ax2.set_title('4b. Từ khóa phổ biến trong tiêu đề\n(Title Keywords Frequency)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Tần suất')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='x')

# --- 4c. Tăng trưởng năm gần nhất vs trước đó ---
ax3 = axes[1, 0]
recent_year = df['year'].max()
prev_year = recent_year - 1
growth_data = []
for dest in destinations:
    curr = len(df[(df['Destination'] == dest) & (df['year'] == recent_year)])
    prev = len(df[(df['Destination'] == dest) & (df['year'] == prev_year)])
    growth = ((curr - prev) / prev * 100) if prev > 0 else 0
    growth_data.append(growth)

bars = ax3.bar(destinations, growth_data, color=[COLORS[d] for d in destinations],
               edgecolor='white', width=0.5)
for bar, val in zip(bars, growth_data):
    color = 'green' if val >= 0 else 'red'
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if val >= 0 else -3),
             f'{val:+.1f}%', ha='center', fontsize=12, fontweight='bold', color=color)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_title(f'4c. Tăng trưởng bài viết {recent_year} vs {prev_year}\n(YoY Growth Rate %)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Tăng trưởng (%)')
ax3.grid(True, alpha=0.3, axis='y')

# --- 4d. Heatmap: Nguồn × Loại nội dung cho Phong Nha Kẻ Bàng ---
ax4 = axes[1, 1]
pnkb = df[df['Destination'] == 'Phong Nha Kẻ Bàng']
heatmap_data = pnkb.groupby(['source_name', 'content_type']).size().unstack(fill_value=0)
top_src_pnkb = pnkb['source_name'].value_counts().head(6).index.tolist()
heatmap_data = heatmap_data.reindex(top_src_pnkb).fillna(0)

im = ax4.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
ax4.set_xticks(range(heatmap_data.shape[1]))
ax4.set_xticklabels(heatmap_data.columns, rotation=30, fontsize=8, ha='right')
ax4.set_yticks(range(heatmap_data.shape[0]))
ax4.set_yticklabels(heatmap_data.index, fontsize=9)
# Thêm số liệu vào ô
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = int(heatmap_data.values[i, j])
        if val > 0:
            ax4.text(j, i, str(val), ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if val > heatmap_data.values.max() * 0.6 else 'black')
plt.colorbar(im, ax=ax4, shrink=0.8)
ax4.set_title('4d. Heatmap: Nguồn × Loại nội dung (Phong Nha KB)\n(Source × Content Type)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(r'd:\HK2_Nam3\PNKB\fig4_best_practices.png', bbox_inches='tight', dpi=150)
plt.show()
print("✔ Đã lưu: fig4_best_practices.png")

# =====================================================
# FIGURE 5: PHÂN TÍCH SÂU PHONG NHA KẺ BÀNG
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('PHÂN TÍCH SÂU — Phong Nha Kẻ Bàng: Điểm mạnh & Cơ hội',
             fontsize=16, fontweight='bold', y=1.02)

# --- 5a. Timeline bài viết PNKB theo tháng ---
ax1 = axes[0, 0]
pnkb_monthly = pnkb.set_index('publish_date').resample('M').size()
ax1.fill_between(pnkb_monthly.index, pnkb_monthly.values, alpha=0.3, color=COLORS['Phong Nha Kẻ Bàng'])
ax1.plot(pnkb_monthly.index, pnkb_monthly.values, color=COLORS['Phong Nha Kẻ Bàng'], linewidth=2)
ax1.set_title('5a. Timeline bài viết Phong Nha KB theo tháng\n(Monthly Article Timeline)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Thời gian')
ax1.set_ylabel('Số bài viết')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax1.tick_params(axis='x', rotation=45)

# --- 5b. Content type trend cho PNKB theo năm ---
ax2 = axes[0, 1]
ct_year = pnkb.groupby(['year', 'content_type']).size().unstack(fill_value=0)
ct_year_pct = ct_year.div(ct_year.sum(axis=1), axis=0) * 100
ct_year_pct.plot(kind='area', stacked=True, ax=ax2, alpha=0.7, colormap='Set2')
ax2.set_title('5b. Xu hướng loại nội dung theo năm (PNKB)\n(Content Type Trend %)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Năm')
ax2.set_ylabel('% bài viết')
ax2.legend(fontsize=7, loc='upper left')
ax2.grid(True, alpha=0.3)

# --- 5c. So sánh Content Type giữa PNKB và đối thủ (grouped) ---
ax3 = axes[1, 0]
ct_compare = df.groupby(['Destination', 'content_type']).size().unstack(fill_value=0)
ct_compare_pct = ct_compare.div(ct_compare.sum(axis=1), axis=0) * 100
ct_compare_pct.T.plot(kind='bar', ax=ax3, color=[COLORS.get(d, '#999') for d in ct_compare_pct.index],
                       width=0.7, edgecolor='white')
ax3.set_title('5c. So sánh Content Mix giữa các điểm đến\n(Content Mix Comparison %)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Loại nội dung')
ax3.set_ylabel('% bài viết')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=25, fontsize=8, ha='right')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# --- 5d. Gap Analysis — PNKB vs đối thủ tốt nhất ---
ax4 = axes[1, 1]
metrics = ['Số bài viết', 'Độ dài TB nội dung', 'Đa dạng nguồn', 'Bài/tháng', 'Loại nội dung']
pnkb_vals = [
    total_articles.get('Phong Nha Kẻ Bàng', 0),
    avg_content.get('Phong Nha Kẻ Bàng', 0),
    source_diversity.get('Phong Nha Kẻ Bàng', 0),
    monthly_avg.get('Phong Nha Kẻ Bàng', 0),
    len(pnkb['content_type'].unique())
]
best_competitor_vals = [
    max(total_articles.get('Ba Na Hills', 0), total_articles.get('Cố đô Huế', 0)),
    max(avg_content.get('Ba Na Hills', 0), avg_content.get('Cố đô Huế', 0)),
    max(source_diversity.get('Ba Na Hills', 0), source_diversity.get('Cố đô Huế', 0)),
    max(monthly_avg.get('Ba Na Hills', 0), monthly_avg.get('Cố đô Huế', 0)),
    max(len(df[df['Destination'] == 'Ba Na Hills']['content_type'].unique()),
        len(df[df['Destination'] == 'Cố đô Huế']['content_type'].unique()))
]

# Normalize
max_all = [max(p, b) for p, b in zip(pnkb_vals, best_competitor_vals)]
pnkb_norm = [p / m * 100 if m > 0 else 0 for p, m in zip(pnkb_vals, max_all)]
comp_norm = [b / m * 100 if m > 0 else 0 for b, m in zip(best_competitor_vals, max_all)]

y = np.arange(len(metrics))
ax4.barh(y - 0.15, pnkb_norm, 0.3, color=COLORS['Phong Nha Kẻ Bàng'], label='Phong Nha KB')
ax4.barh(y + 0.15, comp_norm, 0.3, color='#FF9800', label='Đối thủ tốt nhất', alpha=0.7)
ax4.set_yticks(y)
ax4.set_yticklabels(metrics, fontsize=10)
ax4.set_xlabel('% so với mức tốt nhất')
ax4.axvline(x=100, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax4.set_title('5d. Gap Analysis — PNKB vs Đối thủ tốt nhất\n(Normalized %)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(r'd:\HK2_Nam3\PNKB\fig5_deep_analysis_pnkb.png', bbox_inches='tight', dpi=150)
plt.show()
print("✔ Đã lưu: fig5_deep_analysis_pnkb.png")

# =====================================================
# TỔNG KẾT
# =====================================================
print("\n" + "="*70)
print("📊 TỔNG KẾT EDA & CHẨN ĐOÁN — PHONG NHA KẺ BÀNG")
print("="*70)

print(f"\n📌 Tổng số bài: {len(df)} (PNKB: {len(pnkb)}, Ba Na Hills: {len(df[df['Destination']=='Ba Na Hills'])}, Huế: {len(df[df['Destination']=='Cố đô Huế'])})")
print(f"📌 PNKB chiếm {len(pnkb)/len(df)*100:.1f}% tổng bài viết")
print(f"📌 Độ dài TB bài viết PNKB: {avg_content.get('Phong Nha Kẻ Bàng', 0):.0f} ký tự")
print(f"📌 Số nguồn báo chí đưa tin PNKB: {source_diversity.get('Phong Nha Kẻ Bàng', 0)}")
print(f"📌 Tần suất TB: {monthly_avg.get('Phong Nha Kẻ Bàng', 0):.1f} bài/tháng")

print("\n🔍 NHẬN XÉT CHÍNH:")
if len(pnkb) < total_articles.max():
    print(f"  ⚠️  PNKB có ít bài viết hơn đối thủ dẫn đầu ({total_articles.idxmax()}: {total_articles.max()} bài)")
if avg_content.get('Phong Nha Kẻ Bàng', 0) < avg_content.max():
    print(f"  ⚠️  Độ dài bài viết TB thấp hơn {avg_content.idxmax()} ({avg_content.max():.0f} ký tự)")
print(f"  📈 Tăng trưởng {recent_year} vs {prev_year}: {growth_data[0]:+.1f}%")

top_ct = pnkb['content_type'].value_counts()
print(f"\n📝 TOP LOẠI NỘI DUNG PNKB:")
for ct, count in top_ct.items():
    print(f"    - {ct}: {count} bài ({count/len(pnkb)*100:.1f}%)")

print("\n" + "="*70)
print("✅ Đã tạo 5 biểu đồ phân tích:")
print("   1. fig1_trend_comparison.png     — So sánh xu hướng")
print("   2. fig2_benchmark_kpi.png        — Benchmark KPI")
print("   3. fig3_strategy_breakdown.png   — Phân rã chiến lược")
print("   4. fig4_best_practices.png       — Best practices")
print("   5. fig5_deep_analysis_pnkb.png   — Phân tích sâu PNKB")
print("="*70)
