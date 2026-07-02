"""
EDA 전체 개요 슬라이드 (참고 슬라이드 형식 재현)
================================================
왼쪽: 기본정보 요약(터미널 스타일) / 오른쪽: 분포 그래프 / 아래: 기술통계 표
출력: eda_overview_slide.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

def num(df, s):
    return pd.to_numeric(df[s].astype(str).str.replace(',', '.'), errors='coerce')

m = pd.read_csv('merged_full.csv')
pvt = ['mean_rt', 'sd_rt', 'lapses', 'n_valid', 'false_starts']
for c in pvt:
    m[c] = num(m, c)

DARK = '#1a1a2e'      # 터미널 배경
GREEN = '#4ade80'     # 터미널 텍스트
C = {'blue':'#4A78E2','teal':'#2B9C8A','amber':'#EF9F27','coral':'#E24B4A','navy':'#1F3864'}

fig = plt.figure(figsize=(16, 9), facecolor='white')
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1],
              width_ratios=[1.1, 1, 1], hspace=0.28, wspace=0.25,
              left=0.03, right=0.97, top=0.88, bottom=0.06)

# 제목
fig.text(0.03, 0.945, '전체 데이터 개요 및 정보 (EDA)', fontsize=26, fontweight='bold', color=C['navy'])
fig.text(0.03, 0.905, 'OpenNeuro ds000201 — 부분 수면박탈 PVT·설문 통합 데이터', fontsize=13, color='#555')

# ── 왼쪽: 기본정보 요약 (터미널 스타일) ──
ax_info = fig.add_subplot(gs[0, 0]); ax_info.axis('off')
ax_info.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=DARK, transform=ax_info.transAxes))
info = """1-1. 기본 정보 요약
========================================
총 세션 수    : 169 세션
고유 참가자   : 85 명
컬럼 수       : 124 개
세션 구성     : 1인당 2세션(정상/박탈)
데이터 성격   : 교차설계(within-subject)

[참가자 구성]
 연령군   : Young 46, Old 39
 성별     : Female 44, Male 41
 수면조건 : 정상 85, 박탈 84

[주요 변수 유형]
 # PVT 파생  : mean_rt, sd_rt, lapses,
               n_valid, false_starts
 # 설문      : STAI, PSS, ISI, BFI ...
 # 타깃 y    : SicknessQ (주관 피로)

dtypes: float64, int64, object
측정시점: 수면 다음날 저녁(17-20시)"""
ax_info.text(0.06, 0.95, info, transform=ax_info.transAxes, fontsize=10.5,
             color=GREEN, family='monospace', va='top', linespacing=1.5)

# ── 오른쪽 상단: 연령대별 분포 ──
ax1 = fig.add_subplot(gs[0, 1])
age_order = m.drop_duplicates('sub')['AgeGroup'].value_counts()
ax1.bar(['Young\n(20-30)', 'Old\n(65-75)'], [46, 39], color=[C['teal'], C['navy']])
ax1.set_title('연령군별 참가자 수', fontsize=12, fontweight='bold')
for i, v in enumerate([46, 39]):
    ax1.text(i, v+0.5, str(v), ha='center', fontweight='bold')
ax1.set_ylim(0, 52)

# ── 오른쪽 중앙: 성별 파이 ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.pie([44, 41], labels=['여성', '남성'], autopct='%1.1f%%',
        colors=[C['blue'], C['amber']], startangle=90, textprops={'fontsize': 11})
ax2.set_title('성별 비율', fontsize=12, fontweight='bold')

# ── 아래: PVT 기술통계 표 ──
ax_tbl = fig.add_subplot(gs[1, :]); ax_tbl.axis('off')
ax_tbl.text(0.0, 1.02, 'PVT 파생변수 기술통계 (5분 기준, n=169)',
            fontsize=13, fontweight='bold', color=C['navy'], transform=ax_tbl.transAxes)
desc = m[pvt].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2)
labels_kr = {'mean_rt':'평균반응시간(ms)','sd_rt':'반응변동성','lapses':'주의실패수',
             'n_valid':'유효반응수','false_starts':'성급반응수'}
col_labels = ['변수', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
cell_text = []
for idx, row in desc.iterrows():
    cell_text.append([labels_kr[idx]] + [f'{row[c]:.2f}' for c in ['mean','std','min','25%','50%','75%','max']])
tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0, 0.15, 1, 0.75])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
for j in range(len(col_labels)):
    c = tbl[0, j]; c.set_facecolor(C['navy']); c.set_text_props(color='white', fontweight='bold')
for i in range(1, len(cell_text)+1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            tbl[i, j].set_facecolor('#f2f4f8')
        if j == 0:
            tbl[i, j].set_text_props(fontweight='bold')

plt.savefig('eda_overview_slide.png', dpi=110, bbox_inches='tight', facecolor='white')
plt.close()
print("eda_overview_slide.png 생성")
