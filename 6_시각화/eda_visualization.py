"""
발표 PPT용 EDA 시각화 코드
============================
merged_full.csv, drozy_kss_pvt.csv, state_2x2.csv를 입력으로
발표에 쓸 핵심 그래프 6종을 생성 (PNG 저장).

실행: python eda_visualization.py
필요: pandas, numpy, matplotlib, scipy
출력: fig1~fig6 PNG 파일
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# 한글 폰트 설정 (Noto Sans CJK)
for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# 색상 팔레트
C = {'blue':'#4A78E2','teal':'#2B9C8A','amber':'#EF9F27','coral':'#E24B4A',
     'purple':'#7B6FDE','gray':'#999999','green':'#639922'}

def num(df, s):
    return pd.to_numeric(df[s].astype(str).str.replace(',', '.'), errors='coerce')


def fig1_sleep_effect():
    """그림1: 수면박탈 효과 — 객관(PVT) 무반응 vs 주관 반응"""
    m = pd.read_csv('merged_full.csv')
    labels = ['SicknessQ\n(주관·아픔)', 'PANAS긍정\n(주관·기분)', '자가건강\n(주관)',
              'PVT변동성\n(객관)', 'PVT반응시간\n(객관)', 'lapse율\n(객관)']
    d = [0.52, -0.46, -0.32, 0.06, 0.04, 0.02]
    colors = [C['coral'] if abs(x) >= 0.3 and x > 0 else C['blue'] if abs(x) >= 0.3 else C['gray'] for x in d]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, d, color=colors)
    ax.axvline(0, color='#333', lw=0.8)
    ax.axvline(0.3, color=C['coral'], ls='--', lw=1, alpha=0.5)
    ax.axvline(-0.3, color=C['blue'], ls='--', lw=1, alpha=0.5)
    ax.set_xlabel("효과크기 (Cohen's d) — |0.3| 이상이 의미있는 차이")
    ax.set_title("수면박탈 효과: 주관 지표는 반응, 객관 PVT는 무반응", fontweight='bold')
    ax.set_xlim(-0.6, 0.6)
    for b, v in zip(bars, d):
        ax.text(v + (0.02 if v >= 0 else -0.02), b.get_y() + b.get_height()/2,
                f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout(); plt.savefig('fig1_sleep_effect.png', bbox_inches='tight'); plt.close()
    print("fig1_sleep_effect.png 생성")


def fig2_pvt_duration():
    """그림2: 측정시간별 예측 성능"""
    times = ['60초', '90초', '3분', '5분']
    both = [0.721, 0.761, 0.799, 0.806]
    pvt_only = [0.599, 0.616, 0.557, 0.662]
    baseline = [0.738]*4
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, both, 'o-', color=C['green'], lw=2.5, ms=9, label='PVT + 개인특성')
    ax.plot(times, pvt_only, 's-', color=C['coral'], lw=2, ms=7, label='PVT만')
    ax.plot(times, baseline, '--', color=C['gray'], lw=1.5, label='개인특성만 (측정 무관)')
    ax.set_ylabel('AUC (판별 성능)'); ax.set_xlabel('PVT 측정 시간')
    ax.set_title('측정시간별 성능 — 3분이 최적점 (5분과 거의 동일)', fontweight='bold')
    ax.set_ylim(0.5, 0.85); ax.legend(loc='lower right'); ax.grid(alpha=0.2)
    ax.annotate('3분 최적', xy=(2, 0.799), xytext=(1.3, 0.83),
                arrowprops=dict(arrowstyle='->', color=C['green']), color=C['green'], fontweight='bold')
    plt.tight_layout(); plt.savefig('fig2_pvt_duration.png', bbox_inches='tight'); plt.close()
    print("fig2_pvt_duration.png 생성")


def fig3_drozy_kss_pvt():
    """그림3: DROZY — 수면박탈 진행에 따른 KSS·PVT 동반 악화"""
    d = pd.read_csv('drozy_kss_pvt.csv')
    agg = d.groupby('test')[['kss', 'lapse']].mean()
    tests = ['오전 10시\n(각성)', '새벽 3시\n(밤샘)', '낮 12시\n(회복중)']
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax1.bar(x, agg['kss'].values, color=C['purple'], alpha=0.7, label='KSS 주관졸림')
    ax1.set_ylabel('KSS (주관 졸림, 1-9)', color=C['purple'])
    ax1.set_ylim(0, 9); ax1.set_xticks(x); ax1.set_xticklabels(tests)
    ax2 = ax1.twinx()
    ax2.plot(x, agg['lapse'].values, 'o-', color=C['coral'], lw=3, ms=10, label='PVT lapse 객관각성')
    ax2.set_ylabel('PVT lapse (객관 각성)', color=C['coral']); ax2.set_ylim(0, 26)
    ax1.set_title('DROZY: 졸림(KSS)과 각성(PVT)이 나란히 악화 (r=0.63)', fontweight='bold')
    plt.tight_layout(); plt.savefig('fig3_drozy_kss_pvt.png', bbox_inches='tight'); plt.close()
    print("fig3_drozy_kss_pvt.png 생성")


def fig4_2x2_matrix():
    """그림4: 2×2 상태 분포"""
    fig, ax = plt.subplots(figsize=(7, 6))
    cells = [(0, 1, '정상', 36, C['green']), (1, 1, '숨은 위험', 47, C['coral']),
             (0, 0, '주관적 피로', 48, C['gray']), (1, 0, '피로 확실', 38, C['amber'])]
    for cx, cy, label, n, color in cells:
        ax.add_patch(plt.Rectangle((cx, cy), 1, 1, facecolor=color, alpha=0.25, edgecolor=color, lw=2))
        ax.text(cx+0.5, cy+0.62, label, ha='center', fontsize=13, fontweight='bold')
        ax.text(cx+0.5, cy+0.4, f'{n}세션', ha='center', fontsize=15)
        ax.text(cx+0.5, cy+0.25, f'{n/169*100:.1f}%', ha='center', fontsize=10, color='#555')
    ax.text(1.5, 1.72, '⚠ 본인은 모르는 위험', ha='center', color=C['coral'], fontsize=10, fontweight='bold')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['객관 양호', '객관 위험'], fontsize=11)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['주관: 피곤', '주관: 안 피곤'], fontsize=11, rotation=90, va='center')
    ax.set_title('2×2 피로 상태 분류 (n=169)', fontweight='bold', fontsize=13)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(); plt.savefig('fig4_2x2_matrix.png', bbox_inches='tight'); plt.close()
    print("fig4_2x2_matrix.png 생성")


def fig5_hidden_risk_validation():
    """그림5: 숨은 위험 vs 피로확실 — 객관 같고 주관만 다름"""
    m = pd.read_csv('state_2x2.csv')
    m['mean_rt'] = num(m, 'mean_rt')
    m['sick'] = np.where(m['ses_num']==1, num(m, 'SicknessQ_byScanner.x'), num(m, 'SicknessQ_byScanner.y'))
    groups = ['정상', '숨은위험', '피로확실']
    rt = [m[m['state']==g]['mean_rt'].mean() for g in groups]
    sick = [m[m['state']==g]['sick'].mean() for g in groups]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(groups, rt, color=[C['green'], C['coral'], C['amber']])
    ax1.set_title('객관: PVT 반응시간(ms)', fontweight='bold'); ax1.set_ylim(240, 285)
    for i, v in enumerate(rt): ax1.text(i, v+0.5, f'{v:.0f}', ha='center')
    ax1.annotate('숨은위험 ≈ 피로확실\n(둘 다 객관 위험)', xy=(1.5, 275), ha='center', fontsize=9, color='#555')
    ax2.bar(groups, sick, color=[C['green'], C['coral'], C['amber']])
    ax2.set_title('주관: 피로 자각(SicknessQ)', fontweight='bold'); ax2.set_ylim(10, 24)
    for i, v in enumerate(sick): ax2.text(i, v+0.2, f'{v:.1f}', ha='center')
    ax2.annotate('숨은위험 << 피로확실\n(자각만 다름)', xy=(1.5, 16), ha='center', fontsize=9, color='#555')
    fig.suptitle('숨은 위험의 정체: 객관은 위험하나 본인은 인지 못함', fontweight='bold', fontsize=13)
    plt.tight_layout(); plt.savefig('fig5_hidden_risk.png', bbox_inches='tight'); plt.close()
    print("fig5_hidden_risk.png 생성")


def fig6_vif_reduction():
    """그림6: VIF 변수선택 (13개 → 5개)"""
    before = [('lapse_rate', 356.6), ('mean_rt', 179.9), ('sd_rt', 170.8),
              ('cv_rt', 119.7), ('median_rt', 71.6), ('mean_speed', 46.0), ('slowest10_rt', 20.9)]
    after = [('mean_rt', 1.4), ('sd_rt', 1.3), ('lapses', 1.3), ('n_valid', 1.2), ('false_starts', 1.1)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    n1 = [x[0] for x in before]; v1 = [x[1] for x in before]
    ax1.barh(n1[::-1], v1[::-1], color=C['coral'])
    ax1.axvline(10, color='#333', ls='--', lw=1); ax1.set_xscale('log')
    ax1.set_title('선정 전 (VIF, 로그축) — 대부분 10 초과', fontweight='bold')
    ax1.text(10, -0.5, 'VIF=10', fontsize=8)
    n2 = [x[0] for x in after]; v2 = [x[1] for x in after]
    ax2.barh(n2[::-1], v2[::-1], color=C['green'])
    ax2.axvline(10, color='#333', ls='--', lw=1); ax2.set_xlim(0, 12)
    ax2.set_title('선정 후 5개 — 전부 2 미만 (공선성 해결)', fontweight='bold')
    fig.suptitle('VIF 다중공선성 처리: PVT 파생변수 13개 → 5개', fontweight='bold', fontsize=13)
    plt.tight_layout(); plt.savefig('fig6_vif.png', bbox_inches='tight'); plt.close()
    print("fig6_vif.png 생성")


if __name__ == '__main__':
    fig1_sleep_effect()
    fig2_pvt_duration()
    fig3_drozy_kss_pvt()
    fig4_2x2_matrix()
    fig5_hidden_risk_validation()
    fig6_vif_reduction()
    print("\n총 6개 EDA 그래프 생성 완료")
