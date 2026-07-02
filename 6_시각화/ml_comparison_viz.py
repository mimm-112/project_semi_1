"""
발표 PPT용 머신러닝 알고리즘 비교 시각화
==========================================
여러 알고리즘을 적용·비교한 결과와 성능 개선 과정을 그래프로 생성.

실행: python ml_comparison_viz.py
필요: pandas, numpy, matplotlib
출력: ml1~ml3 PNG 파일

※ 성능 수치는 03b_model_comparison.py의 5-fold CV 결과 (AUC).
   재현하려면 03b를 실행해 값을 갱신할 것.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

C = {'blue':'#4A78E2','teal':'#2B9C8A','amber':'#EF9F27','coral':'#E24B4A',
     'purple':'#7B6FDE','gray':'#BBBBBB','green':'#639922','navy':'#1F3864'}


def ml1_algorithm_comparison():
    """알고리즘 7종 비교 (최종 모델 강조)"""
    models = ['Logistic\n(기준선)', 'AdaBoost', 'LightGBM', 'XGBoost',
              'RandomForest', 'GradientBoosting\n(최종)']
    auc = [0.746, 0.777, 0.772, 0.770, 0.792, 0.806]
    colors = [C['gray']]*5 + [C['navy']]
    colors[0] = C['coral']  # 기준선 강조
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, auc, color=colors)
    ax.set_ylabel('AUC (5-fold 교차검증)')
    ax.set_ylim(0.70, 0.83)
    ax.set_title('머신러닝 알고리즘 비교 — GradientBoosting 최종 선정', fontweight='bold', fontsize=13)
    ax.axhline(0.746, color=C['coral'], ls='--', lw=1, alpha=0.5)
    ax.axhline(0.8, color=C['green'], ls=':', lw=1, alpha=0.6)
    ax.text(0.1, 0.802, 'AUC 0.8 (우수 기준)', fontsize=8, color=C['green'])
    for b, v in zip(bars, auc):
        ax.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.3f}',
                ha='center', fontweight='bold' if v == 0.806 else 'normal', fontsize=10)
    plt.xticks(fontsize=9)
    plt.tight_layout(); plt.savefig('ml1_algorithm_comparison.png', bbox_inches='tight'); plt.close()
    print("ml1_algorithm_comparison.png 생성")


def ml2_improvement_process():
    """성능 개선 과정 (기준선 → 트리 → 튜닝)"""
    stages = ['① Logistic\n기준선', '② RandomForest\n트리 전환', '③ GradientBoost\n기본', '④ GradientBoost\n튜닝(최종)']
    auc = [0.746, 0.792, 0.769, 0.806]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(stages))
    ax.plot(x, auc, 'o-', color=C['navy'], lw=2.5, ms=12)
    for i, (xi, v) in enumerate(zip(x, auc)):
        ax.annotate(f'{v:.3f}', (xi, v), textcoords="offset points",
                    xytext=(0, 14), ha='center', fontweight='bold', fontsize=11)
    ax.fill_between(x, 0.72, auc, alpha=0.08, color=C['navy'])
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel('AUC'); ax.set_ylim(0.72, 0.83)
    ax.set_title('성능 개선 과정: 0.746 → 0.806', fontweight='bold', fontsize=13)
    ax.grid(alpha=0.2, axis='y')
    ax.annotate('+0.060 향상', xy=(3, 0.806), xytext=(2.2, 0.75),
                arrowprops=dict(arrowstyle='->', color=C['green']),
                color=C['green'], fontweight='bold', fontsize=11)
    plt.tight_layout(); plt.savefig('ml2_improvement.png', bbox_inches='tight'); plt.close()
    print("ml2_improvement.png 생성")


def ml3_feature_and_input():
    """(좌) 변수 중요도 + (우) 입력 세트별 성능"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 좌: 변수 중요도
    feats = ['특성불안\n(STAI)', 'PVT반응시간', 'PVT변동성', 'PVT반응수', '주간졸림', '신경증']
    imp = [0.30, 0.23, 0.14, 0.08, 0.07, 0.07]
    colors = [C['purple'], C['blue'], C['blue'], C['blue'], C['purple'], C['purple']]
    ax1.barh(feats[::-1], imp[::-1], color=colors[::-1])
    ax1.set_xlabel('중요도'); ax1.set_title('변수 중요도 (최종 모델)', fontweight='bold')
    ax1.text(0.30, 5, ' 주관·심리 특성', fontsize=8, color=C['purple'], va='center')
    ax1.text(0.23, 4, ' 객관 PVT', fontsize=8, color=C['blue'], va='center')
    # 우: 입력 세트별 성능
    sets = ['개인특성만', 'PVT만', 'PVT+개인특성\n(최종)']
    vals = [0.738, 0.559, 0.806]
    bars = ax2.bar(sets, vals, color=[C['purple'], C['gray'], C['navy']])
    ax2.set_ylabel('AUC'); ax2.set_ylim(0.5, 0.85)
    ax2.set_title('입력 조합별 성능', fontweight='bold')
    for b, v in zip(bars, vals):
        ax2.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontweight='bold')
    ax2.annotate('두 축 결합이\n최고 성능', xy=(2, 0.806), xytext=(1.0, 0.79),
                 fontsize=9, color=C['navy'])
    fig.suptitle('무엇이 피로 판별을 이끄는가', fontweight='bold', fontsize=13)
    plt.tight_layout(); plt.savefig('ml3_feature_input.png', bbox_inches='tight'); plt.close()
    print("ml3_feature_input.png 생성")


if __name__ == '__main__':
    ml1_algorithm_comparison()
    ml2_improvement_process()
    ml3_feature_and_input()
    print("\n총 3개 ML 비교 그래프 생성 완료")
