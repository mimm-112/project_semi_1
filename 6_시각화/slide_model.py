"""
모델링 발표 슬라이드 (참고 형식 재현)
=======================================
슬라이드1: 분석 STEP 프로세스 바 + 모델 비교 표 + 핵심 요약
슬라이드2: 모델 성능 결과 표
출력: slide_model_process.png, slide_model_result.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrow, Polygon

for fp in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']:
    try:
        fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2C3E5A'
GREEN = '#8BC34A'
GRAY = '#8A96A8'
LIGHT = '#EAEEF3'


def step_bar(ax, steps, active_idx):
    """상단 STEP 프로세스 화살표 바"""
    n = len(steps)
    w = 1.0 / n
    for i, (num, label) in enumerate(steps):
        x = i * w
        color = GREEN if i == active_idx else (NAVY if i < active_idx else GRAY)
        # 화살표 모양 (오각형)
        pts = [[x, 0.3], [x+w*0.82, 0.3], [x+w*0.97, 0.5],
               [x+w*0.82, 0.7], [x, 0.7], [x+w*0.15, 0.5]]
        if i == 0:
            pts = [[x, 0.3], [x+w*0.82, 0.3], [x+w*0.97, 0.5], [x+w*0.82, 0.7], [x, 0.7]]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor='white', lw=2,
                             transform=ax.transAxes))
        ax.text(x+w*0.45, 0.58, num, ha='center', va='center', color='white',
                fontsize=10, fontweight='bold', transform=ax.transAxes)
        ax.text(x+w*0.45, 0.44, label, ha='center', va='center', color='white',
                fontsize=12, fontweight='bold', transform=ax.transAxes)


def make_process_slide():
    fig = plt.figure(figsize=(16, 9), facecolor='white')

    # 제목
    fig.text(0.04, 0.94, '모델 설계 및 개발 프로세스', fontsize=26, fontweight='bold', color=NAVY)

    # STEP 바
    ax_step = fig.add_axes([0.03, 0.80, 0.94, 0.10]); ax_step.axis('off')
    steps = [('STEP 1', '데이터 정제'), ('STEP 2', '변수 선택'), ('STEP 3', '데이터 분할'),
             ('STEP 4', '모델 비교'), ('STEP 5', '튜닝·최적화'), ('STEP 6', '모델 평가')]
    step_bar(ax_step, steps, active_idx=3)

    # 모델 비교 표
    ax_tbl = fig.add_axes([0.03, 0.30, 0.94, 0.44]); ax_tbl.axis('off')
    headers = ['모델 구분', '모델명', '모델 선정 이유', '주요 기술적 특징']
    rows = [
        ['선형 모델', 'M1: Logistic', "성능 비교의 기준점(Baseline) 역할", '계수로 변수 영향 방향이 명확'],
        ['앙상블(부스팅)', 'M2: AdaBoost', '오분류 샘플 가중으로 성능 보완', '약한 학습기 순차 결합'],
        ['앙상블(배깅)', 'M3: RandomForest', '판별력·안정성 균형, 과적합에 강함', '이상치에 강건, 변수중요도 제공'],
        ['앙상블(부스팅)', 'M4: XGBoost', '변수 간 상호작용으로 성능 최대치 탐색', '정형 데이터 고성능 표준'],
        ['앙상블(부스팅)', 'M5: LightGBM', '빠른 학습으로 하이퍼파라미터 탐색', '학습 속도·메모리 효율 우수'],
        ['앙상블(부스팅)', 'M6: GradientBoosting ★', '소표본에서 정규화 강해 최고 성능', '최종 선정 (AUC 0.806)'],
    ]
    col_x = [0.03, 0.16, 0.34, 0.70]
    col_w = [0.13, 0.18, 0.36, 0.30]
    # 헤더
    for cx, cw, h in zip(col_x, col_w, headers):
        ax_tbl.add_patch(plt.Rectangle((cx, 0.88), cw, 0.10, facecolor=NAVY,
                         transform=ax_tbl.transAxes))
        ax_tbl.text(cx+0.01, 0.93, h, color='white', fontsize=12, fontweight='bold',
                    va='center', transform=ax_tbl.transAxes)
    # 행
    rh = 0.145
    for r, row in enumerate(rows):
        y = 0.88 - (r+1)*rh
        is_final = '★' in row[1]
        bg = '#E8F0E0' if is_final else ('#F5F7FA' if r % 2 else 'white')
        for cx, cw in zip(col_x, col_w):
            ax_tbl.add_patch(plt.Rectangle((cx, y), cw, rh, facecolor=bg,
                             edgecolor='#D5DBE3', lw=0.8, transform=ax_tbl.transAxes))
        ax_tbl.text(col_x[0]+0.01, y+rh/2, row[0], fontsize=10.5, va='center',
                    transform=ax_tbl.transAxes)
        ax_tbl.text(col_x[1]+0.01, y+rh/2, row[1], fontsize=10.5, va='center',
                    fontweight='bold' if is_final else 'normal',
                    color=NAVY if is_final else 'black', transform=ax_tbl.transAxes)
        ax_tbl.text(col_x[2]+0.01, y+rh/2, row[2], fontsize=10, va='center',
                    transform=ax_tbl.transAxes)
        ax_tbl.text(col_x[3]+0.01, y+rh/2, row[3], fontsize=10, va='center',
                    transform=ax_tbl.transAxes)

    # 하단 핵심 요약 (체크 3개)
    ax_sum = fig.add_axes([0.03, 0.02, 0.94, 0.24]); ax_sum.axis('off')
    summary = [
        ('해석 중심 모델', '의료·보건 표준인 Logistic Regression으로 기준점(Baseline) 설정'),
        ('안정성 중심 모델', '이상치에 강하고 과적합 방지에 탁월한 RandomForest로 견고함 확보'),
        ('성능 극대화 모델', 'XGBoost·LightGBM·GradientBoosting 부스팅 계열로 예측 성능 한계 탐색 → GradientBoosting 최종 선정'),
    ]
    for i, (title, desc) in enumerate(summary):
        y = 0.82 - i*0.32
        ax_sum.text(0.0, y, '✓', fontsize=16, color=GREEN, fontweight='bold',
                    transform=ax_sum.transAxes)
        ax_sum.text(0.04, y, title, fontsize=13, fontweight='bold', color=NAVY,
                    va='center', transform=ax_sum.transAxes)
        ax_sum.text(0.20, y, desc, fontsize=11.5, va='center',
                    transform=ax_sum.transAxes)

    plt.savefig('slide_model_process.png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    print("slide_model_process.png 생성")


def make_result_slide():
    """모델 성능 결과 표 슬라이드"""
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(0.04, 0.93, '모델 성능 비교 및 최종 선정 결과', fontsize=26, fontweight='bold', color=NAVY)
    fig.text(0.04, 0.88, '검증: 5-fold 층화 교차검증 (train 약 80% / test 약 20%, 5회 반복) · 지표: AUC', fontsize=13, color='#555')

    ax = fig.add_axes([0.04, 0.30, 0.92, 0.52]); ax.axis('off')
    headers = ['순위', '모델', '계열', 'AUC', '선정 기준(Recall 우선)']
    rows = [
        ['1', 'GradientBoosting ★', '부스팅', '0.806', '최종 선정 — 최고 성능'],
        ['2', 'RandomForest', '배깅', '0.792', '안정적, 2위'],
        ['3', 'XGBoost (튜닝)', '부스팅', '0.797', '튜닝 후 개선'],
        ['4', 'AdaBoost', '부스팅', '0.777', ''],
        ['5', 'LightGBM', '부스팅', '0.772', ''],
        ['6', 'XGBoost (기본)', '부스팅', '0.770', '소표본에서 GB보다 낮음'],
        ['7', 'Logistic', '선형', '0.746', '기준선(Baseline)'],
    ]
    col_x = [0.04, 0.12, 0.36, 0.50, 0.62]
    col_w = [0.08, 0.24, 0.14, 0.12, 0.34]
    for cx, cw, h in zip(col_x, col_w, headers):
        ax.add_patch(plt.Rectangle((cx, 0.90), cw, 0.10, facecolor=NAVY, transform=ax.transAxes))
        ax.text(cx+0.01, 0.95, h, color='white', fontsize=12, fontweight='bold', va='center', transform=ax.transAxes)
    rh = 0.125
    for r, row in enumerate(rows):
        y = 0.90 - (r+1)*rh
        is_final = '★' in row[1]
        bg = '#E8F0E0' if is_final else ('#F5F7FA' if r % 2 else 'white')
        for cx, cw in zip(col_x, col_w):
            ax.add_patch(plt.Rectangle((cx, y), cw, rh, facecolor=bg, edgecolor='#D5DBE3', lw=0.8, transform=ax.transAxes))
        vals = row
        for k, (cx, cw) in enumerate(zip(col_x, col_w)):
            ax.text(cx+0.01, y+rh/2, vals[k], fontsize=11.5, va='center',
                    fontweight='bold' if (is_final and k in (1,3)) else 'normal',
                    color=NAVY if is_final else 'black', transform=ax.transAxes)

    # 하단 핵심 메시지
    ax2 = fig.add_axes([0.04, 0.05, 0.92, 0.20]); ax2.axis('off')
    msgs = [
        '소표본(169세션)에서는 대용량에 강한 XGBoost/LightGBM보다 GradientBoosting의 정규화가 효과적',
        '모델 선정 1차 기준은 Recall — 위험 근로자를 놓치지 않는 것이 산업안전의 최우선',
        '성능 개선: Logistic 0.746 → 트리 전환 → 튜닝 → GradientBoosting 0.806 (+0.060)',
    ]
    for i, msg in enumerate(msgs):
        y = 0.85 - i*0.35
        ax2.text(0.0, y, '▪', fontsize=13, color=GREEN, transform=ax2.transAxes)
        ax2.text(0.025, y, msg, fontsize=12, va='center', transform=ax2.transAxes)

    plt.savefig('slide_model_result.png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close()
    print("slide_model_result.png 생성")


if __name__ == '__main__':
    make_process_slide()
    make_result_slide()
    print("완료")
