"""
ds000201 Sleepy Brain — EDA 및 통계 분석
=========================================
merged_full.csv를 입력으로, 수면박탈 효과 검정·상관·교란변수 통제 등 EDA 수행.

실행: python 02_eda_analysis.py
필요: pandas, numpy, scipy, statsmodels
"""
import pandas as pd
import numpy as np
from scipy import stats

INPUT = 'merged_full.csv'


def to_num(series):
    """유럽식 쉼표 소수점(19,78 → 19.78) 처리 후 숫자화"""
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')


def eda_sleep_deprivation_effect(m):
    """수면박탈이 PVT 지표에 미치는 효과 (같은 사람 짝 비교)"""
    print("=" * 64)
    print("1. 수면박탈 효과 — 정상 vs 박탈 짝 비교 (대응표본 t검정)")
    print("=" * 64)
    metrics = ['mean_rt', 'median_rt', 'lapse_rate', 'sd_rt', 'mean_speed', 'slowest10_rt']
    print(f"{'지표':<16}{'정상':>10}{'박탈':>10}{'차이':>9}{'p값':>9}")
    for mt in metrics:
        piv = m.pivot_table(index='sub', columns='sleep_deprived', values=mt).dropna()
        piv.columns = ['normal', 'deprived']
        t, pv = stats.ttest_rel(piv['deprived'], piv['normal'])
        star = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
        print(f"{mt:<16}{piv['normal'].mean():>10.2f}{piv['deprived'].mean():>10.2f}"
              f"{piv['deprived'].mean()-piv['normal'].mean():>9.2f}{pv:>9.4f} {star}")


def eda_subjective_measures(m):
    """세션별 주관 지표의 수면박탈 효과 (신호가 있는 변수 탐색)"""
    print("\n" + "=" * 64)
    print("2. 주관 지표의 수면박탈 효과 (byScanner 변수)")
    print("=" * 64)

    def get_pair(row, v1, v2):
        s1, s2 = to_num(pd.Series([row[v1]]))[0], to_num(pd.Series([row[v2]]))[0]
        if row['Sl_cond'] == 1:
            return s2, s1  # 세션1이 박탈
        elif row['Sl_cond'] == 2:
            return s1, s2
        return np.nan, np.nan

    pairs = {
        '자가건강': ('SelfRatedHealth_byScanner.x', 'SelfRatedHealth_byScanner.y'),
        'PANAS긍정': ('PAn/aS_Positive_byScanner.x', 'PAn/aS_Positive_byScanner.y'),
        'PANAS부정': ('PAn/aS_Negative_byScanner.x', 'PAn/aS_Negative_byScanner.y'),
        'SicknessQ': ('SicknessQ_byScanner.x', 'SicknessQ_byScanner.y'),
    }
    print(f"{'변수':<14}{'정상':>9}{'박탈':>9}{'차이':>8}{'p값':>9}{'Cohen d':>9}")
    p = m.drop_duplicates('sub')
    for name, (v1, v2) in pairs.items():
        if v1 not in p.columns or v2 not in p.columns:
            continue
        normal, deprived = [], []
        for _, row in p.iterrows():
            n, d = get_pair(row, v1, v2)
            if pd.notna(n) and pd.notna(d):
                normal.append(n); deprived.append(d)
        normal, deprived = np.array(normal), np.array(deprived)
        diff = deprived - normal
        t, pv = stats.ttest_rel(deprived, normal)
        dcohen = diff.mean() / diff.std()
        star = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'n.s.'
        print(f"{name:<14}{normal.mean():>9.2f}{deprived.mean():>9.2f}"
              f"{diff.mean():>+8.2f}{pv:>9.4f}{dcohen:>9.2f} {star}")


def eda_age_control(m):
    """나이 교란변수 통제 (편상관·2요인 회귀)"""
    import statsmodels.formula.api as smf
    print("\n" + "=" * 64)
    print("3. 교란변수(나이) 통제 분석")
    print("=" * 64)
    young = m[m['AgeGroup'] == 'Young']['mean_rt']
    old = m[m['AgeGroup'] == 'Old']['mean_rt']
    t, pv = stats.ttest_ind(old, young)
    print(f"나이 효과: 젊은층 {young.mean():.1f}ms vs 노인층 {old.mean():.1f}ms, p={pv:.4f}")

    m = m.copy()
    m['age_bin'] = (m['AgeGroup'] == 'Old').astype(int)
    mod = smf.ols('mean_rt ~ sleep_deprived + age_bin', data=m).fit()
    print(f"2요인 회귀 R²={mod.rsquared:.3f} "
          f"(수면박탈 p={mod.pvalues['sleep_deprived']:.3f}, 나이 p={mod.pvalues['age_bin']:.3f})")


def main():
    m = pd.read_csv(INPUT)
    eda_sleep_deprivation_effect(m)
    eda_subjective_measures(m)
    eda_age_control(m)
    print("\n완료.")


if __name__ == '__main__':
    main()
