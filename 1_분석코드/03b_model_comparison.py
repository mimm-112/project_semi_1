"""
ds000201 Sleepy Brain — 측정시간 검증 및 머신러닝 모델 비교
=============================================================
1) PVT 측정시간(60/90/180/300초)별 피로도 예측 성능 비교
2) 부스팅 계열 알고리즘 비교 및 최종 모델 선정

실행: python 03_model_comparison.py
필요: pandas, numpy, scikit-learn, xgboost, lightgbm
사전: pipeline_multiwindow.py로 pvt_{60,90,180,300}s.csv 생성되어 있어야 함
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

PVT_FEATS = ['mean_rt', 'sd_rt', 'lapses', 'n_valid', 'false_starts']
PERSON_FEATS = ['age_old', 'sex_m', 'ESS', 'ISI', 'BFI_Neuroticism', 'STAI_T']
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')


def load_target_and_person():
    """타깃(SicknessQ)과 개인특성 — 측정시간과 무관하게 고정"""
    m = pd.read_csv('merged_full.csv')
    m['sick'] = np.where(m['ses_num'] == 1,
                         to_num(m['SicknessQ_byScanner.x']),
                         to_num(m['SicknessQ_byScanner.y']))
    m['age_old'] = (m['AgeGroup'] == 'Old').astype(int)
    m['sex_m'] = (m['Sex'] == 'Male').astype(int)
    for c in ['ESS', 'ISI', 'BFI_Neuroticism', 'STAI_T']:
        m[c] = to_num(m[c])
    return m[['sub', 'ses_num', 'sick'] + PERSON_FEATS]


def gb_model():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      learning_rate=0.1, subsample=0.8, random_state=42)


def duration_analysis(key):
    """측정시간별 예측 성능 비교"""
    print("=" * 68)
    print("A. PVT 측정시간별 피로도 예측 성능 (AUC)")
    print("=" * 68)
    print(f"{'측정시간':<10}{'trial':>8}{'PVT만':>10}{'PVT+개인':>11}{'개인만':>9}")
    for cutoff, wname in [(60, '60초'), (90, '90초'), (180, '3분'), (300, '5분')]:
        p = pd.read_csv(f'pvt_{cutoff}s.csv')
        d = p.merge(key, on=['sub', 'ses_num'], how='left').dropna(
            subset=['sick'] + PVT_FEATS + PERSON_FEATS)
        y = (d['sick'].values >= np.median(d['sick'].values)).astype(int)
        auc = lambda feats: cross_val_score(gb_model(), d[feats].values, y,
                                            cv=CV, scoring='roc_auc').mean()
        print(f"{wname:<10}{d['n_trial'].mean():>8.1f}"
              f"{auc(PVT_FEATS):>10.3f}{auc(PVT_FEATS+PERSON_FEATS):>11.3f}"
              f"{auc(PERSON_FEATS):>9.3f}")


def algorithm_comparison(key):
    """부스팅 계열 알고리즘 비교 (5분 기준)"""
    print("\n" + "=" * 68)
    print("B. 알고리즘 비교 (5분 PVT + 개인특성, 5-fold AUC)")
    print("=" * 68)
    p = pd.read_csv('pvt_300s.csv')
    d = p.merge(key, on=['sub', 'ses_num'], how='left').dropna(
        subset=['sick'] + PVT_FEATS + PERSON_FEATS)
    X = d[PVT_FEATS + PERSON_FEATS].values
    y = (d['sick'].values >= np.median(d['sick'].values)).astype(int)

    models = {
        '로지스틱': make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, class_weight='balanced')),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        '랜덤포레스트': RandomForestClassifier(n_estimators=400, max_depth=6,
                                        max_features=0.5, class_weight='balanced',
                                        random_state=42),
        'GradientBoosting': gb_model(),
    }
    # XGBoost, LightGBM은 설치돼 있으면 추가
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(n_estimators=150, max_depth=3,
                                              learning_rate=0.1, subsample=0.8,
                                              eval_metric='logloss', random_state=42)
    except ImportError:
        print("(xgboost 미설치 — 건너뜀)")
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=150, max_depth=3,
                                                learning_rate=0.1, subsample=0.8,
                                                verbose=-1, random_state=42)
    except ImportError:
        print("(lightgbm 미설치 — 건너뜀)")

    results = []
    for name, mdl in models.items():
        auc = cross_val_score(mdl, X, y, cv=CV, scoring='roc_auc').mean()
        results.append((name, auc))
        print(f"  {name:18s} AUC={auc:.3f}")
    best = max(results, key=lambda x: x[1])
    print(f"\n최종 선정: {best[0]} (AUC={best[1]:.3f})")


def main():
    key = load_target_and_person()
    duration_analysis(key)
    algorithm_comparison(key)
    print("\n완료.")


if __name__ == '__main__':
    main()
