# 근로자 피로·각성 상태 판별 서비스 — DA 파트 산출물

산업현장 근로자·안전관리자를 위한 피로/각성 상태 판별 서비스의 데이터 분석 전체 산출물.

---

## 프로젝트 한눈에 보기

**목표:** PVT(반응속도 검사) + 주관 설문 + 개인특성으로 근로자의 피로/각성 상태 판별

**핵심 발견:** 피로는 독립된 두 축(객관 PVT / 주관 자각)으로 구성되며,
둘이 어긋나는 "숨은 위험"(본인은 모르는 각성 저하)이 산업안전의 핵심 (27.8% 실재)

**최종 모델:** GradientBoosting, AUC 0.806

**데이터:** ds000201 (주, 85명) + DROZY (검증, 14명)

---

## 폴더 구성 (6개 폴더, 총 43개 파일)

### 1_분석코드 (Python 4개) — 실행 순서

| 파일 | 내용 |
|---|---|
| 01_pvt_pipeline.py | PVT 정제 → 파생변수 → 위험도 분류 → 설문 병합 |
|원시 PVT(Psychomotor Vigilance Task) 데이터를 불러와 5분 단위로 절삭한 후, 반응 시간(RT) 및 지연(Lapse) 등의 핵심 파생변수를 계산합니다. 또한 계산된 지표를 바탕으로 피로도(위험/주의/양호)를 분류하고 설문 데이터와 병합하여 분석용 마스터 데이터를 만듭니다.|
| 02_eda_analysis.py | 통계 검정 (수면박탈 효과·주관지표·나이 통제) |
|1번 스크립트에서 만든 merged_full.csv를 사용하여, 수면 박탈 조건이 PVT 지표와 주관적 지표에 미치는 영향을 대응표본 t-검정(Paired t-test)으로 분석합니다. 또한 연령(Age)과 같은 교란 변수를 통제한 회귀 분석을 수행합니다.|
| 03a_multiwindow_pvt.py | PVT를 60/90/180/300초로 재계산 |
|PVT 검사는 보통 10분 정도 진행되지만 시간 단축 가능성을 보기 위해 검사 시작 후 60초, 90초, 3분, 5분 지점에서 각각 데이터를 끊어 파생변수를 별도로 재계산합니다.|
| 03b_model_comparison.py | 측정시간별 성능 + 알고리즘 비교 |
|다양한 기계학습 모델을 사용해, PVT 수행 지표가 피로도(SicknessQ 등)를 얼마나 잘 예측(이진 분류)할 수 있는지 확인합니다. 측정 시간별(60초~5분) 예측 성능을 비교하고, 최적의 부스팅 알고리즘을 찾아냅니다.|

### 2_데이터 (CSV/TSV 9개)

| 파일 | 내용 |
|---|---|
| merged_full.csv | **메인 통합 데이터** (169세션 × 124변수) |
| pvt_5min.csv | 5분 PVT 파생변수 + 위험도 등급 |
| pvt_60s/90s/180s/300s.csv | 측정시간별 PVT 파생변수 (4개) |
| state_2x2.csv | 2×2 상태 분류 결과 |
| vulnerability_residual.csv | 개인 취약성 점수 (잔차 기반) |
| drozy_kss_pvt.csv | DROZY 검증 데이터 (KSS-PVT, 36세션) |
| participants.tsv | 원본 설문 (재현용) |

### 3_모델 (1개)

| 파일 | 내용 |
|---|---|
| best_model.pkl | 최종 모델 (GradientBoosting, AUC 0.806) |

### 4_리포트 (Markdown 7개) — 읽는 순서 추천

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | 최종_분석리포트.md | **전체 종합** (비전공자용, 실험조건·측정시점) |
| 2 | 데이터셋_선정과정.md | 6종 검토 → 선정 + DROZY 검증 |
| 3 | EDA_리포트.md | 데이터소개·기본정보·기술통계·전처리·분석 |
| 4 | 변수선택_전과정_상세.md | 13→5개(VIF), 106→9개(필터링) 전 과정 |
| 5 | 센터프로젝트_DA상세.md | 측정시간·모델비교·검증방식(train/test) |
| 6 | 2x2구조_검증결과.md | 숨은위험 27.8% 검증 + 개인차 한계 |
| 7 | 프로젝트_범위_구분.md | 증명된 것 vs 향후 과제 구분 |

### 5_발표문서 (3개)

| 파일 | 내용 |
|---|---|
| 발표_PPT.pptx | **완성된 발표 슬라이드 13장** |
| 세미1_발표_상세본.md | 논문 수준 상세 원고 (8장, 491줄) |
| 프로젝트_기획서.docx | 양식 채운 기획서 (Word) |

### 6_시각화

**코드 4개:**
| 파일 | 생성 그래프 |
|---|---|
| eda_visualization.py | fig1~6 (EDA 6종) |
| ml_comparison_viz.py | ml1~3 (알고리즘 비교 3종) |
| eda_overview_slide.py | 데이터 개요 슬라이드 |
| slide_model.py | 모델 프로세스·결과 슬라이드 2종 |

**그래프이미지/ (PNG 12개):**
| 파일 | 내용 |
|---|---|
| eda_overview_slide.png | 데이터 전체 개요 (정보요약+분포+기술통계) |
| fig1_sleep_effect.png | 수면박탈 효과 (주관 vs 객관) |
| fig2_pvt_duration.png | 측정시간별 성능 (3분 최적) |
| fig3_drozy_kss_pvt.png | DROZY: KSS-PVT 동반 악화 |
| fig4_2x2_matrix.png | 2×2 상태 분포 |
| fig5_hidden_risk.png | 숨은위험 정체 검증 |
| fig6_vif.png | VIF 변수선택 (13→5) |
| ml1_algorithm_comparison.png | 알고리즘 7종 비교 |
| ml2_improvement.png | 성능 개선 과정 |
| ml3_feature_input.png | 변수 중요도 + 입력 조합 |
| slide_model_process.png | 모델 STEP 프로세스 슬라이드 |
| slide_model_result.png | 모델 성능 결과 슬라이드 |

시각화_안내.md — 각 그래프가 어느 발표 항목에 쓰이는지 정리

---

## 발표 시 빠른 참조

- **핵심 수치:** 숨은위험 27.8% / AUC 0.806 / 측정시간 3분 / KSS-PVT r=0.63
- **발표 파일:** 5_발표문서/발표_PPT.pptx (13장)
- **상세 원고:** 5_발표문서/세미1_발표_상세본.md
- **질문 대비:** 4_리포트/ 각 문서에 근거·수치 정리

## 재현 방법

pip install pandas numpy scipy statsmodels scikit-learn xgboost lightgbm matplotlib
- python 1_분석코드/01_pvt_pipeline.py  (merged_full.csv 생성)
- python 1_분석코드/02_eda_analysis.py
- python 1_분석코드/03a_multiwindow_pvt.py
- python 1_분석코드/03b_model_comparison.py
- python 6_시각화/eda_visualization.py
- python 6_시각화/ml_comparison_viz.py

※ 중간 산출물 CSV가 포함되어 있어 대부분 바로 실행 가능
