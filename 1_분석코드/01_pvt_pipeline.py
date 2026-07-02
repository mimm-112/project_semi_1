"""ds000201 PVT 파이프라인 v2 — 5분 기준 정제 → 파생변수 → 위험도 분류"""
import pandas as pd, numpy as np, glob, re, os

FALSE_START_MS = 100
LAPSE_MS = 500
CUTOFF_S = 300  # 5분

def load_events(path):
    df = pd.read_csv(path, sep='\t', lineterminator='\n')
    df.columns = [c.strip() for c in df.columns]
    df['onset'] = pd.to_numeric(df['onset'].astype(str).str.replace('\r','').str.strip(), errors='coerce')
    df['rt_ms'] = pd.to_numeric(df['response_time'].astype(str).str.replace('\r','').str.strip(), errors='coerce')*1000
    return df.dropna(subset=['onset','rt_ms'])

def metrics_5min(df):
    """5분 절삭 후 표준 PVT 파생변수"""
    t0 = df['onset'].min()
    df = df[df['onset'] <= t0 + CUTOFF_S]
    rt = df['rt_ms']
    n_total = len(rt)
    fs = int((rt < FALSE_START_MS).sum())
    lap = int((rt >= LAPSE_MS).sum())
    valid = rt[(rt >= FALSE_START_MS) & (rt < LAPSE_MS)]
    nv = len(valid)
    speed = 1000.0/valid  # 1/s
    return {
        'n_trial': n_total, 'n_valid': nv,
        'false_starts': fs, 'lapses': lap,
        'lapse_rate': lap/n_total if n_total else np.nan,
        'mean_rt': valid.mean(), 'median_rt': valid.median(),
        'sd_rt': valid.std(),
        'min_rt': valid.min(), 'max_rt': valid.max(),
        'mean_speed': speed.mean(),
        'slowest10_rt': valid.quantile(0.90),
        'cv_rt': valid.std()/valid.mean() if valid.mean() else np.nan,  # 변동계수
    }

rows=[]
for f in sorted(glob.glob('pvt/*.tsv')):
    m = re.search(r'(sub-\d+)_(ses-\d+)', os.path.basename(f))
    d = metrics_5min(load_events(f))
    d['sub']=m.group(1); d['ses']=m.group(2); d['ses_num']=int(m.group(2)[-1])
    rows.append(d)
pvt = pd.DataFrame(rows)

# --- STEP 3: 위험/주의/양호 분류 ---
# PVT 문헌 기준: lapse 수와 mean_rt로 각성/피로 상태 판정
# 여기선 데이터 분포 기반 3분위 + 문헌 임계 혼합
def classify(row):
    # 위험 신호: lapse가 있거나 반응이 느림
    score = 0
    if row['lapses'] >= 2: score += 2
    elif row['lapses'] == 1: score += 1
    if row['mean_rt'] >= 300: score += 2
    elif row['mean_rt'] >= 270: score += 1
    if row['lapse_rate'] >= 0.05: score += 1
    if score >= 3: return '위험'
    elif score >= 1: return '주의'
    else: return '양호'
pvt['pvt_status'] = pvt.apply(classify, axis=1)

pvt.to_csv('pvt_5min.csv', index=False)
print("=== 5분 정제 후 PVT 파생변수 ===")
print("세션 수:", len(pvt), "| trial 평균:", round(pvt['n_trial'].mean(),1),
      "범위", pvt['n_trial'].min(), "~", pvt['n_trial'].max())
print()
print("=== 파생변수 목록 ===")
print([c for c in pvt.columns if c not in ['sub','ses','ses_num']])
print()
print("=== STEP 3: 위험도 분류 결과 ===")
print(pvt['pvt_status'].value_counts().to_string())

# --- STEP 4: 설문(participants.tsv) 전체 병합 ---
# participants.tsv가 같은 폴더에 있어야 함
try:
    p = pd.read_csv('participants.tsv', sep='\t').rename(columns={'participant_id': 'sub'})
    pvt = pvt.merge(p[['sub', 'Sl_cond']], on='sub', how='left')
    pvt['sleep_deprived'] = (pvt['ses_num'] == pvt['Sl_cond']).astype(int)
    merged = pvt.merge(p, on='sub', how='left', suffixes=('', '_p'))
    merged.to_csv('merged_full.csv', index=False)
    print(f"\n=== STEP 4: 설문 병합 완료 → merged_full.csv ({merged.shape[0]}행 × {merged.shape[1]}열) ===")
except FileNotFoundError:
    print("\n[주의] participants.tsv가 없어 병합을 건너뜀 (pvt_5min.csv만 생성)")
