"""PVT 측정시간별(60/90/180/300초) 파생변수 재계산"""
import pandas as pd, numpy as np, glob, re, os

FALSE_START_MS=100; LAPSE_MS=500
WINDOWS={'60초':60,'90초':90,'3분':180,'5분':300}

def load_events(path):
    df=pd.read_csv(path,sep='\t',lineterminator='\n')
    df.columns=[c.strip() for c in df.columns]
    df['onset']=pd.to_numeric(df['onset'].astype(str).str.replace('\r','').str.strip(),errors='coerce')
    df['rt_ms']=pd.to_numeric(df['response_time'].astype(str).str.replace('\r','').str.strip(),errors='coerce')*1000
    return df.dropna(subset=['onset','rt_ms'])

def metrics(df,cutoff):
    t0=df['onset'].min()
    df=df[df['onset']<=t0+cutoff]
    rt=df['rt_ms']; n=len(rt)
    fs=int((rt<FALSE_START_MS).sum()); lap=int((rt>=LAPSE_MS).sum())
    valid=rt[(rt>=FALSE_START_MS)&(rt<LAPSE_MS)]
    if len(valid)<3: return None
    speed=1000.0/valid
    return {'n_trial':n,'n_valid':len(valid),'false_starts':fs,'lapses':lap,
        'lapse_rate':lap/n if n else np.nan,'mean_rt':valid.mean(),
        'median_rt':valid.median(),'sd_rt':valid.std(),
        'mean_speed':speed.mean(),'slowest10_rt':valid.quantile(0.90)}

for wname,cutoff in WINDOWS.items():
    rows=[]
    for f in sorted(glob.glob('pvt/*.tsv')):
        m=re.search(r'(sub-\d+)_(ses-\d+)',os.path.basename(f))
        met=metrics(load_events(f),cutoff)
        if met is None: continue
        met['sub']=m.group(1); met['ses_num']=int(m.group(2)[-1])
        rows.append(met)
    d=pd.DataFrame(rows)
    d.to_csv(f'pvt_{cutoff}s.csv',index=False)
    print(f"{wname}: {len(d)}세션, trial 평균 {d['n_trial'].mean():.1f} (범위 {d['n_trial'].min()}~{d['n_trial'].max()}), lapse평균 {d['lapses'].mean():.2f}")
