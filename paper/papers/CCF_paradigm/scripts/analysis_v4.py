#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection (paper: CCF_paradigm, v4)

TITLE:
------
analysis_v4.py

MAIN OBJECTIVE:
---------------
Analyses for the two-regime theory of discursive competition (v4).

  T1  Transition anatomy: structural breaks on the Politics-Science dominance
      gap (PELT, ruptures), cascade production by frame and year, and the
      lead-lag between dominant-pair cascade activity and dominance.
  T2  Saturation-decoupling: dominant-pair cascade production as a function
      of joint dominance saturation; asymmetry with Science.
  T3  Regime-resolved mechanisms: half-lives of dominance deviations by era
      (moving-block bootstrap CIs) and impulse responses by era.
  T4  Era-resolved first order: displacement and driver ratio by era
      (already computed by year; aggregated here with Wilson CIs).

Reads results/production and writes tables/v4_*.csv plus a stdout digest.

Author:
-------
Antoine Lemor
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
R = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'
FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']
ERAS = [('contest', 1978, 1995), ('consolidation', 1996, 2009), ('lockin', 2010, 2024)]


def era_of(y):
    for name, a, b in ERAS:
        if a <= y <= b:
            return name
    return None


print('=== chargement ===')
tl = pd.concat([pd.read_parquet(p) for p in
                sorted(glob.glob(str(R / '*/paradigm_shifts/paradigm_timeline.parquet')))])
tl['date'] = pd.to_datetime(tl['date'])
tl = tl.drop_duplicates('date').sort_values('date').set_index('date').asfreq('D')
dom = tl[[f'paradigm_{f}' for f in FRAMES]].interpolate(limit=7)

cascades = []
for p in sorted(glob.glob(str(R / '*/cascades.json'))):
    with open(p) as f:
        for c in json.load(f):
            if c.get('classification') == 'not_cascade':
                continue
            cascades.append({'frame': c['frame'],
                             'peak': pd.to_datetime(c.get('peak_date')),
                             'onset': pd.to_datetime(c.get('onset_date')),
                             'n_articles': c.get('n_articles'),
                             'duration': c.get('duration_days')})
CA = pd.DataFrame(cascades)
CA['yr'] = CA.peak.dt.year

clusters = []
for p in sorted(glob.glob(str(R / '*/events/event_clusters.json'))):
    with open(p) as f:
        arr = json.load(f)
    for c in (arr if isinstance(arr, list) else arr.values()):
        clusters.append({'peak_date': pd.to_datetime(c.get('peak_date')),
                         'dominant_type': c.get('dominant_type'),
                         'n_articles': c.get('n_articles')})
EC = pd.DataFrame(clusters)

cc = pd.concat([pd.read_parquet(p).assign(year=int(Path(p).parts[-3]))
                for p in sorted(glob.glob(str(R / '*/impact_analysis/cluster_cascade.parquet')))])
sig = cc[cc.role.isin(['driver', 'suppressor'])].copy()

# ---------------------------------------------------------------------------
print('\n=== T1. anatomie de la transition ===')
gap = (dom['paradigm_Pol'] - dom['paradigm_Sci']).dropna()
gm = gap.resample('MS').mean().dropna()
algo = rpt.Pelt(model='rbf', min_size=24).fit(gm.values.reshape(-1, 1))
for pen in (8, 12, 20):
    bk = algo.predict(pen=pen)
    dates = [gm.index[i - 1].strftime('%Y-%m') for i in bk[:-1]]
    print(f'  ruptures PELT (pen={pen}) sur écart Pol-Sci mensuel : {dates}')

# production de cascades par cadre et année
prod = CA.pivot_table(index='yr', columns='frame', values='peak', aggfunc='count').fillna(0)
prod.to_csv(OUT / 'v4_cascade_production.csv')
pair = prod.get('Pol', 0) + prod.get('Eco', 0)
print('  cascades Pol+Eco par période :',
      {f'{a}-{b}': int(pair.loc[a:b].sum()) for _, a, b in
       [('x', 1978, 1987), ('x', 1988, 1995), ('x', 1996, 2005), ('x', 2006, 2024)]})
print('  cascades Sci par période :',
      {f'{a}-{b}': int(prod.get("Sci", pd.Series(0, index=prod.index)).loc[a:b].sum()) for _, a, b in
       [('x', 1978, 1987), ('x', 1988, 1995), ('x', 1996, 2005), ('x', 2006, 2024)]})

# lead-lag : part trimestrielle d'articles en cascade Pol+Eco vs dominance conjointe
CA['q'] = CA.peak.dt.to_period('Q')
qs = CA.groupby(['q', 'frame']).n_articles.sum().unstack().fillna(0)
pair_share = ((qs.get('Pol', 0) + qs.get('Eco', 0)) / qs.sum(1).replace(0, np.nan))
domq = ((dom['paradigm_Pol'] + dom['paradigm_Eco']) / 2).resample('QS').mean()
domq.index = domq.index.to_period('Q')
window = (pair_share.index >= '1986Q1') & (pair_share.index <= '2000Q4')
a = pair_share[window].fillna(0)
b = domq.reindex(a.index)
best = None
for lag in range(-8, 9):
    r = a.corr(b.shift(-lag))
    if best is None or (r or -9) > best[1]:
        best = (lag, r)
    if lag in (-4, -2, 0, 2, 4):
        print(f'  corr(part cascades Pol+Eco_t, dominance_(t+{lag}q)) = {r:+.2f}')
print(f'  meilleur décalage : dominance suit de {best[0]} trimestre(s) (r={best[1]:+.2f})')

# ---------------------------------------------------------------------------
print('\n=== T2. découplage par saturation ===')
domy = ((dom['paradigm_Pol'] + dom['paradigm_Eco']) / 2).resample('YS').mean()
domy.index = domy.index.year
dfc = pd.DataFrame({'pair_casc': pair.reindex(domy.index, fill_value=0),
                    'sat': domy}).dropna()
X = sm.add_constant(dfc[['sat']])
m = sm.OLS(dfc.pair_casc, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
print(f"  cascades Pol+Eco/an ~ dominance conjointe : coef={m.params['sat']:+.1f}, p={m.pvalues['sat']:.3f}")
# le lien production-dominance est en U INVERSÉ : les cascades accompagnent
# l'ASCENSION, pas le niveau — production maximale à dominance intermédiaire,
# extinction à saturation.
dfc['sat2'] = dfc.sat ** 2
mq = sm.OLS(dfc.pair_casc, sm.add_constant(dfc[['sat', 'sat2']])).fit(
    cov_type='HAC', cov_kwds={'maxlags': 3})
peak_sat = -mq.params['sat'] / (2 * mq.params['sat2']) if mq.params['sat2'] < 0 else np.nan
print(f"  U inversé : sat {mq.params['sat']:+.0f} (p={mq.pvalues['sat']:.3f}), "
      f"sat² {mq.params['sat2']:+.0f} (p={mq.pvalues['sat2']:.3f}), "
      f"pic à dominance ≈ {peak_sat:.2f}")
# production ~ croissance de la dominance (Δ annuel)
dfc['dsat'] = dfc.sat.diff()
mg = sm.OLS(dfc.pair_casc.iloc[1:], sm.add_constant(dfc[['dsat']].iloc[1:])).fit(
    cov_type='HAC', cov_kwds={'maxlags': 3})
print(f"  cascades Pol+Eco/an ~ Δdominance : coef={mg.params['dsat']:+.0f}, p={mg.pvalues['dsat']:.3f}")
# asymétrie Science : cascades sans dominance
sd = pd.DataFrame({'sci_casc': prod.get('Sci', pd.Series(0, index=prod.index)),
                   'sci_dom': dom['paradigm_Sci'].resample('YS').mean().set_axis(
                       dom['paradigm_Sci'].resample('YS').mean().index.year)}).dropna()
m2 = sm.OLS(sd.sci_casc, sm.add_constant(sd[['sci_dom']])).fit(
    cov_type='HAC', cov_kwds={'maxlags': 3})
print(f"  cascades Sci/an ~ dominance Sci : coef={m2.params['sci_dom']:+.1f}, p={m2.pvalues['sci_dom']:.3f}")
dfc.to_csv(OUT / 'v4_decoupling.csv')

# ---------------------------------------------------------------------------
print('\n=== T3. mécanismes par régime ===')
rng = np.random.default_rng(7)

def half_life(series):
    rho = series.autocorr(1)
    return np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.nan

rows = []
for name, a, b in ERAS:
    for f in ['Pol', 'Eco', 'Sci', 'Envt']:
        y = dom[f'paradigm_{f}'].loc[f'{a}':f'{b}'].dropna()
        dev = (y - y.rolling(365, min_periods=180, center=True).mean()).dropna()
        hl = half_life(dev)
        # bootstrap par blocs mobiles (60 j)
        boots = []
        n = len(dev)
        for _ in range(400):
            idx = rng.integers(0, n - 60, size=n // 60 + 1)
            sample = np.concatenate([dev.values[i:i + 60] for i in idx])[:n]
            s = pd.Series(sample)
            boots.append(half_life(s))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append({'era': name, 'frame': f, 'half_life': hl, 'lo': lo, 'hi': hi})
        print(f'  {name:13s} {f}: demi-vie {hl:.1f} j [{lo:.1f}, {hi:.1f}]')
HL = pd.DataFrame(rows)
HL.to_csv(OUT / 'v4_half_life_era.csv', index=False)

# IRF par ère (météo->Pol ; publications->Sci)
mass = {}
for t in ['evt_weather', 'evt_publication']:
    s = EC[EC.dominant_type == t].groupby('peak_date')['n_articles'].sum()
    mass[t] = s.reindex(dom.index, fill_value=0.0)
MASS = pd.DataFrame(mass)

def lp_era(target, shock, era_range, horizons=(3, 7, 14, 21)):
    a, b = era_range
    y = dom[f'paradigm_{target}'].loc[f'{a}':f'{b}']
    x = (MASS[shock] / MASS[shock].std()).loc[f'{a}':f'{b}']
    out = []
    for h in horizons:
        df = pd.DataFrame({'dy': y.shift(-h) - y.shift(1), 'x': x})
        for l in range(1, 6):
            df[f'ar{l}'] = y.shift(l) - y.shift(l + 1)
        df = df.dropna()
        X = sm.add_constant(df[['x'] + [f'ar{l}' for l in range(1, 6)]])
        m = sm.OLS(df.dy, X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 7})
        out.append({'h': h, 'beta': m.params['x'] * 100, 'se': m.bse['x'] * 100,
                    'p': m.pvalues['x']})
    return pd.DataFrame(out)

irf_rows = []
for shock, tgt in [('evt_weather', 'Pol'), ('evt_publication', 'Sci')]:
    for name, a, b in ERAS:
        r = lp_era(tgt, shock, (a, b))
        r['shock'], r['target'], r['era'] = shock, tgt, name
        irf_rows.append(r)
        peak = r.loc[r.beta.abs().idxmax()]
        print(f'  {shock}->{tgt} [{name}] : pic h{int(peak.h)} {peak.beta:+.2f} (p={peak.p:.2g})')
IRFE = pd.concat(irf_rows)
IRFE.to_csv(OUT / 'v4_irf_era.csv', index=False)

# ---------------------------------------------------------------------------
print('\n=== T4. premier ordre par ère ===')
NAT = {'evt_weather': 'Envt', 'evt_election': 'Pol', 'evt_meeting': 'Pol',
       'evt_policy': 'Pol', 'evt_protest': 'Pol', 'evt_publication': 'Sci',
       'evt_judiciary': 'Just', 'evt_cultural': 'Cult'}
sig['era'] = sig.year.map(era_of)
sig['displaced'] = sig.cascade_frame != sig.dominant_type.map(NAT)
g = sig.groupby('era').agg(n=('role', 'size'),
                           driver=('role', lambda r: (r == 'driver').mean() * 100),
                           displaced=('displaced', lambda d: d.mean() * 100))
g = g.reindex([e[0] for e in ERAS])
print(g.round(1))
g.to_csv(OUT / 'v4_first_order_era.csv')

print('\nFini.')
