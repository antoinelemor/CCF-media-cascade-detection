#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-media-cascade-detection (paper: CCF_paradigm, v3)

TITLE:
------
analysis_v3.py

MAIN OBJECTIVE:
---------------
Second-stage analyses for the v3 paper. All computations read the production
results (results/production) and quantify the MECHANISMS of paradigm
maintenance and change, beyond the descriptive statistics of v2.

  N1  Paradigm persistence and impulse responses (local projections):
      how long does a perturbation of each frame's dominance survive?
  N2  State-dependent permeability (windows of vulnerability): are event
      effects conditional on the concentration of the paradigm at arrival?
  N3  Anomaly accumulation: are clustered weather anomalies superadditive?
  N4  Event potency: what distinguishes the events that move the paradigm?
  N5  Cascade potency: which cascade properties predict paradigmatic impact?
  N6  Cascade succession: which frames' cascades beget which, versus a
      circular-shift null?

Dependencies:
-------------
- pandas, numpy, scipy, statsmodels (framework venv)

MAIN FEATURES:
--------------
Writes a digest to stdout and CSV side-products to ../tables/v3_*.csv.

Author:
-------
Antoine Lemor
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
R = ROOT / 'results' / 'production'
OUT = Path(__file__).resolve().parent.parent / 'tables'
FRAMES = ['Pol', 'Eco', 'Sci', 'Envt', 'Pbh', 'Just', 'Cult', 'Secu']

print('=== chargement ===')
tl = pd.concat([pd.read_parquet(p) for p in
                sorted(glob.glob(str(R / '*/paradigm_shifts/paradigm_timeline.parquet')))])
tl['date'] = pd.to_datetime(tl['date'])
tl = tl.drop_duplicates('date').sort_values('date').set_index('date')
tl = tl.asfreq('D')
dom = tl[[f'paradigm_{f}' for f in FRAMES]].interpolate(limit=7)
print(f'timeline: {len(tl)} jours, {tl.index.min():%Y-%m-%d} -> {tl.index.max():%Y-%m-%d}')

clusters = []
for p in sorted(glob.glob(str(R / '*/events/event_clusters.json'))):
    year = int(Path(p).parts[-3])
    with open(p) as f:
        arr = json.load(f)
    arr = arr if isinstance(arr, list) else list(arr.values())
    for c in arr:
        clusters.append({'year': year, 'cluster_id': c.get('cluster_id'),
                         'peak_date': c.get('peak_date'),
                         'dominant_type': c.get('dominant_type'),
                         'n_articles': c.get('n_articles'),
                         'total_mass': c.get('total_mass'),
                         'n_occurrences': c.get('n_occurrences'),
                         'is_multi_type': bool(c.get('is_multi_type')),
                         'n_entities': len(c.get('entities') or []),
                         'strength': c.get('strength')})
EC = pd.DataFrame(clusters)
EC['peak_date'] = pd.to_datetime(EC['peak_date'])
print(f'clusters: {len(EC):,}')

ma = pd.concat([pd.read_parquet(p).assign(year=int(Path(p).parts[-3]))
                for p in sorted(glob.glob(str(R / '*/impact_analysis/stabsel_cluster_dominance.parquet')))])
ma['peak_date'] = pd.to_datetime(ma['peak_date'])
mas = ma[ma.role.isin(['catalyst', 'disruptor'])].copy()
print(f'Model A: {len(ma):,} lignes, {len(mas):,} significatives')

mb = pd.concat([pd.read_parquet(p).assign(year=int(Path(p).parts[-3]))
                for p in sorted(glob.glob(str(R / '*/impact_analysis/stabsel_cascade_dominance.parquet')))])
print(f'Model B: {len(mb):,} lignes | colonnes: {list(mb.columns)[:12]}')

cascades = []
for p in sorted(glob.glob(str(R / '*/cascades.json'))):
    with open(p) as f:
        for c in json.load(f):
            dm = c.get('dominant_messengers') or {}
            tot = sum(dm.values()) or 1
            cascades.append({
                'cascade_id': c['cascade_id'], 'frame': c['frame'],
                'onset': pd.to_datetime(c.get('onset_date')),
                'end': pd.to_datetime(c.get('end_date')),
                'peak': pd.to_datetime(c.get('peak_date')),
                'n_articles': c.get('n_articles'), 'n_media': c.get('n_media'),
                'duration': c.get('duration_days'),
                'classification': c.get('classification'),
                'semantic_similarity': c.get('semantic_similarity'),
                'network_modularity': c.get('network_modularity'),
                'network_density': c.get('network_density'),
                'messenger_concentration': c.get('messenger_concentration'),
                'novelty_decay': c.get('novelty_decay_rate'),
                'cross_media_alignment': c.get('cross_media_alignment'),
                'sh_scientist': dm.get('msg_scientist', 0) / tot,
                'sh_official': dm.get('msg_official', 0) / tot,
                'sh_activist': dm.get('msg_activist', 0) / tot,
            })
CA = pd.DataFrame(cascades)
CA = CA[CA.classification != 'not_cascade']
print(f'cascades: {len(CA)}')

era = lambda d: 'emergence' if d.year < 1992 else ('contestation' if d.year < 2010 else 'lockin')

# ---------------------------------------------------------------------------
print('\n=== N1. persistance et réponses impulsionnelles ===')
# persistance AR(1) sur l'indice de dominance quotidien (écart à une tendance lente)
half = {}
for f in FRAMES:
    y = dom[f'paradigm_{f}'].dropna()
    dev = y - y.rolling(365, min_periods=180, center=True).mean()
    dev = dev.dropna()
    rho = dev.autocorr(1)
    hl = np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.nan
    half[f] = (rho, hl)
    print(f'  {f}: rho1={rho:.3f}, demi-vie={hl:.0f} j')
pd.DataFrame(half, index=['rho', 'half_life_days']).T.to_csv(OUT / 'v3_half_life.csv')

# masses quotidiennes d'évènements par type
mass = {}
for t in ['evt_weather', 'evt_election', 'evt_publication', 'evt_meeting', 'evt_policy', 'evt_judiciary']:
    s = EC[EC.dominant_type == t].groupby('peak_date')['n_articles'].sum()
    mass[t] = s.reindex(dom.index, fill_value=0.0)
MASS = pd.DataFrame(mass)

def local_projection(target, shock, horizons=(0, 3, 7, 14, 21, 30), controls_ar=5, cond=None):
    """dom_{t+h} - dom_{t-1} ~ shock_t (Newey-West), AR controls, option masque."""
    y = dom[f'paradigm_{target}']
    x = MASS[shock] / MASS[shock].std()
    rows = []
    for h in horizons:
        df = pd.DataFrame({'dy': y.shift(-h) - y.shift(1), 'x': x})
        for l in range(1, controls_ar + 1):
            df[f'ar{l}'] = y.shift(l) - y.shift(l + 1)
        if cond is not None:
            df = df[cond.reindex(df.index).fillna(False)]
        df = df.dropna()
        X = sm.add_constant(df[['x'] + [f'ar{l}' for l in range(1, controls_ar + 1)]])
        m = sm.OLS(df['dy'], X).fit(cov_type='HAC', cov_kwds={'maxlags': h + 7})
        rows.append({'h': h, 'beta': m.params['x'] * 100, 'se': m.bse['x'] * 100,
                     'p': m.pvalues['x'], 'n': int(m.nobs)})
    return pd.DataFrame(rows)

irf_out = []
for shock, tgt in [('evt_weather', 'Pol'), ('evt_weather', 'Envt'), ('evt_weather', 'Secu'),
                   ('evt_weather', 'Pbh'), ('evt_election', 'Pol'), ('evt_election', 'Secu'),
                   ('evt_publication', 'Sci'), ('evt_publication', 'Pol'),
                   ('evt_meeting', 'Pol'), ('evt_meeting', 'Envt')]:
    r = local_projection(tgt, shock)
    r['shock'], r['target'] = shock, tgt
    irf_out.append(r)
    line = ' '.join(f"h{int(x.h)}:{x.beta:+.2f}{'*' if x.p < 0.05 else ''}" for x in r.itertuples())
    print(f'  {shock}->{tgt}: {line}')
IRF = pd.concat(irf_out)
IRF.to_csv(OUT / 'v3_irf.csv', index=False)

# ---------------------------------------------------------------------------
print('\n=== N2. perméabilité dépendante de l\'état (fenêtres) ===')
conc = tl['concentration'].interpolate(limit=7)
pre_state = conc.rolling(14).mean().shift(1)
mas['pre_conc'] = mas.peak_date.map(pre_state)
mas['era'] = mas.peak_date.map(era)
msig = mas.dropna(subset=['pre_conc']).copy()
msig['absb'] = msig.net_beta.abs()
terc = msig.pre_conc.quantile([1 / 3, 2 / 3]).values
msig['state'] = np.where(msig.pre_conc <= terc[0], 'faible',
                         np.where(msig.pre_conc <= terc[1], 'moyenne', 'forte'))
print('  concentration pré-évènement -> effets significatifs :')
g = msig.groupby('state').agg(n=('absb', 'size'), med_absb=('absb', 'median'),
                              pct_disrupt=('role', lambda r: (r == 'disruptor').mean() * 100))
print(g.round(3))
# lien continu, contrôles type+masse (clusters joints par année+id)
msig2 = msig.merge(EC[['year', 'cluster_id', 'n_articles']],
                   on=['year', 'cluster_id'], how='left')
msig2 = msig2.dropna(subset=['n_articles'])
X = pd.get_dummies(msig2['dominant_type'], drop_first=True).astype(float)
X['log_mass'] = np.log1p(msig2.n_articles.values)
X['pre_conc'] = msig2.pre_conc.values
X = sm.add_constant(X)
mreg = sm.OLS(np.log(msig2.absb.values + 1e-9), X).fit(cov_type='HC1')
print(f"  log|beta| ~ pre_conc : coef={mreg.params['pre_conc']:+.2f}, p={mreg.pvalues['pre_conc']:.1e}, n={int(mreg.nobs)}")
ld = sm.Logit((msig2.role == 'disruptor').astype(float).values, X).fit(disp=0)
print(f"  P(disruptor) ~ pre_conc : coef={ld.params['pre_conc']:+.2f}, p={ld.pvalues['pre_conc']:.1e}")
g.to_csv(OUT / 'v3_state_dependence.csv')

# quart le plus concentré vs le moins : ratio des effets
lo, hi = g.loc['faible', 'med_absb'], g.loc['forte', 'med_absb']
print(f'  ratio médian faible/forte concentration : {lo / hi:.2f}x')

# ---------------------------------------------------------------------------
print('\n=== N3. accumulation d\'anomalies (superadditivité) ===')
w = MASS['evt_weather']
trail = w.rolling(30).sum().shift(1)          # pression des 30 j précédents
med = trail[trail > 0].median()
cond_hi = trail > med
cond_lo = (trail <= med)
for tgt in ['Envt', 'Secu', 'Pol']:
    r_hi = local_projection(tgt, 'evt_weather', horizons=(7, 14), cond=cond_hi)
    r_lo = local_projection(tgt, 'evt_weather', horizons=(7, 14), cond=cond_lo)
    for h in (7, 14):
        bh = r_hi[r_hi.h == h].iloc[0]
        bl = r_lo[r_lo.h == h].iloc[0]
        print(f'  météo->{tgt} h={h} : pression FORTE {bh.beta:+.2f} (p={bh.p:.2g}) '
              f'vs FAIBLE {bl.beta:+.2f} (p={bl.p:.2g})')

# ---------------------------------------------------------------------------
print('\n=== N4. puissance des évènements ===')
pot = mas.merge(EC[['year', 'cluster_id', 'n_articles', 'n_occurrences', 'n_entities',
                    'is_multi_type', 'strength']].rename(columns={'strength': 'cl_strength'}),
                on=['year', 'cluster_id'], how='left').dropna(subset=['n_articles'])
pot['absb'] = pot.net_beta.abs()
X = pd.get_dummies(pot['dominant_type'], drop_first=True).astype(float)
X['log_mass'] = np.log1p(pot.n_articles.values)
X['multi_type'] = pot.is_multi_type.astype(float).values
X['n_entities'] = np.log1p(pot.n_entities.values)
X['strength'] = pot.cl_strength.values
X['lockin'] = (pot.era == 'lockin').astype(float).values
X = sm.add_constant(X)
mp = sm.OLS(np.log(pot.absb.values + 1e-9), X).fit(cov_type='HC1')
print(mp.summary2().tables[1].round(3).to_string())
pd.DataFrame({'coef': mp.params, 'se': mp.bse, 'p': mp.pvalues,
              'n': int(mp.nobs)}).to_csv(OUT / 'v3_event_potency.csv')

# ---------------------------------------------------------------------------
print('\n=== N5. puissance des cascades ===')
colb = 'role' if 'role' in mb.columns else 'classification'
mbs = mb[mb[colb].isin(['catalyst', 'disruptor', 'auto_amplification', 'auto_suppression'])].copy()
imp = mbs.groupby('cascade_id').net_beta.apply(lambda s: s.abs().mean()).rename('absb')
cp = CA.merge(imp, on='cascade_id', how='inner').dropna(
    subset=['semantic_similarity', 'network_modularity', 'sh_scientist'])
X = pd.get_dummies(cp['frame'], drop_first=True).astype(float)
for c in ['sh_scientist', 'sh_official', 'sh_activist', 'semantic_similarity',
          'network_modularity', 'messenger_concentration']:
    X[c] = cp[c].values
X['log_size'] = np.log1p(cp.n_articles.values)
X['log_dur'] = np.log1p(cp.duration.values)
X = sm.add_constant(X)
mc = sm.OLS(np.log(cp.absb.values + 1e-9), X).fit(cov_type='HC1')
print(mc.summary2().tables[1].round(3).to_string())
pd.DataFrame({'coef': mc.params, 'se': mc.bse, 'p': mc.pvalues,
              'n': int(mc.nobs)}).to_csv(OUT / 'v3_cascade_potency.csv')

# ---------------------------------------------------------------------------
print('\n=== N6. succession cascade -> cascade ===')
CA2 = CA.dropna(subset=['onset', 'end']).sort_values('onset').reset_index(drop=True)
def successions(df, w=21):
    out = np.zeros((len(FRAMES), len(FRAMES)))
    idx = {f: i for i, f in enumerate(FRAMES)}
    ends = df['end'].values; onsets = df['onset'].values
    for i in range(len(df)):
        dt = (onsets - ends[i]).astype('timedelta64[D]').astype(int)
        for j in np.where((dt > 0) & (dt <= w))[0]:
            out[idx[df.frame.iat[i]], idx[df.frame.iat[j]]] += 1
    return out
obs = successions(CA2)
rng = np.random.default_rng(42)
null = np.zeros((1000, len(FRAMES), len(FRAMES)))
for b in range(1000):
    sh = CA2.copy()
    sh['frame'] = rng.permutation(sh['frame'].values)
    null[b] = successions(sh)
z = (obs - null.mean(0)) / (null.std(0) + 1e-9)
print('  motifs z>2 :')
for i, fi in enumerate(FRAMES):
    for j, fj in enumerate(FRAMES):
        if z[i, j] > 2 and obs[i, j] >= 5:
            print(f'    {fi} -> {fj} : {int(obs[i, j])} (attendu {null.mean(0)[i, j]:.1f}, z={z[i, j]:.1f})')
pd.DataFrame(z, index=FRAMES, columns=FRAMES).round(2).to_csv(OUT / 'v3_succession_z.csv')
pd.DataFrame(obs, index=FRAMES, columns=FRAMES).astype(int).to_csv(OUT / 'v3_succession_obs.csv')

print('\n=== périodisation ===')
CA['yr'] = CA.peak.dt.year
last_pol = CA[CA.frame == 'Pol'].yr.max()
print(f'  dernière cascade Politique : {last_pol}')
dyr = tl['dominant_frames'].astype(str)
share = ((dyr.str.contains('Pol') | dyr.str.contains('Eco')).groupby(tl.index.year).mean() * 100)
print('  Pol∪Eco % jours par période :',
      {p: round(share.loc[a:b].mean(), 1) for p, (a, b) in
       {'<1992': (1978, 1991), '1992-2009': (1992, 2009), '>=2010': (2010, 2024)}.items()})
print('\nFini.')