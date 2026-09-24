"""
Camera-ready: network-level statistics recomputed from held-out unimodal encoding.
Uses the official Schaefer-1000 7-network labels (resources/schaefer1000_7net.txt).
"""
from pathlib import Path
import os, json
import numpy as np
from scipy import stats

ROOT = str(Path(__file__).resolve().parents[1])
CR = os.path.join(ROOT, 'runs', 'camera_ready')
SUBS = [1, 2, 3, 5]
MODS = ['visual', 'audio', 'language']
NETS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

lab = [l.split('\t')[1].split('_')[2] for l in open(os.path.join(ROOT, 'resources', 'schaefer1000_7net.txt')).read().strip().split('\n')]
lab = np.array(lab); assert len(lab) == 1000


def icc11(Y):
    """ICC(1,1), Shrout & Fleiss. Y: targets x raters."""
    n, k = Y.shape
    gm = Y.mean()
    msb = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((Y - Y.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def analyse(key):
    R = {s: np.stack([np.load(os.path.join(CR, 'heldout', f'sub-0{s}_unimodal_r.npz'))[f'{m}__{key}']
                      for m in MODS], 1) for s in SUBS}          # 1000 x 3
    res = {'key': key}
    res['mean_r'] = {f'sub-0{s}': R[s].mean(0).round(4).tolist() for s in SUBS}
    g = np.mean([R[s].mean(0) for s in SUBS], 0)
    res['group_mean_r'] = g.round(4).tolist()
    res['pct_dominant'] = {f'sub-0{s}': (np.bincount(R[s].argmax(1), minlength=3) / 10).tolist() for s in SUBS}
    Rg = np.mean([R[s] for s in SUBS], 0)
    res['pct_dominant_groupmap'] = (np.bincount(Rg.argmax(1), minlength=3) / 10).tolist()
    res['eff_alpha_group'] = (g / g.sum()).round(4).tolist()
    res['eff_alpha_subject'] = {f'sub-0{s}': (R[s].mean(0) / R[s].mean(0).sum()).round(4).tolist() for s in SUBS}
    # network means, MSI, MII
    netr = {s: np.array([R[s][lab == n].mean(0) for n in NETS]) for s in SUBS}   # 7x3
    res['net_mean_r_group'] = {n: np.mean([netr[s][i] for s in SUBS], 0).round(4).tolist() for i, n in enumerate(NETS)}
    mii = np.array([[1 - netr[s][i].std() / netr[s][i].mean() for s in SUBS] for i in range(7)])  # 7x4
    res['MII_net_group'] = dict(zip(NETS, mii.mean(1).round(3).tolist()))
    res['MII_net_sd'] = dict(zip(NETS, mii.std(1, ddof=1).round(3).tolist()))
    res['MII_ICC11'] = round(float(icc11(mii)), 3)

    def msi(r):
        rs = np.sort(np.clip(r, 0, None), 1)
        mx, oth = rs[:, 2], rs[:, :2].mean(1)
        return np.where(mx + oth > 0, (mx - oth) / (mx + oth + 1e-12), 0)
    res['MSI_net_group'] = {n: round(float(np.mean([msi(R[s])[lab == n].mean() for s in SUBS])), 3) for n in NETS}
    # ANOVA per network (parcels x modalities, group map), Bonferroni over 7 nets x 3 pairs
    an = {}
    for n in NETS:
        x = Rg[lab == n]
        F, p = stats.f_oneway(*x.T)
        an[n] = {'F': round(float(F), 2), 'p': float(p)}
    res['anova_net'] = an
    # similarity between parcel encoding profiles: within vs between network
    Z = Rg - Rg.mean(1, keepdims=True)
    C = np.corrcoef(Z)
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(1000, 1)
    w, b = C[iu][same[iu]], C[iu][~same[iu]]
    res['profile_sim_within_between'] = [round(float(w.mean()), 3), round(float(b.mean()), 3)]
    return res, R


out = {}
for key in ['r_train_insample', 'r_heldout_all', 'r_heldout_friends_s6', 'r_heldout_movie10']:
    out[key], R = analyse(key)

# split-half (first vs second half of Friends S1-S5 training TRs, same held-out evaluation)
sh = {}
for s in SUBS:
    d = np.load(os.path.join(CR, 'heldout', f'sub-0{s}_splithalf_r.npz'))
    A = np.stack([d[f'{m}__A'] for m in MODS], 1); B = np.stack([d[f'{m}__B'] for m in MODS], 1)
    sh[f'sub-0{s}'] = {
        'mean_r_A': A.mean(0).round(4).tolist(), 'mean_r_B': B.mean(0).round(4).tolist(),
        'eff_alpha_A': (A.mean(0) / A.mean(0).sum()).round(4).tolist(),
        'eff_alpha_B': (B.mean(0) / B.mean(0).sum()).round(4).tolist(),
        'parcel_r_corr_AB': [round(float(np.corrcoef(A[:, j], B[:, j])[0, 1]), 3) for j in range(3)],
        'pct_vis_dom_A': float((A.argmax(1) == 0).mean() * 100), 'pct_vis_dom_B': float((B.argmax(1) == 0).mean() * 100),
    }
out['split_half'] = sh
json.dump(out, open(os.path.join(CR, 'network_stats.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
