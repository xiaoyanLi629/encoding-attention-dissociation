"""
Camera-ready: unimodal Ridge encoding evaluated on held-out data.

Same features, HRF window and alpha grid as src/train_unimodal_models_module.py, but
fit on Friends S1-S5 only and evaluated on Friends S6 and Movie10 (never seen in fitting).
Also reports the in-sample r that the submitted Table I used, for comparison.
"""
from pathlib import Path
import os, sys, json
import numpy as np
from sklearn.linear_model import RidgeCV

ROOT = str(Path(__file__).resolve().parents[1])
CR = os.path.join(ROOT, 'runs', 'camera_ready')
ALPHAS = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.001])
BLOCKS = {'visual': slice(0, 1250), 'audio': slice(1250, 1350), 'language': slice(1350, 1600)}


def colcorr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1), 0.0)


def main(subject):
    d = np.load(os.path.join(CR, 'cache', f'sub-0{subject}.npz'))
    Xtr, ytr, Xva, yva, n6 = d['Xtr'], d['ytr'], d['Xva'], d['yva'], int(d['n_friends_s6'])
    out = {}
    for m, sl in BLOCKS.items():
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr[:, sl], ytr)
        p_tr = model.predict(Xtr[:, sl]); p_va = model.predict(Xva[:, sl])
        out[m] = {
            'r_train_insample': colcorr(p_tr, ytr),
            'r_heldout_all': colcorr(p_va, yva),
            'r_heldout_friends_s6': colcorr(p_va[:n6], yva[:n6]),
            'r_heldout_movie10': colcorr(p_va[n6:], yva[n6:]),
        }
        print(subject, m, {k: round(float(v.mean()), 4) for k, v in out[m].items()}, flush=True)
    np.savez(os.path.join(CR, 'heldout', f'sub-0{subject}_unimodal_r.npz'),
             **{f'{m}__{k}': v for m, dd in out.items() for k, v in dd.items()})

    # split-half: fit on Friends S1-S2 vs S3-S5 halves is not recoverable from the cache
    # (season boundaries were concatenated), so split the training TRs into first/second half
    # (roughly S1-S3 vs S3-S5) and re-fit each half, evaluated on the same held-out set.
    half = len(Xtr) // 2
    sh = {}
    for h, idx in (('A', slice(0, half)), ('B', slice(half, None))):
        for m, sl in BLOCKS.items():
            model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr[idx, sl], ytr[idx])
            sh[f'{m}__{h}'] = colcorr(model.predict(Xva[:, sl]), yva)
    np.savez(os.path.join(CR, 'heldout', f'sub-0{subject}_splithalf_r.npz'), **sh)
    print(subject, 'split-half done', flush=True)


if __name__ == '__main__':
    os.makedirs(os.path.join(CR, 'heldout'), exist_ok=True)
    main(int(sys.argv[1]))
