# Which matrix is Fig. 2d, and where do "within 0.72 vs between 0.34, t = 15.6" come from?
from pathlib import Path
import numpy as np, h5py, sys, glob, json
from scipy import stats
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{ROOT}/src')
from brain_region_mapping import SCHAEFER_7_NETWORKS as NET
subs, mods = ['01', '02', '03', '05'], ['visual', 'audio', 'language']
order = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 'Limbic', 'Frontoparietal', 'Default']
lab_apx = np.full(1000, -1)
for i, k in enumerate(order):
    lab_apx[NET[k]['left_hemisphere'] + NET[k]['right_hemisphere']] = i
off = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
lab_off = np.array([off.index(l.split('\t')[1].split('_')[2]) for l in open(f'{ROOT}/resources/schaefer1000_7net.txt').read().strip().split('\n')])

def summarize(C, lab, name):
    m = lab >= 0; C = C[np.ix_(m, m)]; lab = lab[m]
    iu = np.triu_indices(len(lab), 1); same = (lab[:, None] == lab[None, :])[iu]; v = C[iu]
    N = np.array([[C[np.ix_(lab == a, lab == b)][~np.eye((lab == a).sum(), dtype=bool)].mean() if a == b else C[np.ix_(lab == a, lab == b)].mean() for b in range(7)] for a in range(7)])
    d, o = np.diag(N), N[~np.eye(7, dtype=bool)][:21] if False else N[np.triu_indices(7, 1)]
    t, p = stats.ttest_ind(d, o)
    tp, pp = stats.ttest_ind(v[same], v[~same], equal_var=False)
    print(f'{name}: parcel within {v[same].mean():.3f} between {v[~same].mean():.3f} (Welch t={tp:.1f}) | network-level diag {d.mean():.3f} offdiag {o.mean():.3f} t({len(d)+len(o)-2})={t:.2f} p={p:.2g}')
    return dict(parcel_within=float(v[same].mean()), parcel_between=float(v[~same].mean()), parcel_t=float(tp),
                net_within=float(d.mean()), net_between=float(o.mean()), net_t=float(t), net_p=float(p), df=int(len(d) + len(o) - 2))

out = {}
# 1) encoding-profile similarity (in-sample Ridge r, as Table I), centred profiles across the 3 modalities
R = np.mean([np.stack([np.load(f'{ROOT}/runs/run_revised/unimodal_models/ridge_model_sub-{s}_modality-{m}.npy', allow_pickle=True).item()['correlations'] for m in mods], 1) for s in subs], 0)
E = np.corrcoef(R)                      # Pearson across (rV, rA, rL) per parcel pair
for lab, ln in [(lab_apx, 'approx'), (lab_off, 'official')]:
    out[f'encoding_profile_{ln}'] = summarize(E, lab, f'encoding-profile sim [{ln}]')

# 2) BOLD functional connectivity, Fisher-z averaged over every run of every subject
n_runs, Z = 0, np.zeros((1000, 1000))
for s in subs:
    for fn in sorted(glob.glob(f'{ROOT}/data/fmri/sub-{s}/func/*.h5')):
        with h5py.File(fn, 'r') as f:
            for k in f.keys():
                y = f[k][:]
                c = np.corrcoef(y.T); np.fill_diagonal(c, 0)
                Z += np.arctanh(np.clip(c, -0.999, 0.999)); n_runs += 1
FC = np.tanh(Z / n_runs); np.fill_diagonal(FC, 1)
print('runs used for FC:', n_runs, ' FC value range', round(float(FC[np.triu_indices(1000,1)].min()),2), round(float(FC[np.triu_indices(1000,1)].max()),2))
out['n_runs'] = n_runs
for lab, ln in [(lab_apx, 'approx'), (lab_off, 'official')]:
    out[f'fc_{ln}'] = summarize(FC, lab, f'BOLD FC [{ln}]')
np.save(f'{ROOT}/runs/camera_ready/fc_group_mean.npy', FC.astype(np.float32))
np.save(f'{ROOT}/runs/camera_ready/encoding_profile_sim.npy', E.astype(np.float32))
json.dump(out, open(f'{ROOT}/runs/camera_ready/fig2d_check.json', 'w'), indent=1)
