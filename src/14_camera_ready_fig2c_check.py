# Does Ward clustering of encoding profiles (rV, rA, rL) recover the 7 canonical networks?
from pathlib import Path
import numpy as np, sys
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{ROOT}/src')
from brain_region_mapping import SCHAEFER_7_NETWORKS as NET
order = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention', 'Limbic', 'Frontoparietal', 'Default']
lab = np.full(1000, -1)
for i, k in enumerate(order): lab[NET[k]['left_hemisphere'] + NET[k]['right_hemisphere']] = i
R = np.mean([np.stack([np.load(f'{ROOT}/runs/run_revised/unimodal_models/ridge_model_sub-{s}_modality-{m}.npy', allow_pickle=True).item()['correlations'] for m in ['visual', 'audio', 'language']], 1) for s in ['01', '02', '03', '05']], 0)
Z = linkage(R, 'ward'); m = lab >= 0
rng = np.random.default_rng(0)
for k in [2, 7]:
    cl = fcluster(Z, k, 'maxclust')
    null = [adjusted_rand_score(rng.permutation(lab[m]), cl[m]) for _ in range(200)]
    print(f'k={k}: ARI vs networks = {adjusted_rand_score(lab[m], cl[m]):.3f}  (label-shuffle null 95th pct {np.percentile(null, 95):.3f})')
    if k == 2:
        for c in (1, 2):
            frac = np.bincount(lab[m][cl[m] == c], minlength=7) / np.bincount(lab[m], minlength=7)
            print('  cluster', c, 'n =', (cl[m] == c).sum(), 'share of each network:', dict(zip(['Vis', 'Som', 'DAN', 'VAN', 'Lim', 'FPN', 'DMN'], frac.round(2))))
