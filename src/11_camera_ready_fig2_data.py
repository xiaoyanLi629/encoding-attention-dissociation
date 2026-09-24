# Real values for camera-ready Fig. 2b (MII by network) and Fig. 2f (per-subject weights)
import os
import numpy as np, sys, json
sys.path.insert(0, os.path.join(ROOT, 'src'))
from brain_region_mapping import SCHAEFER_7_NETWORKS as NET
RUN = os.path.join(ROOT, 'runs', 'run_revised')
subs, mods = ['01', '02', '03', '05'], ['visual', 'audio', 'language']
r = {s: np.stack([np.load(f'{RUN}/unimodal_models/ridge_model_sub-{s}_modality-{m}.npy', allow_pickle=True).item()['correlations'] for m in mods], 1) for s in subs}
order = [('Vis', 'Visual'), ('Som', 'Somatomotor'), ('DAN', 'DorsalAttention'), ('VAN', 'VentralAttention'), ('Lim', 'Limbic'), ('FPN', 'Frontoparietal'), ('DMN', 'Default')]
mii = {}
for short, key in order:
    idx = np.array(NET[key]['left_hemisphere'] + NET[key]['right_hemisphere'])
    per_sub = [float(np.mean(1 - r[s][idx].std(1) / r[s][idx].mean(1))) for s in subs]
    mii[short] = {'mean': float(np.mean(per_sub)), 'sd': float(np.std(per_sub, ddof=1)), 'per_subject': per_sub}
w = json.load(open(f'{RUN}/trained_models/multimodal_training_summary.json'))['modality_weights']
out = {'MII': mii, 'weights': w,
       'note': 'MII = parcel-level 1 - SD/mean over (rV, rA, rL), averaged within network (project network ranges), per subject; in-sample Ridge r as in Table I'}
os.makedirs(os.path.join(ROOT, 'runs', 'camera_ready'), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, 'runs', 'camera_ready', 'fig2_data.json'), 'w'), indent=1)
print(json.dumps({k: round(v['mean'], 3) for k, v in mii.items()}), json.dumps({k: round(v['sd'], 3) for k, v in mii.items()}))
