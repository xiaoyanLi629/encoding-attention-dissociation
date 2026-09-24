"""
Camera-ready control: does the learned modality weight depend on its initialization?

Same data pipeline, architecture and optimizer as src/train_multimodal_model_module.py.
Only differences: (1) modality_weights are initialized to log(p0) so that softmax = p0,
(2) optionally frozen, (3) per-epoch weight trajectory is logged, (4) optional parcel subset.
"""
from pathlib import Path
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ROOT, 'src'))
from train_multimodal_model_module import (PersonalizedMultiModalNetwork,
                                           FMRIDataset, MultimodalTrainer)

PCA_DIR = os.environ.get('PCA_DIR', os.path.join(ROOT, 'data', 'features', 'official_stimulus_features', 'pca', 'friends_movie10'))
CACHE = os.path.join(ROOT, 'runs', 'camera_ready', 'cache')
INITS = {
    'uniform':  [1/3, 1/3, 1/3],
    'visual':   [0.6, 0.2, 0.2],
    'audio':    [0.2, 0.6, 0.2],
    'language': [0.2, 0.2, 0.6],
}


class Trainer(MultimodalTrainer):
    def _get_features_dir(self):
        return PCA_DIR


def load_data(subject):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f'sub-0{subject}.npz')
    if os.path.exists(f):
        d = np.load(f)
        return d['Xtr'], d['ytr'], d['Xva'], d['yva'], int(d['n_friends_s6'])
    tr = Trainer(ROOT, subjects=[1, 2, 3, 5], output_dir=CACHE)
    ytr, trn, trs, yva, van, vas = tr.load_fmri(subject)
    Xtr = tr.load_stimulus_features(trn, trs)
    Xva = tr.load_stimulus_features(van, vas)
    n = min(len(Xtr), len(ytr)); Xtr, ytr = Xtr[:n], ytr[:n]
    n = min(len(Xva), len(yva)); Xva, yva = Xva[:n], yva[:n]
    n_s6 = int(sum(s for nm, s in zip(van, vas) if nm.startswith('s06')))
    np.savez(f, Xtr=Xtr, ytr=ytr.astype(np.float32), Xva=Xva,
             yva=yva.astype(np.float32), n_friends_s6=n_s6)
    return Xtr, ytr, Xva, yva, n_s6


def run(subject, init, freeze, seed, out, parcels=None, lr=1e-4, tag=''):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xva, yva, n_s6 = load_data(subject)
    if parcels is not None:
        ytr, yva = ytr[:, parcels], yva[:, parcels]
    sid = [1, 2, 3, 5].index(subject)
    dev = torch.device('cuda')
    vdim, adim = 1250, 100
    ldim = Xtr.shape[1] - vdim - adim
    model = PersonalizedMultiModalNetwork(vdim, adim, ldim, ytr.shape[1],
                                          num_subjects=4, hidden_dim=512).to(dev)
    if init == 'random':
        p0 = np.random.dirichlet([1, 1, 1])
    else:
        p0 = np.array(INITS[init])
    with torch.no_grad():
        model.modality_weights.copy_(torch.tensor(np.log(p0), dtype=torch.float32))
    model.modality_weights.requires_grad_(not freeze)

    n_params = sum(p.numel() for p in model.parameters())
    n_used = n_params - sum(p.numel() for i, a in enumerate(model.subject_adapters)
                            if i != sid for p in a.parameters()) \
        - sum(p.numel() for p in model.global_adapter.parameters())

    tl = DataLoader(FMRIDataset(Xtr, ytr, sid), batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(FMRIDataset(Xva, yva, sid), batch_size=32, shuffle=False,
                    num_workers=4, pin_memory=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()

    sm = lambda: torch.softmax(model.modality_weights, 0).detach().cpu().numpy().tolist()
    traj = [{'epoch': 0, 'w': sm()}]
    best, best_state, best_ep, pc = float('inf'), None, 0, 0
    for ep in range(1, 201):
        model.train(); trl = 0; nb = 0
        for xb, yb, _ in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = crit(model(xb, subject_id=sid), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            trl += loss.item(); nb += 1
        model.eval(); vls = 0; nv = 0
        with torch.no_grad():
            for xb, yb, _ in vl:
                vls += crit(model(xb.to(dev), subject_id=sid), yb.to(dev)).item(); nv += 1
        vls /= nv; sch.step(vls)
        traj.append({'epoch': ep, 'w': sm(), 'train_loss': trl / nb, 'val_loss': vls})
        if vls < best:
            best, best_ep, pc = vls, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pc += 1
        if pc >= 10:
            break
    model.load_state_dict(best_state)
    model.eval(); P = []
    with torch.no_grad():
        for xb, _, _ in vl:
            P.append(model(xb.to(dev), subject_id=sid).cpu().numpy())
    P = np.concatenate(P)
    r = np.array([pearsonr(yva[:, i], P[:, i])[0] if P[:, i].std() > 0 else 0
                  for i in range(yva.shape[1])])
    r = np.nan_to_num(r)
    res = {'subject': subject, 'init': init, 'p0': p0.tolist(), 'freeze': freeze,
           'seed': seed, 'lr': lr, 'tag': tag, 'final_w': sm(), 'best_epoch': best_ep,
           'stop_epoch': traj[-1]['epoch'], 'best_val_loss': best,
           'val_mean_r': float(r.mean()), 'val_n_r_gt_0.1': int((r > 0.1).sum()),
           'n_params_total': int(n_params), 'n_params_used': int(n_used),
           'n_train_TR': int(len(Xtr)), 'n_val_TR': int(len(Xva)),
           'n_val_friends_s6_TR': int(n_s6), 'n_parcels': int(ytr.shape[1]),
           'trajectory': traj}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
    np.save(out.replace('.json', '_valr.npy'), r)
    print(json.dumps({k: v for k, v in res.items() if k != 'trajectory'}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--init', default='uniform')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--parcels', default=None, help='npy file with parcel indices')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    parcels = np.load(a.parcels) if a.parcels else None
    run(a.subject, a.init, a.freeze, a.seed, a.out, parcels, a.lr, a.tag)
