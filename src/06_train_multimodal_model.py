"""
Step 6: 训练个性化多模态神经网络
Train Personalized Multimodal Neural Network

训练包含显式可学习模态权重的多模态网络，用于分析跨模态注意力分配策略。
架构：
- 模态特异性编码器（visual, audio, language）
- 多头交叉注意力融合
- 显式可学习模态权重（softmax归一化）
- 被试特异性适配器

训练完成后保存 .pth 文件，供 03_crossmodal_attention_analysis.py 加载分析。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import h5py
import json
from datetime import datetime


class PersonalizedMultiModalNetwork(nn.Module):
    """个性化多模态神经网络"""

    def __init__(self, visual_dim, audio_dim, language_dim, output_dim,
                 num_subjects=4, hidden_dim=512):
        super().__init__()

        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.language_dim = language_dim

        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # 核心：显式可学习模态权重
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)

        self.subject_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_subjects)
        ])

        self.global_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.fmri_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, subject_id=None, return_attention=False):
        batch_size = x.size(0)

        visual_features = x[:, :self.visual_dim]
        audio_features = x[:, self.visual_dim:self.visual_dim + self.audio_dim]
        language_features = x[:, self.visual_dim + self.audio_dim:]

        visual_encoded = self.visual_encoder(visual_features)
        audio_encoded = self.audio_encoder(audio_features)
        language_encoded = self.language_encoder(language_features)

        modalities = torch.stack([visual_encoded, audio_encoded, language_encoded], dim=1)

        attended_features, attention_weights = self.cross_modal_attention(
            modalities, modalities, modalities
        )

        weights = torch.softmax(self.modality_weights, dim=0)
        weighted_features = attended_features * weights.view(1, 3, 1)

        combined_features = weighted_features.reshape(batch_size, -1)

        if subject_id is not None and 0 <= subject_id < len(self.subject_adapters):
            adapted_features = self.subject_adapters[subject_id](combined_features)
        else:
            adapted_features = self.global_adapter(combined_features)

        fmri_signals = self.fmri_generator(adapted_features)

        if return_attention:
            return fmri_signals, attention_weights, weights
        return fmri_signals


class FMRIDataset(Dataset):
    """fMRI 数据集"""

    def __init__(self, features, fmri, subject_id=None):
        self.features = torch.FloatTensor(features)
        self.fmri = torch.FloatTensor(fmri)
        self.subject_id = subject_id

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.fmri[idx], self.subject_id


class MultimodalTrainer:
    """多模态模型训练器"""

    def __init__(self, project_dir, subjects=[1, 2, 3, 5], output_dir=None,
                 hidden_dim=512, lr=1e-4, batch_size=32, max_epochs=200,
                 patience=10, feature_set='official'):
        self.project_dir = project_dir
        self.subjects = subjects
        self.subject_id_map = {s: i for i, s in enumerate(subjects)}
        self.modalities = ['visual', 'audio', 'language']
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.feature_set = feature_set

        self.output_dir = output_dir or os.path.join(
            project_dir, 'analysis', 'results', 'trained_encoding_models')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # HRF参数
        self.excluded_samples_start = 5
        self.excluded_samples_end = 5
        self.hrf_delay = 3
        self.stimulus_window = 5

    def _get_features_dir(self):
        """获取特征目录"""
        if self.feature_set == 'official':
            return os.path.join(self.project_dir, 'data', 'features',
                                'official_stimulus_features', 'pca', 'friends_movie10')
        else:
            return os.path.join(self.project_dir, 'data', 'features',
                                'additional_features', self.feature_set, 'pca')

    def load_fmri(self, subject):
        """加载fMRI数据，分为训练集和验证集"""
        fmri_dir = os.path.join(self.project_dir, 'data', 'fmri', f'sub-0{subject}', 'func')

        fmri_file_friends = (
            f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_'
            f'atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5')
        fmri_file_movie10 = (
            f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_'
            f'atlas-Schaefer18_parcel-1000Par7Net_bold.h5')

        fmri_friends = h5py.File(os.path.join(fmri_dir, fmri_file_friends), 'r')
        fmri_movie10 = h5py.File(os.path.join(fmri_dir, fmri_file_movie10), 'r')

        fmri_train = []
        fmri_val = []
        train_names = []
        train_samples = []
        val_names = []
        val_samples = []

        # Friends S1-S5 → 训练集, S6 → 验证集
        for key, val in fmri_friends.items():
            fmri_part = val[self.excluded_samples_start:-self.excluded_samples_end]
            name = key[13:]
            if name.startswith('s06'):
                fmri_val.append(fmri_part)
                val_names.append(name)
                val_samples.append(len(fmri_part))
            else:
                fmri_train.append(fmri_part)
                train_names.append(name)
                train_samples.append(len(fmri_part))

        # Movie10 → 验证集
        for key, val in fmri_movie10.items():
            fmri_part = val[self.excluded_samples_start:-self.excluded_samples_end]
            if key[13:20] == 'figures':
                name = key[13:22]
            elif key[13:17] == 'life':
                name = key[13:19]
            else:
                name = key[13:]
            fmri_val.append(fmri_part)
            val_names.append(name)
            val_samples.append(len(fmri_part))

        fmri_friends.close()
        fmri_movie10.close()

        fmri_train = np.concatenate(fmri_train, axis=0) if fmri_train else np.array([])
        fmri_val = np.concatenate(fmri_val, axis=0) if fmri_val else np.array([])

        return (fmri_train, train_names, train_samples,
                fmri_val, val_names, val_samples)

    def load_stimulus_features(self, movie_split_names, movie_split_samples):
        """加载并拼接所有模态的刺激特征"""
        features_dir = self._get_features_dir()

        features = {}
        for modality in self.modalities:
            feature_file = os.path.join(features_dir, modality, 'features_train.npy')
            features[modality] = np.load(feature_file, allow_pickle=True).item()

        stim_features = []
        for m, split in enumerate(movie_split_names):
            for s in range(movie_split_samples[m]):
                f_all = []
                for modality in self.modalities:
                    if modality in ['visual', 'audio']:
                        if s < (self.stimulus_window + self.hrf_delay):
                            idx_start = self.excluded_samples_start
                            idx_end = idx_start + self.stimulus_window
                        else:
                            idx_start = (s + self.excluded_samples_start
                                         - self.hrf_delay - self.stimulus_window + 1)
                            idx_end = idx_start + self.stimulus_window

                        if idx_end > len(features[modality][split]):
                            idx_end = len(features[modality][split])
                            idx_start = idx_end - self.stimulus_window

                        f = features[modality][split][idx_start:idx_end].flatten()
                    else:  # language
                        idx = s + self.excluded_samples_start - self.hrf_delay
                        if idx >= len(features[modality][split]) - self.hrf_delay:
                            f = features[modality][split][-1].flatten()
                        else:
                            f = features[modality][split][max(0, idx)].flatten()
                    f_all.append(f)
                stim_features.append(np.concatenate(f_all))

        return np.array(stim_features, dtype=np.float32)

    def train_subject_model(self, subject):
        """训练单个被试的多模态模型"""
        print(f"\n{'='*60}")
        print(f"Training multimodal model for Subject {subject}")
        print(f"{'='*60}")

        # 加载数据
        (fmri_train, train_names, train_samples,
         fmri_val, val_names, val_samples) = self.load_fmri(subject)

        X_train = self.load_stimulus_features(train_names, train_samples)
        X_val = self.load_stimulus_features(val_names, val_samples)

        # 对齐样本数
        min_train = min(len(X_train), len(fmri_train))
        X_train, fmri_train = X_train[:min_train], fmri_train[:min_train]
        min_val = min(len(X_val), len(fmri_val))
        X_val, fmri_val = X_val[:min_val], fmri_val[:min_val]

        print(f"  Train: X={X_train.shape}, y={fmri_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={fmri_val.shape}")

        # 确定各模态特征维度
        # PCA features: visual=250*5=1250, audio=20*5=100, language=250
        # 从实际数据推算
        total_dim = X_train.shape[1]
        # 基于官方特征配置: visual 250D*5TR, audio 20D*5TR, language 250D*1TR
        visual_dim = 250 * self.stimulus_window  # 1250
        audio_dim = 20 * self.stimulus_window     # 100
        language_dim = total_dim - visual_dim - audio_dim  # 250
        output_dim = fmri_train.shape[1]  # 1000

        print(f"  Feature dims: visual={visual_dim}, audio={audio_dim}, "
              f"language={language_dim}, output={output_dim}")

        subject_id = self.subject_id_map[subject]

        # 创建数据集
        train_dataset = FMRIDataset(X_train, fmri_train, subject_id)
        val_dataset = FMRIDataset(X_val, fmri_val, subject_id)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        # 创建模型
        model = PersonalizedMultiModalNetwork(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            language_dim=language_dim,
            output_dim=output_dim,
            num_subjects=len(self.subjects),
            hidden_dim=self.hidden_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            model.train()
            train_loss = 0
            n_batches = 0
            for batch_X, batch_y, batch_sid in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                sid = batch_sid[0].item()

                optimizer.zero_grad()
                pred = model(batch_X, subject_id=sid)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # Validation
            model.eval()
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for batch_X, batch_y, batch_sid in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    sid = batch_sid[0].item()
                    pred = model(batch_X, subject_id=sid)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
                    n_val += 1

            val_loss /= n_val
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == 0:
                weights = torch.softmax(model.modality_weights, dim=0).detach().cpu().numpy()
                print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Weights: V={weights[0]:.3f} A={weights[1]:.3f} L={weights[2]:.3f}"
                      f"{' *best*' if patience_counter == 0 else ''}")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # 恢复最优模型
        model.load_state_dict(best_state)
        model = model.to(self.device)

        # 提取最终权重
        with torch.no_grad():
            weights = torch.softmax(model.modality_weights, dim=0).cpu().numpy()

        print(f"\n  Final modality weights: Visual={weights[0]:.4f}, "
              f"Audio={weights[1]:.4f}, Language={weights[2]:.4f}")
        print(f"  Best validation loss: {best_val_loss:.6f}")

        # 计算验证集上的编码相关
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_sid in val_loader:
                batch_X = batch_X.to(self.device)
                sid = batch_sid[0].item()
                pred = model(batch_X, subject_id=sid)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        correlations = []
        for i in range(targets.shape[1]):
            if np.std(targets[:, i]) > 0 and np.std(preds[:, i]) > 0:
                r, _ = pearsonr(targets[:, i], preds[:, i])
                correlations.append(r if not np.isnan(r) else 0)
            else:
                correlations.append(0)
        correlations = np.array(correlations)

        print(f"  Val encoding: mean r={np.mean(correlations):.4f}, "
              f"max r={np.max(correlations):.4f}, "
              f"regions r>0.1: {np.sum(correlations > 0.1)}")

        # 保存模型
        save_path = os.path.join(
            self.output_dir,
            f'personalized_multimodal_model_sub-0{subject}_modality-all.pth')

        torch.save({
            'model_state_dict': best_state,
            'modality_weights': weights,
            'val_loss': best_val_loss,
            'val_correlations': correlations,
            'config': {
                'visual_dim': visual_dim,
                'audio_dim': audio_dim,
                'language_dim': language_dim,
                'output_dim': output_dim,
                'hidden_dim': self.hidden_dim,
                'num_subjects': len(self.subjects),
                'feature_set': self.feature_set,
            }
        }, save_path)

        print(f"  Model saved to: {save_path}")

        return weights, correlations, best_val_loss

    def train_all(self):
        """训练所有被试的模型"""
        print("=" * 70)
        print("Personalized Multimodal Neural Network Training")
        print(f"Feature set: {self.feature_set}")
        print(f"Subjects: {self.subjects}")
        print(f"Device: {self.device}")
        print("=" * 70)

        all_weights = {}
        all_correlations = {}
        all_losses = {}

        for subject in self.subjects:
            weights, correlations, val_loss = self.train_subject_model(subject)
            all_weights[f'sub-0{subject}'] = weights.tolist()
            all_correlations[f'sub-0{subject}'] = {
                'mean': float(np.mean(correlations)),
                'max': float(np.max(correlations)),
                'std': float(np.std(correlations)),
            }
            all_losses[f'sub-0{subject}'] = float(val_loss)

        # 保存汇总
        summary = {
            'timestamp': datetime.now().isoformat(),
            'feature_set': self.feature_set,
            'modality_weights': all_weights,
            'val_correlations': all_correlations,
            'val_losses': all_losses,
            'config': {
                'hidden_dim': self.hidden_dim,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'patience': self.patience,
            }
        }

        summary_path = os.path.join(self.output_dir, 'multimodal_training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)
        print(f"\nModality Weights (softmax-normalized):")
        for sub, w in all_weights.items():
            print(f"  {sub}: Visual={w[0]:.4f}, Audio={w[1]:.4f}, Language={w[2]:.4f}")

        avg_w = np.mean([w for w in all_weights.values()], axis=0)
        print(f"\n  Average: Visual={avg_w[0]:.4f}, Audio={avg_w[1]:.4f}, "
              f"Language={avg_w[2]:.4f}")

        print(f"\nResults saved to: {self.output_dir}")
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Train Personalized Multimodal Neural Network')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition',
                        help='Project root directory')
    parser.add_argument('--subjects', default='1,2,3,5', help='Subject list')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--feature_set', default='official',
                        help='Feature set to use: "official" or model name')

    args = parser.parse_args()
    subjects = [int(s.strip()) for s in args.subjects.split(',')]

    trainer = MultimodalTrainer(
        project_dir=args.project_dir,
        subjects=subjects,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        feature_set=args.feature_set,
    )

    trainer.train_all()


if __name__ == "__main__":
    main()
