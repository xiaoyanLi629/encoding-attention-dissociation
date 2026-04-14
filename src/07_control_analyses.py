"""
Step 7: 控制分析
Control Analyses to Address Reviewer Concerns

三个控制实验，排除对编码-注意力解离的替代解释：

1. Permutation Test (排列检验)
   - 打乱模态标签后重新训练多模态网络
   - 如果打乱后权重不再均衡 → 原始均衡反映了真实的数据特征

2. High-Encoding Subset Analysis (高编码子集分析)
   - 仅保留编码相关 r > threshold 的脑区
   - 如果权重仍然均衡 → 排除"模型拟合差"的替代解释

3. End-to-End Training Control (端到端训练对照)
   - 训练一个编码器+权重联合优化的端到端模型
   - 如果权重仍然均衡 → 排除"非端到端训练"导致的人为均衡

4. Multi-Model Consistency (多模型一致性)
   - 用不同特征模型重跑编码分析和注意力分析
   - 如果结果一致 → 发现是关于模态而非特定模型
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, ttest_1samp
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import h5py
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import the multimodal model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_multimodal_model_module import (
    PersonalizedMultiModalNetwork, FMRIDataset, MultimodalTrainer
)


# ============================================================
# Shared Data Loading
# ============================================================

class DataLoader_:
    """Helper for loading features and fMRI data."""

    def __init__(self, project_dir, subjects=[1, 2, 3, 5]):
        self.project_dir = project_dir
        self.subjects = subjects
        self.excluded_samples_start = 5
        self.excluded_samples_end = 5
        self.hrf_delay = 3
        self.stimulus_window = 5

    def load_unimodal_correlations(self, results_dir):
        """加载单模态编码相关结果"""
        correlations = {}
        for subject in self.subjects:
            correlations[subject] = {}
            for modality in ['visual', 'audio', 'language']:
                path = os.path.join(
                    results_dir,
                    f'ridge_model_sub-0{subject}_modality-{modality}.npy')
                if os.path.exists(path):
                    data = np.load(path, allow_pickle=True).item()
                    correlations[subject][modality] = data['correlations']
        return correlations


# ============================================================
# Control 1: Permutation Test
# ============================================================

class PermutationTest:
    """排列检验：打乱模态标签"""

    def __init__(self, project_dir, output_dir, n_permutations=100,
                 subjects=[1, 2, 3, 5]):
        self.project_dir = project_dir
        self.output_dir = os.path.join(output_dir, 'permutation_test')
        os.makedirs(self.output_dir, exist_ok=True)
        self.n_permutations = n_permutations
        self.subjects = subjects
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        """运行排列检验"""
        print("\n" + "="*60)
        print("Control 1: Permutation Test")
        print(f"N permutations: {self.n_permutations}")
        print("="*60)

        trainer = MultimodalTrainer(
            project_dir=self.project_dir,
            subjects=self.subjects,
            output_dir=os.path.join(self.output_dir, 'models'),
            max_epochs=50,  # Fewer epochs for permutations
            patience=5,
        )

        # Get original weights (train with correct labels)
        print("\nTraining with correct labels...")
        original_weights = {}
        for subject in self.subjects:
            weights, _, _ = trainer.train_subject_model(subject)
            original_weights[subject] = weights

        # Run permutations
        permuted_weights_all = {s: [] for s in self.subjects}

        for perm_idx in range(self.n_permutations):
            print(f"\n--- Permutation {perm_idx+1}/{self.n_permutations} ---")

            # Create a permuted trainer that shuffles modality labels
            perm_trainer = PermutedMultimodalTrainer(
                project_dir=self.project_dir,
                subjects=self.subjects,
                output_dir=os.path.join(self.output_dir, f'perm_{perm_idx}'),
                max_epochs=50,
                patience=5,
            )

            for subject in self.subjects:
                try:
                    weights, _, _ = perm_trainer.train_subject_model(subject)
                    permuted_weights_all[subject].append(weights)
                except Exception as e:
                    print(f"  Permutation {perm_idx} failed for sub-{subject}: {e}")

        # Analyze results
        results = self._analyze_permutation_results(
            original_weights, permuted_weights_all)

        # Save
        results_path = os.path.join(self.output_dir, 'permutation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_permutation_results(original_weights, permuted_weights_all)

        return results

    def _analyze_permutation_results(self, original_weights, permuted_weights_all):
        """分析排列检验结果"""
        results = {}
        modalities = ['visual', 'audio', 'language']

        for subject in self.subjects:
            sub_key = f'sub-0{subject}'
            orig_w = original_weights[subject]
            perm_w = np.array(permuted_weights_all[subject])

            if len(perm_w) == 0:
                continue

            # Test: are original weights significantly different from permuted?
            # Measure: coefficient of variation (CV) of weights
            orig_cv = np.std(orig_w) / np.mean(orig_w)
            perm_cvs = np.std(perm_w, axis=1) / np.mean(perm_w, axis=1)

            # p-value: fraction of permutations with CV <= original CV
            # (smaller CV = more balanced)
            p_value = float(np.mean(perm_cvs <= orig_cv))

            results[sub_key] = {
                'original_weights': orig_w.tolist(),
                'original_cv': float(orig_cv),
                'permuted_cv_mean': float(np.mean(perm_cvs)),
                'permuted_cv_std': float(np.std(perm_cvs)),
                'p_value': p_value,
                'n_permutations': len(perm_w),
                'permuted_weights_mean': perm_w.mean(axis=0).tolist(),
                'permuted_weights_std': perm_w.std(axis=0).tolist(),
            }

        return results

    def _plot_permutation_results(self, original_weights, permuted_weights_all):
        """绘制排列检验结果"""
        fig, axes = plt.subplots(1, len(self.subjects), figsize=(4*len(self.subjects), 4))
        if len(self.subjects) == 1:
            axes = [axes]

        modalities = ['Visual', 'Audio', 'Language']
        colors = ['#E74C3C', '#3498DB', '#2ECC71']

        for idx, subject in enumerate(self.subjects):
            ax = axes[idx]
            perm_w = np.array(permuted_weights_all[subject])
            orig_w = original_weights[subject]

            if len(perm_w) == 0:
                continue

            # Plot permuted distributions
            for m_idx, (mod, color) in enumerate(zip(modalities, colors)):
                ax.hist(perm_w[:, m_idx], bins=20, alpha=0.3, color=color, label=f'{mod} (perm)')
                ax.axvline(orig_w[m_idx], color=color, linestyle='--', linewidth=2,
                           label=f'{mod} (orig)')

            ax.axvline(1/3, color='black', linestyle=':', linewidth=1, label='Uniform')
            ax.set_title(f'Subject {subject}', fontsize=12)
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Count')
            if idx == 0:
                ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'permutation_distributions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


class PermutedMultimodalTrainer(MultimodalTrainer):
    """带模态标签打乱的训练器"""

    def load_stimulus_features(self, movie_split_names, movie_split_samples):
        """加载特征后打乱模态标签"""
        features = super().load_stimulus_features(movie_split_names, movie_split_samples)

        # 打乱模态列（随机置换三个模态的特征列）
        total_dim = features.shape[1]
        visual_dim = 250 * self.stimulus_window
        audio_dim = 20 * self.stimulus_window
        language_dim = total_dim - visual_dim - audio_dim

        # Split into modalities
        visual = features[:, :visual_dim].copy()
        audio = features[:, visual_dim:visual_dim+audio_dim].copy()
        language = features[:, visual_dim+audio_dim:].copy()

        # Random permutation of modality assignment
        perm = np.random.permutation(3)
        modality_features = [visual, audio, language]
        shuffled = [modality_features[p] for p in perm]

        return np.concatenate(shuffled, axis=1).astype(np.float32)


# ============================================================
# Control 2: High-Encoding Subset Analysis
# ============================================================

class HighEncodingSubsetAnalysis:
    """高编码子集分析：仅分析编码准确度高的脑区"""

    def __init__(self, project_dir, unimodal_results_dir, output_dir,
                 thresholds=[0.1, 0.15, 0.2], subjects=[1, 2, 3, 5]):
        self.project_dir = project_dir
        self.unimodal_results_dir = unimodal_results_dir
        self.output_dir = os.path.join(output_dir, 'high_encoding_subset')
        os.makedirs(self.output_dir, exist_ok=True)
        self.thresholds = thresholds
        self.subjects = subjects
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        """运行高编码子集分析"""
        print("\n" + "="*60)
        print("Control 2: High-Encoding Subset Analysis")
        print(f"Thresholds: {self.thresholds}")
        print("="*60)

        results = {}

        for threshold in self.thresholds:
            print(f"\n--- Threshold: r > {threshold} ---")
            threshold_results = {}

            for subject in self.subjects:
                # Load unimodal correlations to determine which regions to keep
                corrs = {}
                for modality in ['visual', 'audio', 'language']:
                    path = os.path.join(
                        self.unimodal_results_dir,
                        f'ridge_model_sub-0{subject}_modality-{modality}.npy')
                    if os.path.exists(path):
                        data = np.load(path, allow_pickle=True).item()
                        corrs[modality] = data['correlations']

                if not corrs:
                    continue

                # Find regions where ANY modality exceeds threshold
                max_corr = np.maximum.reduce(
                    [corrs[m] for m in ['visual', 'audio', 'language']])
                high_mask = max_corr > threshold
                n_high = int(np.sum(high_mask))

                print(f"  Sub-{subject}: {n_high}/1000 regions above threshold")

                if n_high < 10:
                    print(f"  Too few regions, skipping")
                    continue

                # Train multimodal model on subset of regions
                trainer = MultimodalTrainer(
                    project_dir=self.project_dir,
                    subjects=[subject],
                    output_dir=os.path.join(
                        self.output_dir, f'threshold_{threshold}'),
                    max_epochs=100,
                    patience=10,
                )

                # Override output_dim to only predict high-encoding regions
                weights, _, _ = self._train_subset_model(
                    trainer, subject, high_mask)

                threshold_results[f'sub-0{subject}'] = {
                    'weights': weights.tolist(),
                    'n_regions': n_high,
                    'threshold': threshold,
                }

            results[f'threshold_{threshold}'] = threshold_results

        # Save results
        results_path = os.path.join(self.output_dir, 'subset_analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_subset_results(results)
        return results

    def _train_subset_model(self, trainer, subject, region_mask):
        """训练仅预测子集脑区的模型"""
        # Load full data
        (fmri_train, train_names, train_samples,
         fmri_val, val_names, val_samples) = trainer.load_fmri(subject)

        X_train = trainer.load_stimulus_features(train_names, train_samples)
        X_val = trainer.load_stimulus_features(val_names, val_samples)

        min_train = min(len(X_train), len(fmri_train))
        X_train, fmri_train = X_train[:min_train], fmri_train[:min_train]
        min_val = min(len(X_val), len(fmri_val))
        X_val, fmri_val = X_val[:min_val], fmri_val[:min_val]

        # Apply region mask
        fmri_train = fmri_train[:, region_mask]
        fmri_val = fmri_val[:, region_mask]

        output_dim = fmri_train.shape[1]
        total_dim = X_train.shape[1]
        visual_dim = 250 * trainer.stimulus_window
        audio_dim = 20 * trainer.stimulus_window
        language_dim = total_dim - visual_dim - audio_dim

        # Create and train model
        model = PersonalizedMultiModalNetwork(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            language_dim=language_dim,
            output_dim=output_dim,
            num_subjects=1,
            hidden_dim=trainer.hidden_dim
        ).to(self.device)

        train_dataset = FMRIDataset(X_train, fmri_train, 0)
        val_dataset = FMRIDataset(X_val, fmri_val, 0)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(trainer.max_epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            for batch_X, batch_y, batch_sid in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = model(batch_X, subject_id=0)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= n_batches

            model.eval()
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for batch_X, batch_y, batch_sid in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    pred = model(batch_X, subject_id=0)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item()
                    n_val += 1
            val_loss /= n_val
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= trainer.patience:
                break

        model.load_state_dict(best_state)
        with torch.no_grad():
            weights = torch.softmax(model.modality_weights, dim=0).cpu().numpy()

        print(f"  Sub-{subject} subset weights: V={weights[0]:.4f} A={weights[1]:.4f} "
              f"L={weights[2]:.4f}")

        return weights, None, best_val_loss

    def _plot_subset_results(self, results):
        """绘制子集分析结果"""
        fig, ax = plt.subplots(figsize=(10, 6))

        modalities = ['Visual', 'Audio', 'Language']
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        x_positions = []
        x_labels = []
        offset = 0

        for threshold_key, threshold_data in sorted(results.items()):
            threshold = float(threshold_key.split('_')[1])
            for sub_key, sub_data in sorted(threshold_data.items()):
                weights = sub_data['weights']
                x_pos = offset
                x_positions.append(x_pos)
                x_labels.append(f"r>{threshold}\n{sub_key}")

                for m_idx, (mod, color) in enumerate(zip(modalities, colors)):
                    ax.bar(x_pos + m_idx*0.25 - 0.25, weights[m_idx], 0.25,
                           color=color, alpha=0.8,
                           label=mod if offset == 0 and sub_key == sorted(threshold_data.keys())[0] else '')

                offset += 1.2

        ax.axhline(y=1/3, color='black', linestyle='--', linewidth=1, label='Uniform (0.33)')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Modality Weights by Encoding Threshold')
        ax.set_xticks([p for p in range(len(x_labels))])
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.legend()
        ax.set_ylim(0, 0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'subset_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# Control 3: End-to-End Training
# ============================================================

class EndToEndControl:
    """端到端训练对照：编码器+权重联合优化"""

    def __init__(self, project_dir, output_dir, subjects=[1, 2, 3, 5]):
        self.project_dir = project_dir
        self.output_dir = os.path.join(output_dir, 'end_to_end')
        os.makedirs(self.output_dir, exist_ok=True)
        self.subjects = subjects
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        """运行端到端训练对照"""
        print("\n" + "="*60)
        print("Control 3: End-to-End Training Control")
        print("="*60)

        results = {}

        for subject in self.subjects:
            print(f"\nTraining end-to-end model for Subject {subject}...")

            trainer = MultimodalTrainer(
                project_dir=self.project_dir,
                subjects=[subject],
                output_dir=os.path.join(self.output_dir, 'models'),
                max_epochs=200,
                patience=10,
                lr=5e-5,  # Lower LR for stable end-to-end training
            )

            weights, correlations, val_loss = trainer.train_subject_model(subject)

            results[f'sub-0{subject}'] = {
                'weights': weights.tolist(),
                'val_loss': float(val_loss),
                'mean_val_corr': float(np.mean(correlations)),
            }

        # Save results
        results_path = os.path.join(self.output_dir, 'end_to_end_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("\nEnd-to-end results:")
        for sub, data in results.items():
            w = data['weights']
            print(f"  {sub}: V={w[0]:.4f} A={w[1]:.4f} L={w[2]:.4f}")

        return results


# ============================================================
# Control 4: Multi-Model Consistency
# ============================================================

class MultiModelConsistency:
    """多模型一致性分析"""

    def __init__(self, project_dir, output_dir, subjects=[1, 2, 3, 5]):
        self.project_dir = project_dir
        self.output_dir = os.path.join(output_dir, 'multi_model')
        os.makedirs(self.output_dir, exist_ok=True)
        self.subjects = subjects

    def run_unimodal_comparison(self, additional_features_dir):
        """比较不同特征模型的单模态编码结果"""
        print("\n" + "="*60)
        print("Control 4: Multi-Model Consistency - Unimodal Encoding")
        print("="*60)

        from train_unimodal_models_module import UnimodalModelTrainer

        results = {}

        # List available additional feature sets
        if not os.path.exists(additional_features_dir):
            print(f"Additional features not found at {additional_features_dir}")
            return results

        for model_dir in sorted(os.listdir(additional_features_dir)):
            pca_dir = os.path.join(additional_features_dir, model_dir, 'pca')
            if not os.path.isdir(pca_dir):
                continue

            print(f"\n  Model: {model_dir}")

            # Determine which modalities this model provides
            available_modalities = [m for m in ['visual', 'audio', 'language']
                                    if os.path.exists(os.path.join(pca_dir, m))]

            if not available_modalities:
                continue

            # Train unimodal models with these features
            model_output = os.path.join(self.output_dir, f'unimodal_{model_dir}')
            os.makedirs(model_output, exist_ok=True)

            for subject in self.subjects:
                for modality in available_modalities:
                    trainer = UnimodalModelTrainer(
                        self.project_dir, [subject], model_output)
                    # Override feature path
                    trainer.features_dir_override = pca_dir
                    try:
                        _, correlations = trainer.train_single_subject_modality(
                            subject, modality)
                        key = f'{model_dir}/{modality}/sub-0{subject}'
                        results[key] = {
                            'mean_r': float(np.mean(correlations)),
                            'max_r': float(np.max(correlations)),
                        }
                    except Exception as e:
                        print(f"    Error: {e}")

        # Save
        results_path = os.path.join(self.output_dir, 'multi_model_unimodal_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Control Analyses')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition',
                        help='Project root directory')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for control analyses')
    parser.add_argument('--unimodal_results_dir', default=None,
                        help='Directory with unimodal model results')
    parser.add_argument('--subjects', default='1,2,3,5', help='Subject list')
    parser.add_argument('--controls', default='permutation,subset,e2e',
                        help='Which controls to run (comma-separated)')
    parser.add_argument('--n_permutations', type=int, default=100,
                        help='Number of permutations for permutation test')
    parser.add_argument('--thresholds', default='0.1,0.15,0.2',
                        help='Encoding thresholds for subset analysis')

    args = parser.parse_args()
    subjects = [int(s.strip()) for s in args.subjects.split(',')]
    controls = [c.strip() for c in args.controls.split(',')]
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    if 'permutation' in controls:
        perm = PermutationTest(
            args.project_dir, args.output_dir,
            n_permutations=args.n_permutations, subjects=subjects)
        all_results['permutation'] = perm.run()

    if 'subset' in controls:
        if args.unimodal_results_dir is None:
            print("WARNING: --unimodal_results_dir required for subset analysis")
        else:
            subset = HighEncodingSubsetAnalysis(
                args.project_dir, args.unimodal_results_dir, args.output_dir,
                thresholds=thresholds, subjects=subjects)
            all_results['subset'] = subset.run()

    if 'e2e' in controls:
        e2e = EndToEndControl(
            args.project_dir, args.output_dir, subjects=subjects)
        all_results['e2e'] = e2e.run()

    # Save combined results
    summary_path = os.path.join(args.output_dir, 'control_analyses_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("All control analyses complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
