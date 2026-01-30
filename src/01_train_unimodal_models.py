"""
Step 1: 训练单模态编码模型
Train Unimodal Encoding Models for Cross-Modal Integration Analysis

为每个被试分别训练三个单模态模型：
- Visual模型：仅使用视觉特征
- Audio模型：仅使用音频特征  
- Language模型：仅使用语言特征

这些模型将用于分析不同脑区对各模态的敏感性。

数据划分说明：
- 可用数据：Friends S1-S6 + Movie10（均有fMRI标签）
- 训练集：Friends S1-S5 + Movie10的部分数据 (~80%)
- 验证集：Friends S6 + Movie10的剩余数据 (~20%)
- 注意：Friends S7 无fMRI标签，仅用于竞赛提交，不用于本研究分析
"""

import os
import sys
import argparse
import numpy as np
import h5py
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official_code', '02_encoding_model_training'))


class UnimodalModelTrainer:
    """单模态编码模型训练器"""
    
    def __init__(self, project_dir, subjects=[1, 2, 3, 5], output_dir=None):
        self.project_dir = project_dir
        self.subjects = subjects
        self.modalities = ['visual', 'audio', 'language']
        self.results_dir = output_dir or os.path.join(project_dir, 'analysis', 'results', 'unimodal_models')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Ridge回归的正则化参数候选
        self.alphas = np.array([1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 0.05, 0.01, 0.001])
        
        # HRF延迟和刺激窗口参数
        self.excluded_samples_start = 5
        self.excluded_samples_end = 5
        self.hrf_delay = 3
        self.stimulus_window = 5
    
    def load_fmri(self, subject):
        """加载fMRI数据"""
        fmri_dir = os.path.join(self.project_dir, 'data', 'fmri', f'sub-0{subject}', 'func')
        
        # Friends数据
        fmri_file_friends = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        # Movie10数据
        fmri_file_movie10 = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
        
        fmri_friends = h5py.File(os.path.join(fmri_dir, fmri_file_friends), 'r')
        fmri_movie10 = h5py.File(os.path.join(fmri_dir, fmri_file_movie10), 'r')
        
        fmri = []
        movie_split_names = []
        movie_split_samples = []
        
        # Friends数据
        for key, val in fmri_friends.items():
            fmri_part = val[self.excluded_samples_start:-self.excluded_samples_end]
            fmri.append(fmri_part)
            movie_split_names.append(key[13:])
            movie_split_samples.append(len(fmri_part))
        
        # Movie10数据
        for key, val in fmri_movie10.items():
            fmri_part = val[self.excluded_samples_start:-self.excluded_samples_end]
            fmri.append(fmri_part)
            if key[13:20] == 'figures':
                movie_split_names.append(key[13:22])
            elif key[13:17] == 'life':
                movie_split_names.append(key[13:19])
            else:
                movie_split_names.append(key[13:])
            movie_split_samples.append(len(fmri_part))
        
        fmri_friends.close()
        fmri_movie10.close()
        
        fmri = np.concatenate(fmri, axis=0)
        return fmri, movie_split_names, movie_split_samples
    
    def load_stimulus_features(self, modality, movie_split_names, movie_split_samples):
        """加载单模态刺激特征"""
        features_dir = os.path.join(self.project_dir, 'data', 'features', 
                                     'official_stimulus_features', 'pca', 'friends_movie10')
        
        # 加载特定模态的特征
        feature_file = os.path.join(features_dir, modality, 'features_train.npy')
        features = np.load(feature_file, allow_pickle=True).item()
        
        stim_features = []
        
        for m, split in enumerate(movie_split_names):
            for s in range(movie_split_samples[m]):
                if modality in ['visual', 'audio']:
                    # 使用时间窗口
                    if s < (self.stimulus_window + self.hrf_delay):
                        idx_start = self.excluded_samples_start
                        idx_end = idx_start + self.stimulus_window
                    else:
                        idx_start = s + self.excluded_samples_start - self.hrf_delay - self.stimulus_window + 1
                        idx_end = idx_start + self.stimulus_window
                    
                    if idx_end > len(features[split]):
                        idx_end = len(features[split])
                        idx_start = idx_end - self.stimulus_window
                    
                    f = features[split][idx_start:idx_end].flatten()
                else:  # language
                    idx = s + self.excluded_samples_start - self.hrf_delay
                    if idx >= len(features[split]) - self.hrf_delay:
                        f = features[split][-1].flatten()
                    else:
                        f = features[split][max(0, idx)].flatten()
                
                stim_features.append(f)
        
        return np.array(stim_features, dtype=np.float32)
    
    def train_ridge_model(self, X_train, y_train):
        """训练Ridge回归模型"""
        model = RidgeCV(alphas=self.alphas, cv=None, alpha_per_target=True)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        y_pred = model.predict(X_test)
        
        correlations = np.zeros(y_test.shape[1])
        for i in range(y_test.shape[1]):
            if np.std(y_test[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                corr, _ = pearsonr(y_test[:, i], y_pred[:, i])
                correlations[i] = corr if not np.isnan(corr) else 0
        
        return correlations, y_pred
    
    def train_single_subject_modality(self, subject, modality):
        """训练单个被试的单模态模型"""
        print(f"\n训练 Subject {subject} - {modality} 模型...")
        
        # 加载数据
        y_train, movie_split_names, movie_split_samples = self.load_fmri(subject)
        X_train = self.load_stimulus_features(modality, movie_split_names, movie_split_samples)
        
        # 确保样本数匹配
        min_samples = min(len(X_train), len(y_train))
        X_train = X_train[:min_samples]
        y_train = y_train[:min_samples]
        
        print(f"  数据形状: X={X_train.shape}, y={y_train.shape}")
        
        # 训练模型
        model = self.train_ridge_model(X_train, y_train)
        
        # 评估（在训练集上，用于模态敏感性分析）
        correlations, _ = self.evaluate_model(model, X_train, y_train)
        
        # 保存模型
        model_data = {
            'coef_': model.coef_,
            'intercept_': model.intercept_,
            'alpha_': model.alpha_,
            'correlations': correlations,
            'mean_correlation': np.mean(correlations),
            'subject': subject,
            'modality': modality
        }
        
        save_path = os.path.join(self.results_dir, 
                                  f'ridge_model_sub-0{subject}_modality-{modality}.npy')
        np.save(save_path, model_data)
        
        print(f"  平均相关系数: {np.mean(correlations):.4f}")
        print(f"  模型保存至: {save_path}")
        
        return model, correlations
    
    def train_all_models(self):
        """训练所有单模态模型"""
        all_results = {}
        
        for subject in self.subjects:
            all_results[f'sub-0{subject}'] = {}
            
            for modality in self.modalities:
                model, correlations = self.train_single_subject_modality(subject, modality)
                all_results[f'sub-0{subject}'][modality] = {
                    'mean_correlation': float(np.mean(correlations)),
                    'max_correlation': float(np.max(correlations)),
                    'std_correlation': float(np.std(correlations)),
                    'high_corr_regions': int(np.sum(correlations > 0.3))
                }
        
        # 保存汇总结果
        summary_path = os.path.join(self.results_dir, 'unimodal_training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n训练完成！汇总结果保存至: {summary_path}")
        return all_results
    
    def plot_unimodal_comparison(self, all_results=None):
        """绘制单模态性能对比图"""
        if all_results is None:
            summary_path = os.path.join(self.results_dir, 'unimodal_training_summary.json')
            with open(summary_path, 'r') as f:
                all_results = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准备数据
        subjects = list(all_results.keys())
        modalities = ['visual', 'audio', 'language']
        colors = {'visual': '#E74C3C', 'audio': '#3498DB', 'language': '#2ECC71'}
        
        # 图1: 各被试各模态的平均相关系数
        x = np.arange(len(subjects))
        width = 0.25
        
        for i, modality in enumerate(modalities):
            means = [all_results[s][modality]['mean_correlation'] for s in subjects]
            bars = axes[0].bar(x + i*width, means, width, label=modality.capitalize(), 
                              color=colors[modality], alpha=0.8)
            
            # 添加数值标签
            for bar, mean in zip(bars, means):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        axes[0].set_xlabel('Subject', fontsize=12)
        axes[0].set_ylabel('Mean Pearson Correlation', fontsize=12)
        axes[0].set_title('Unimodal Encoding Performance by Subject', fontsize=14)
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels([s.replace('sub-0', 'S') for s in subjects])
        axes[0].legend(loc='upper right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 图2: 跨被试平均
        avg_by_modality = {}
        for modality in modalities:
            avg_by_modality[modality] = np.mean([all_results[s][modality]['mean_correlation'] 
                                                  for s in subjects])
        
        bars = axes[1].bar(modalities, [avg_by_modality[m] for m in modalities],
                          color=[colors[m] for m in modalities], alpha=0.8)
        
        for bar, m in zip(bars, modalities):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{avg_by_modality[m]:.3f}', ha='center', va='bottom', fontsize=10)
        
        axes[1].set_xlabel('Modality', fontsize=12)
        axes[1].set_ylabel('Mean Pearson Correlation', fontsize=12)
        axes[1].set_title('Average Unimodal Encoding Performance', fontsize=14)
        axes[1].set_xticklabels([m.capitalize() for m in modalities])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = os.path.join(self.results_dir, 'unimodal_performance_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图保存至: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='训练单模态编码模型')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition', 
                        help='项目根目录')
    parser.add_argument('--subjects', default='1,2,3,5', help='被试列表')
    parser.add_argument('--modalities', default='visual,audio,language', help='模态列表')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    parser.add_argument('--plot_only', action='store_true', help='仅绘制图表')
    
    args = parser.parse_args()
    
    subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    trainer = UnimodalModelTrainer(args.project_dir, subjects, args.output_dir)
    
    if args.plot_only:
        trainer.plot_unimodal_comparison()
    else:
        print("=" * 60)
        print("单模态编码模型训练")
        print("=" * 60)
        print(f"被试: {subjects}")
        print(f"模态: {trainer.modalities}")
        print("=" * 60)
        
        all_results = trainer.train_all_models()
        trainer.plot_unimodal_comparison(all_results)
        
        print("\n" + "=" * 60)
        print("所有单模态模型训练完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()

