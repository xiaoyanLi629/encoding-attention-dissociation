"""
Step 3: 跨模态注意力分析
Cross-Modal Attention Analysis

分析PersonalizedMultiModalNetwork模型中的跨模态注意力机制：
1. 模态权重分析
2. 跨模态注意力矩阵可视化
3. 注意力随时间/内容的动态变化
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import h5py
import json

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official_code', '02_encoding_model_training'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'official_code', '02_encoding_model_training', 'models'))

# BuGn-based modality colors for consistency with project colormap
MODALITY_COLORS = {
    'visual': '#005824',    # Darkest BuGn
    'audio': '#41AE76',     # Medium BuGn
    'language': '#99D8C9'   # Light BuGn
}

# Configurable figure filenames
FIGURE_NAMES = {
    'modality_weights': 'fig_crossmodal_attention_weights',
}


class CrossModalAttentionAnalyzer:
    """跨模态注意力分析器"""
    
    def __init__(self, project_dir, subjects=[1, 2, 3, 5], output_dir=None):
        self.project_dir = project_dir
        self.subjects = subjects
        self.modalities = ['visual', 'audio', 'language']
        
        # 模型可能在两个位置，优先检查competition目录
        self.models_dir = os.path.join(project_dir, 'competition', 'results', 'trained_encoding_models')
        if not os.path.exists(self.models_dir):
            self.models_dir = os.path.join(project_dir, 'results', 'trained_encoding_models')
        self.results_dir = output_dir or os.path.join(project_dir, 'analysis', 'results', 'crossmodal_attention')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # HRF参数
        self.excluded_samples_start = 5
        self.excluded_samples_end = 5
        self.hrf_delay = 3
        self.stimulus_window = 5
    
    def create_personalized_model(self, input_dim, output_dim, num_subjects=4, hidden_dim=512):
        """创建PersonalizedMultiModalNetwork模型结构"""
        
        class PersonalizedMultiModalNetwork(nn.Module):
            def __init__(self, input_dim, output_dim, num_subjects=4, hidden_dim=512):
                super(PersonalizedMultiModalNetwork, self).__init__()
                
                self.visual_dim = input_dim // 3
                self.audio_dim = input_dim // 3
                self.language_dim = input_dim - 2 * (input_dim // 3)
                
                self.visual_encoder = nn.Sequential(
                    nn.Linear(self.visual_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                
                self.audio_encoder = nn.Sequential(
                    nn.Linear(self.audio_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                
                self.language_encoder = nn.Sequential(
                    nn.Linear(self.language_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                
                self.cross_modal_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
                )
                
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
        
        return PersonalizedMultiModalNetwork(input_dim, output_dim, num_subjects, hidden_dim)
    
    def load_model(self, subject):
        """加载训练好的PersonalizedMultiModalNetwork模型"""
        model_path = os.path.join(self.models_dir, 
                                   f'personalized_multimodal_model_sub-0{subject}_modality-all.pth')
        
        if not os.path.exists(model_path):
            print(f"警告: 未找到模型 {model_path}")
            return None
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从checkpoint获取模型维度信息
        # 需要根据实际保存的checkpoint结构调整
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 推断输入输出维度
        for key in state_dict:
            if 'visual_encoder.0.weight' in key:
                visual_dim = state_dict[key].shape[1]
                hidden_dim = state_dict[key].shape[0]
                break
        
        for key in state_dict:
            if 'fmri_generator' in key and 'weight' in key:
                if len(state_dict[key].shape) == 2:
                    output_dim = state_dict[key].shape[0]
                    break
        
        # 估算input_dim
        input_dim = visual_dim * 3  # 假设三个模态维度相近
        
        model = self.create_personalized_model(input_dim, output_dim)
        
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            return None
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_stimulus_features(self, movie_split_names, movie_split_samples):
        """加载所有模态的刺激特征"""
        features_dir = os.path.join(self.project_dir, 'data', 'features',
                                     'official_stimulus_features', 'pca', 'friends_movie10')
        
        # 加载三个模态
        features = {}
        for modality in self.modalities:
            feature_file = os.path.join(features_dir, modality, 'features_train.npy')
            features[modality] = np.load(feature_file, allow_pickle=True).item()
        
        # 组合特征
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
                            idx_start = s + self.excluded_samples_start - self.hrf_delay - self.stimulus_window + 1
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
    
    def load_fmri(self, subject):
        """加载fMRI数据"""
        fmri_dir = os.path.join(self.project_dir, 'data', 'fmri', f'sub-0{subject}', 'func')
        
        fmri_file_friends = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
        fmri_file_movie10 = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
        
        fmri_friends = h5py.File(os.path.join(fmri_dir, fmri_file_friends), 'r')
        fmri_movie10 = h5py.File(os.path.join(fmri_dir, fmri_file_movie10), 'r')
        
        fmri = []
        movie_split_names = []
        movie_split_samples = []
        
        for key, val in fmri_friends.items():
            fmri_part = val[self.excluded_samples_start:-self.excluded_samples_end]
            fmri.append(fmri_part)
            movie_split_names.append(key[13:])
            movie_split_samples.append(len(fmri_part))
        
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
    
    def extract_attention_weights(self, model, X_data, batch_size=256):
        """提取模型的注意力权重"""
        model.eval()
        
        all_attention_weights = []
        all_modality_weights = []
        
        n_samples = len(X_data)
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = torch.FloatTensor(X_data[i:i+batch_size]).to(self.device)
                
                # 获取注意力权重
                _, attention_weights, modality_weights = model(batch_X, return_attention=True)
                
                all_attention_weights.append(attention_weights.cpu().numpy())
                all_modality_weights.append(modality_weights.cpu().numpy())
        
        attention_weights = np.concatenate(all_attention_weights, axis=0)
        modality_weights = all_modality_weights[0]  # 模态权重是固定的
        
        return attention_weights, modality_weights
    
    def analyze_modality_weights(self):
        """分析各被试模型的模态权重"""
        modality_weights_all = {}
        
        for subject in self.subjects:
            model_path = os.path.join(self.models_dir,
                                       f'personalized_multimodal_model_sub-0{subject}_modality-all.pth')
            
            if not os.path.exists(model_path):
                print(f"警告: 未找到模型 {model_path}")
                continue
            
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # 提取模态权重
            if 'modality_weights' in state_dict:
                weights = state_dict['modality_weights'].cpu().numpy()
                # 应用softmax
                weights = np.exp(weights) / np.sum(np.exp(weights))
                modality_weights_all[f'sub-0{subject}'] = weights
                print(f"Subject {subject} 模态权重: Visual={weights[0]:.4f}, Audio={weights[1]:.4f}, Language={weights[2]:.4f}")
        
        return modality_weights_all
    
    def plot_modality_weights_comparison(self, modality_weights_all):
        """绘制模态权重对比图 - Enhanced professional visualization"""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec
        
        if not modality_weights_all:
            print("没有可用的模态权重数据")
            return
        
        subjects = list(modality_weights_all.keys())
        
        # Create data matrix for heatmap
        weights_matrix = np.array([modality_weights_all[s] for s in subjects])
        
        # Create enhanced figure with professional styling
        fig = plt.figure(figsize=(14, 6), facecolor='#FAFAFA')
        gs = GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[1, 1])
        
        # BuGn colormap for consistency
        bugn_colors = ['#F7FCFD', '#E5F5F9', '#CCECE6', '#99D8C9', '#66C2A4', '#41AE76', '#238B45', '#005824']
        cmap_bugn = LinearSegmentedColormap.from_list('BuGn', bugn_colors, N=256)
        
        # Panel A: Professional heatmap
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor('#F8F9FA')
        
        im = ax1.imshow(weights_matrix, cmap=cmap_bugn, aspect='auto', vmin=0.25, vmax=0.45)
        
        # Add value annotations
        for i in range(len(subjects)):
            for j in range(len(self.modalities)):
                val = weights_matrix[i, j]
                color = 'white' if val > 0.36 else '#1a1a2e'
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)
        
        ax1.set_xticks(range(len(self.modalities)))
        ax1.set_xticklabels([m.capitalize() for m in self.modalities], fontsize=12, fontweight='bold')
        ax1.set_yticks(range(len(subjects)))
        ax1.set_yticklabels([s.replace('sub-0', 'Subject ') for s in subjects], fontsize=11)
        ax1.set_xlabel('Stimulus Modality', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_ylabel('Participant', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_title('A. Cross-Modal Attention Weights', fontsize=14,
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label('Attention Weight', fontsize=10, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Add reference line annotation
        ax1.axhline(y=-0.5, color='#E74C3C', linewidth=0, alpha=0)  # Placeholder
        
        # Panel B: Enhanced grouped bar chart
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('#F8F9FA')
        
        x = np.arange(len(subjects))
        width = 0.25
        
        for i, modality in enumerate(self.modalities):
            weights = [modality_weights_all[s][i] for s in subjects]
            bars = ax2.bar(x + i*width - width, weights, width, 
                          label=modality.capitalize(),
                          color=MODALITY_COLORS[modality], alpha=0.85,
                          edgecolor='white', linewidth=1.5)
            
            # Add value labels on bars
            for bar, w in zip(bars, weights):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f'{w:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add uniform weight reference line
        ax2.axhline(y=1/3, color='#E74C3C', linestyle='--', linewidth=2, 
                   alpha=0.8, label='Uniform (0.33)')
        
        ax2.set_xlabel('Participant', fontsize=12, fontweight='bold', labelpad=10)
        ax2.set_ylabel('Attention Weight', fontsize=12, fontweight='bold', labelpad=10)
        ax2.set_title('B. Subject-Level Comparison', fontsize=14,
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('sub-0', 'S') for s in subjects], fontsize=11)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, fancybox=True)
        ax2.set_ylim(0, 0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Main title
        fig.suptitle('Cross-Modal Attention Analysis',
                    fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e')
        
        # Save in both PNG and SVG formats
        fig_path_png = os.path.join(self.results_dir, f"{FIGURE_NAMES['modality_weights']}.png")
        fig_path_svg = os.path.join(self.results_dir, f"{FIGURE_NAMES['modality_weights']}.svg")
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
        plt.savefig(fig_path_svg, format='svg', bbox_inches='tight', facecolor='#FAFAFA')
        plt.close()
        
        print(f"模态权重对比图保存至: {fig_path_png}, {fig_path_svg}")
    
    def plot_attention_heatmap(self, attention_weights, subject):
        """绘制跨模态注意力热图"""
        # attention_weights: (n_samples, 3, 3) - 3个模态之间的注意力
        
        # 平均跨所有样本
        mean_attention = np.mean(attention_weights, axis=0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(mean_attention, 
                   xticklabels=[m.capitalize() for m in self.modalities],
                   yticklabels=[m.capitalize() for m in self.modalities],
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax, vmin=0, vmax=1)
        
        ax.set_xlabel('Key Modality', fontsize=12)
        ax.set_ylabel('Query Modality', fontsize=12)
        ax.set_title(f'Cross-Modal Attention Matrix\n(Subject {subject})', fontsize=14)
        
        plt.tight_layout()
        
        fig_path = os.path.join(self.results_dir, f'attention_heatmap_sub-0{subject}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力热图保存至: {fig_path}")
        
        return mean_attention
    
    def analyze_attention_dynamics(self, attention_weights, movie_split_samples):
        """分析注意力随时间的动态变化"""
        # 将注意力权重按电影片段分组
        split_attentions = []
        start_idx = 0
        
        for n_samples in movie_split_samples:
            split_attention = attention_weights[start_idx:start_idx + n_samples]
            split_attentions.append(split_attention)
            start_idx += n_samples
        
        return split_attentions
    
    def run_full_analysis(self):
        """运行完整的跨模态注意力分析"""
        print("=" * 60)
        print("跨模态注意力分析")
        print("=" * 60)
        
        # 1. 分析模态权重
        print("\n1. 分析模态权重...")
        modality_weights_all = self.analyze_modality_weights()
        
        # 2. 绘制模态权重对比图
        print("\n2. 绘制模态权重对比图...")
        self.plot_modality_weights_comparison(modality_weights_all)
        
        # 保存结果
        results = {
            'modality_weights': {k: v.tolist() for k, v in modality_weights_all.items()}
        }
        
        results_path = os.path.join(self.results_dir, 'crossmodal_attention_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append("跨模态注意力分析报告")
        report.append("=" * 60)
        report.append("")
        report.append("1. 模态权重分析")
        report.append("-" * 40)
        
        for subject, weights in modality_weights_all.items():
            report.append(f"\n{subject}:")
            for i, modality in enumerate(self.modalities):
                report.append(f"  {modality.capitalize()}: {weights[i]:.4f}")
            
            # 判断主导模态
            dominant_idx = np.argmax(weights)
            report.append(f"  主导模态: {self.modalities[dominant_idx].capitalize()}")
        
        # 跨被试分析
        report.append("\n\n2. 跨被试平均")
        report.append("-" * 40)
        
        if modality_weights_all:
            avg_weights = np.mean([w for w in modality_weights_all.values()], axis=0)
            for i, modality in enumerate(self.modalities):
                report.append(f"{modality.capitalize()}: {avg_weights[i]:.4f}")
        
        report_text = "\n".join(report)
        report_path = os.path.join(self.results_dir, 'crossmodal_attention_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n报告保存至: {report_path}")
        
        print("\n" + "=" * 60)
        print("跨模态注意力分析完成！")
        print(f"结果保存目录: {self.results_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='跨模态注意力分析')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition',
                        help='项目根目录')
    parser.add_argument('--subjects', default='1,2,3,5', help='被试列表')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    
    args = parser.parse_args()
    subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    analyzer = CrossModalAttentionAnalyzer(args.project_dir, subjects, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

