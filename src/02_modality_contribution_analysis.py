"""
Step 2: 模态贡献度分析
Modality Contribution Analysis for Cross-Modal Integration Study

分析每个脑区对不同模态的敏感性，计算：
1. 各脑区的优势模态
2. 多模态组合的增益效应
3. 模态特异性指数
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel
from tqdm import tqdm
import json

# Schaefer 1000分区的7个功能网络 (verified from nilearn Schaefer 2018 atlas)
SCHAEFER_7_NETWORKS = {
    'Visual': list(range(0, 81)) + list(range(500, 581)),              # 162 视觉网络
    'Somatomotor': list(range(81, 172)) + list(range(581, 684)),       # 194 躯体运动网络
    'DorsalAttention': list(range(172, 233)) + list(range(684, 745)),  # 122 背侧注意力网络
    'VentralAttention': list(range(233, 288)) + list(range(745, 811)), # 121 腹侧注意力网络
    'Limbic': list(range(288, 317)) + list(range(811, 842)),           # 60 边缘系统
    'Frontoparietal': list(range(317, 374)) + list(range(842, 912)) + [998, 999],  # 129 额顶网络
    'Default': list(range(374, 500)) + list(range(912, 998))           # 212 默认模式网络
}

# 网络颜色定义
NETWORK_COLORS = {
    'Visual': '#9B59B6',
    'Somatomotor': '#3498DB', 
    'DorsalAttention': '#2ECC71',
    'VentralAttention': '#F39C12',
    'Limbic': '#E74C3C',
    'Frontoparietal': '#1ABC9C',
    'Default': '#E91E63'
}

# Softer modality colors for consistency with project colormap
MODALITY_COLORS = {
    'visual': '#EF6548',    # Softer red (OrRd - lighter shade)
    'audio': '#74A9CF',     # Softer blue (PuBu - lighter shade)
    'language': '#66C2A4'   # Softer green (BuGn - lighter shade)
}


class ModalityContributionAnalyzer:
    """模态贡献度分析器"""
    
    def __init__(self, project_dir, subjects=[1, 2, 3, 5], input_dir=None, output_dir=None):
        self.project_dir = project_dir
        self.subjects = subjects
        self.modalities = ['visual', 'audio', 'language']
        
        self.unimodal_dir = input_dir or os.path.join(project_dir, 'analysis', 'results', 'unimodal_models')
        self.results_dir = output_dir or os.path.join(project_dir, 'analysis', 'results', 'modality_contribution')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.num_regions = 1000
    
    def load_unimodal_results(self):
        """加载单模态模型的相关系数结果"""
        results = {}
        
        for subject in self.subjects:
            results[f'sub-0{subject}'] = {}
            for modality in self.modalities:
                model_path = os.path.join(self.unimodal_dir, 
                                          f'ridge_model_sub-0{subject}_modality-{modality}.npy')
                if os.path.exists(model_path):
                    model_data = np.load(model_path, allow_pickle=True).item()
                    results[f'sub-0{subject}'][modality] = model_data['correlations']
                else:
                    print(f"警告: 未找到 {model_path}")
                    results[f'sub-0{subject}'][modality] = np.zeros(self.num_regions)
        
        return results
    
    def load_multimodal_results(self):
        """加载全模态模型的相关系数结果"""
        results = {}
        
        for subject in self.subjects:
            # 加载Ridge全模态模型
            model_path = os.path.join(self.project_dir, 'results', 'trained_encoding_models',
                                      f'trained_encoding_model_sub-0{subject}_modality-all.npy')
            if os.path.exists(model_path):
                model_data = np.load(model_path, allow_pickle=True).item()
                
                # 我们需要计算相关系数，这需要重新加载数据并预测
                # 这里假设已有预先计算的结果，或从其他地方获取
                results[f'sub-0{subject}'] = None  # 需要单独计算
            else:
                print(f"警告: 未找到 {model_path}")
        
        return results
    
    def compute_dominant_modality(self, unimodal_results):
        """计算每个脑区的优势模态"""
        dominant_modality = {}
        
        for subject_key in unimodal_results:
            correlations = np.zeros((self.num_regions, len(self.modalities)))
            
            for i, modality in enumerate(self.modalities):
                correlations[:, i] = unimodal_results[subject_key][modality]
            
            # 找出每个脑区相关系数最高的模态
            dominant_idx = np.argmax(correlations, axis=1)
            dominant_modality[subject_key] = {
                'dominant_idx': dominant_idx,
                'dominant_names': [self.modalities[i] for i in dominant_idx],
                'correlations': correlations
            }
        
        return dominant_modality
    
    def compute_modality_specificity_index(self, unimodal_results):
        """
        计算模态特异性指数 (MSI)
        MSI = (r_max - r_mean_others) / (r_max + r_mean_others)
        
        值越接近1表示该脑区对特定模态高度特异
        值越接近0表示该脑区对多模态响应较均匀
        """
        msi_results = {}
        
        for subject_key in unimodal_results:
            correlations = np.zeros((self.num_regions, len(self.modalities)))
            for i, modality in enumerate(self.modalities):
                correlations[:, i] = unimodal_results[subject_key][modality]
            
            # 确保所有值为正（相关系数可能为负）
            correlations = np.maximum(correlations, 0)
            
            msi = np.zeros(self.num_regions)
            for r in range(self.num_regions):
                r_max = np.max(correlations[r])
                other_mask = np.ones(len(self.modalities), dtype=bool)
                other_mask[np.argmax(correlations[r])] = False
                r_mean_others = np.mean(correlations[r, other_mask])
                
                if r_max + r_mean_others > 0:
                    msi[r] = (r_max - r_mean_others) / (r_max + r_mean_others)
                else:
                    msi[r] = 0
            
            msi_results[subject_key] = msi
        
        return msi_results
    
    def analyze_network_modality_sensitivity(self, unimodal_results):
        """分析各功能网络对不同模态的敏感性"""
        network_sensitivity = {}
        
        for subject_key in unimodal_results:
            network_sensitivity[subject_key] = {}
            
            for network_name, region_indices in SCHAEFER_7_NETWORKS.items():
                # 过滤有效索引
                valid_indices = [i for i in region_indices if i < self.num_regions]
                
                network_sensitivity[subject_key][network_name] = {}
                for modality in self.modalities:
                    correlations = unimodal_results[subject_key][modality]
                    mean_corr = np.mean(correlations[valid_indices])
                    std_corr = np.std(correlations[valid_indices])
                    network_sensitivity[subject_key][network_name][modality] = {
                        'mean': float(mean_corr),
                        'std': float(std_corr)
                    }
        
        return network_sensitivity
    
    def compute_multimodal_gain(self, unimodal_results, multimodal_correlations):
        """
        计算多模态增益
        Gain = r(V+A+L) - max(r(V), r(A), r(L))
        """
        gain_results = {}
        
        for subject_key in unimodal_results:
            if multimodal_correlations is None or subject_key not in multimodal_correlations:
                continue
            
            # 单模态最大值
            uni_max = np.maximum.reduce([
                unimodal_results[subject_key][m] for m in self.modalities
            ])
            
            # 多模态增益
            multi_corr = multimodal_correlations[subject_key]
            gain = multi_corr - uni_max
            
            gain_results[subject_key] = {
                'gain': gain,
                'mean_gain': float(np.mean(gain)),
                'positive_gain_ratio': float(np.mean(gain > 0)),
                'superadditive_regions': int(np.sum(gain > 0))
            }
        
        return gain_results
    
    def generate_analysis_report(self, unimodal_results, dominant_modality, 
                                  msi_results, network_sensitivity):
        """生成分析报告"""
        report = []
        report.append("=" * 70)
        report.append("模态贡献度分析报告 - Modality Contribution Analysis Report")
        report.append("=" * 70)
        report.append("")
        
        # 1. 优势模态分布
        report.append("1. 脑区优势模态分布")
        report.append("-" * 50)
        
        for subject_key in unimodal_results:
            dominant = dominant_modality[subject_key]['dominant_names']
            counts = {m: dominant.count(m) for m in self.modalities}
            report.append(f"\n{subject_key}:")
            for m, c in counts.items():
                report.append(f"  {m.capitalize():10}: {c:4} regions ({c/self.num_regions*100:.1f}%)")
        
        # 2. 模态特异性指数
        report.append("\n\n2. 模态特异性指数 (MSI)")
        report.append("-" * 50)
        
        for subject_key in msi_results:
            msi = msi_results[subject_key]
            report.append(f"\n{subject_key}:")
            report.append(f"  Mean MSI: {np.mean(msi):.4f}")
            report.append(f"  High specificity (MSI > 0.5): {np.sum(msi > 0.5)} regions")
            report.append(f"  Low specificity (MSI < 0.2): {np.sum(msi < 0.2)} regions")
        
        # 3. 网络级别敏感性
        report.append("\n\n3. 功能网络-模态敏感性")
        report.append("-" * 50)
        
        # 跨被试平均
        avg_network_sensitivity = {}
        for network_name in SCHAEFER_7_NETWORKS:
            avg_network_sensitivity[network_name] = {}
            for modality in self.modalities:
                means = [network_sensitivity[s][network_name][modality]['mean'] 
                        for s in network_sensitivity]
                avg_network_sensitivity[network_name][modality] = np.mean(means)
        
        report.append("\n跨被试平均相关系数:")
        header = f"{'Network':<20}" + "".join([f"{m.capitalize():<12}" for m in self.modalities])
        report.append(header)
        report.append("-" * 56)
        
        for network_name in SCHAEFER_7_NETWORKS:
            values = [avg_network_sensitivity[network_name][m] for m in self.modalities]
            best_modality = self.modalities[np.argmax(values)]
            row = f"{network_name:<20}" + "".join([f"{v:.4f}{'*' if m==best_modality else ' ':<11}" 
                                                    for m, v in zip(self.modalities, values)])
            report.append(row)
        
        report.append("\n* 表示该网络的优势模态")
        
        # 保存报告
        report_text = "\n".join(report)
        report_path = os.path.join(self.results_dir, 'modality_contribution_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n报告保存至: {report_path}")
        
        return report_text
    
    def plot_network_modality_heatmap(self, network_sensitivity):
        """绘制网络-模态敏感性热图 - Professional academic style"""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec
        import seaborn as sns
        
        # BuGn colors for consistency with project
        BUGN_COLORS = ['#F7FCFD', '#E5F5F9', '#CCECE6', '#99D8C9', '#66C2A4', '#41AE76', '#238B45', '#006D2C', '#00441B']
        
        # 跨被试平均
        networks = list(SCHAEFER_7_NETWORKS.keys())
        
        data = np.zeros((len(networks), len(self.modalities)))
        std_data = np.zeros((len(networks), len(self.modalities)))
        
        for i, network in enumerate(networks):
            for j, modality in enumerate(self.modalities):
                means = [network_sensitivity[s][network][modality]['mean'] 
                        for s in network_sensitivity]
                data[i, j] = np.mean(means)
                std_data[i, j] = np.std(means)
        
        # Create professional figure with multiple panels
        fig = plt.figure(figsize=(14, 8), facecolor='white')
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.35)
        
        # Panel A: Enhanced heatmap with seaborn
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('white')
        
        # Use BuGn colormap for consistency
        cmap_bugn = LinearSegmentedColormap.from_list('BuGn', BUGN_COLORS, N=256)
        
        # Create heatmap with seaborn for better styling
        sns.heatmap(data, ax=ax1, cmap=cmap_bugn, 
                   annot=True, fmt='.3f', annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                   linewidths=2, linecolor='white',
                   cbar_kws={'label': 'Encoding Accuracy (r)', 'shrink': 0.8},
                   vmin=0, vmax=0.35)
        
        ax1.set_xticklabels([m.capitalize() for m in self.modalities], fontsize=12, fontweight='bold')
        ax1.set_yticklabels(networks, fontsize=11, rotation=0)
        ax1.set_xlabel('Stimulus Modality', fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_ylabel('Brain Network', fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_title('A. Network-Modality Sensitivity Matrix', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        
        # Panel B: Grouped bar chart showing modality comparison per network
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('white')
        
        x = np.arange(len(networks))
        width = 0.25
        
        for i, modality in enumerate(self.modalities):
            offset = (i - 1) * width
            bars = ax2.bar(x + offset, data[:, i], width, 
                          label=modality.capitalize(),
                          color=MODALITY_COLORS[modality], alpha=0.8,
                          edgecolor='white', linewidth=1)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([n[:5] for n in networks], fontsize=10, rotation=45, ha='right')
        ax2.set_ylabel('Encoding Accuracy (r)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Brain Network', fontsize=12, fontweight='bold', labelpad=10)
        ax2.set_title('B. Modality Comparison by Network', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax2.set_ylim(0, 0.4)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add horizontal line for reference
        ax2.axhline(y=np.mean(data), color='#E74C3C', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='Mean')
        
        fig.suptitle('Network-Modality Encoding Analysis',
                    fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e')
        
        # Save in both formats
        fig_path_png = os.path.join(self.results_dir, 'network_modality_heatmap.png')
        fig_path_svg = os.path.join(self.results_dir, 'network_modality_heatmap.svg')
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(fig_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"热图保存至: {fig_path_png}")
    
    def plot_dominant_modality_distribution(self, dominant_modality):
        """绘制优势模态分布图 - Single panel brain region strip visualization"""
        from matplotlib.patches import Patch
        
        subjects = list(dominant_modality.keys())
        n_subjects = len(subjects)
        
        # Single panel figure - brain region strip visualization
        fig, ax3 = plt.subplots(figsize=(16, 4), facecolor='#FAFAFA')
        ax3.set_facecolor('#F8F9FA')
        
        # Create matrix: subjects x regions
        matrix = np.zeros((n_subjects, self.num_regions))
        
        for i, subject_key in enumerate(subjects):
            matrix[i, :] = dominant_modality[subject_key]['dominant_idx']
        
        # Create custom colormap for 3 modalities with softer colors
        from matplotlib.colors import ListedColormap, to_rgba
        # Add alpha to colors for softer appearance
        soft_colors = [to_rgba(MODALITY_COLORS[m], alpha=0.75) for m in self.modalities]
        cmap = ListedColormap(soft_colors)
        
        im = ax3.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)
        
        # Full network names mapping
        network_full_names = {
            'Visual': 'Visual',
            'Somatomotor': 'Somatomotor', 
            'DorsalAttention': 'DorsalAttn',
            'VentralAttention': 'VentralAttn',
            'Limbic': 'Limbic',
            'Frontoparietal': 'Frontoparietal',
            'Default': 'Default'
        }
        
        # Add network boundaries and labels below the plot
        network_names = list(SCHAEFER_7_NETWORKS.keys())
        prev_end = 0
        for network in network_names:
            indices = SCHAEFER_7_NETWORKS[network]
            if len(indices) > 0:
                start = min(indices)
                if start > prev_end:
                    ax3.axvline(x=start-0.5, color='white', linewidth=2, alpha=0.8)
                # Add network label with full name and lighter color
                mid = (start + max(indices)) / 2
                ax3.text(mid, n_subjects + 0.3, network_full_names.get(network, network[:4]), 
                        ha='center', va='top', fontsize=8, fontweight='bold', 
                        color='#999999', rotation=0)
                prev_end = max(indices)
        
        ax3.set_yticks(range(n_subjects))
        ax3.set_yticklabels([s.replace('sub-0', 'S') for s in subjects], fontsize=11)
        ax3.set_xlabel('Brain Regions (1000 Schaefer Parcellations)', fontsize=11, fontweight='bold', labelpad=25)
        ax3.set_ylabel('Subject', fontsize=11, fontweight='bold')
        ax3.set_title('Dominant Modality Across Brain Regions', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        
        # Adjust x-axis limits to show network labels
        ax3.set_xlim(-0.5, self.num_regions - 0.5)
        ax3.set_ylim(n_subjects - 0.5, -0.5)
        
        # Add legend outside plot area (to the right)
        legend_elements = [Patch(facecolor=MODALITY_COLORS[m], edgecolor='white', 
                                linewidth=1, label=m.capitalize(), alpha=0.8) 
                          for m in self.modalities]
        ax3.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=10, framealpha=0.95, ncol=1)
        
        plt.tight_layout()
        
        # Save in both formats
        fig_path_png = os.path.join(self.results_dir, 'dominant_modality_distribution.png')
        fig_path_svg = os.path.join(self.results_dir, 'dominant_modality_distribution.svg')
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
        plt.savefig(fig_path_svg, format='svg', bbox_inches='tight', facecolor='#FAFAFA')
        plt.close()
        
        print(f"分布图保存至: {fig_path_png}, {fig_path_svg}")
    
    def plot_modality_specificity(self, msi_results, dominant_modality):
        """绘制模态特异性分析图 - Professional academic style"""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec
        import seaborn as sns
        
        # BuGn colors for consistency
        BUGN_COLORS = ['#F7FCFD', '#E5F5F9', '#CCECE6', '#99D8C9', '#66C2A4', '#41AE76', '#238B45', '#006D2C', '#00441B']
        cmap_bugn = LinearSegmentedColormap.from_list('BuGn', BUGN_COLORS, N=256)
        
        fig = plt.figure(figsize=(16, 6), facecolor='white')
        gs = GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[1.2, 1, 1])
        
        # Panel A: Enhanced KDE plot instead of histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('white')
        
        # Use KDE for smoother distribution visualization
        subject_colors = ['#66C2A4', '#41AE76', '#238B45', '#006D2C']
        for i, subject_key in enumerate(msi_results):
            data = msi_results[subject_key]
            # KDE plot
            sns.kdeplot(data, ax=ax1, label=subject_key.replace('sub-0', 'Subject '), 
                       color=subject_colors[i % len(subject_colors)], linewidth=2.5, alpha=0.8)
            # Add fill
            sns.kdeplot(data, ax=ax1, color=subject_colors[i % len(subject_colors)], 
                       linewidth=0, fill=True, alpha=0.15)
        
        ax1.axvline(x=0.5, color='#E74C3C', linestyle='--', linewidth=2, 
                   alpha=0.8, label='High specificity')
        ax1.set_xlabel('Modality Specificity Index (MSI)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title('A. MSI Distribution Across Subjects', fontsize=13, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax1.set_xlim(0, 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Panel B: Horizontal bar chart with gradient colors
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('white')
        
        network_msi = {}
        network_std = {}
        for network, indices in SCHAEFER_7_NETWORKS.items():
            valid_indices = [i for i in indices if i < self.num_regions]
            all_msi = []
            for subject_key in msi_results:
                all_msi.extend(msi_results[subject_key][valid_indices])
            network_msi[network] = np.mean(all_msi)
            network_std[network] = np.std(all_msi)
        
        networks = list(network_msi.keys())
        msi_values = [network_msi[n] for n in networks]
        std_values = [network_std[n] for n in networks]
        
        # Sort by MSI value
        sorted_idx = np.argsort(msi_values)[::-1]
        networks_sorted = [networks[i] for i in sorted_idx]
        msi_sorted = [msi_values[i] for i in sorted_idx]
        
        # Create gradient colors based on MSI value
        norm_msi = [(v - min(msi_sorted)) / (max(msi_sorted) - min(msi_sorted) + 1e-10) for v in msi_sorted]
        bar_colors = [cmap_bugn(0.3 + 0.6 * v) for v in norm_msi]
        
        y_pos = np.arange(len(networks_sorted))
        bars = ax2.barh(y_pos, msi_sorted, color=bar_colors, alpha=0.85,
                       edgecolor='white', linewidth=1.5, height=0.7)
        
        # Add value labels
        for bar, val in zip(bars, msi_sorted):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(networks_sorted, fontsize=11)
        ax2.set_xlabel('Mean MSI', fontsize=12, fontweight='bold')
        ax2.set_title('B. MSI by Network (Ranked)', fontsize=13, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax2.set_xlim(0, 0.45)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Panel C: Violin plot showing MSI distribution per network
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor('white')
        
        # Prepare data for violin plot
        violin_data = []
        violin_labels = []
        for network in networks:
            indices = SCHAEFER_7_NETWORKS[network]
            valid_indices = [i for i in indices if i < self.num_regions]
            for subject_key in msi_results:
                violin_data.extend(msi_results[subject_key][valid_indices])
                violin_labels.extend([network[:4]] * len(valid_indices))
        
        # Create simplified violin data
        network_data = {}
        for network in networks:
            indices = SCHAEFER_7_NETWORKS[network]
            valid_indices = [i for i in indices if i < self.num_regions]
            all_vals = []
            for subject_key in msi_results:
                all_vals.extend(msi_results[subject_key][valid_indices])
            network_data[network] = all_vals
        
        positions = range(len(networks))
        violin_parts = ax3.violinplot([network_data[n] for n in networks], 
                                      positions=positions, showmeans=True, showextrema=False)
        
        # Style the violins with BuGn colors
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(cmap_bugn(0.4 + 0.4 * i / len(networks)))
            pc.set_edgecolor('white')
            pc.set_alpha(0.7)
            pc.set_linewidth(1.5)
        
        violin_parts['cmeans'].set_color('#E74C3C')
        violin_parts['cmeans'].set_linewidth(2)
        
        ax3.set_xticks(positions)
        ax3.set_xticklabels([n[:4] for n in networks], fontsize=10, rotation=45, ha='right')
        ax3.set_ylabel('MSI', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Brain Network', fontsize=12, fontweight='bold')
        ax3.set_title('C. MSI Distribution per Network', fontsize=13, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax3.set_ylim(0, 1)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        fig.suptitle('Modality Specificity Analysis',
                    fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e')
        
        # Save in both formats
        fig_path_png = os.path.join(self.results_dir, 'modality_specificity_analysis.png')
        fig_path_svg = os.path.join(self.results_dir, 'modality_specificity_analysis.svg')
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(fig_path_svg, format='svg', bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"特异性分析图保存至: {fig_path_png}")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("=" * 60)
        print("模态贡献度分析")
        print("=" * 60)
        
        # 1. 加载单模态结果
        print("\n1. 加载单模态模型结果...")
        unimodal_results = self.load_unimodal_results()
        
        # 2. 计算优势模态
        print("2. 计算各脑区优势模态...")
        dominant_modality = self.compute_dominant_modality(unimodal_results)
        
        # 3. 计算模态特异性指数
        print("3. 计算模态特异性指数...")
        msi_results = self.compute_modality_specificity_index(unimodal_results)
        
        # 4. 分析网络-模态敏感性
        print("4. 分析功能网络-模态敏感性...")
        network_sensitivity = self.analyze_network_modality_sensitivity(unimodal_results)
        
        # 5. 生成报告
        print("5. 生成分析报告...")
        self.generate_analysis_report(unimodal_results, dominant_modality, 
                                       msi_results, network_sensitivity)
        
        # 6. 绘制图表
        print("6. 绘制分析图表...")
        self.plot_network_modality_heatmap(network_sensitivity)
        self.plot_dominant_modality_distribution(dominant_modality)
        self.plot_modality_specificity(msi_results, dominant_modality)
        
        # 保存结果
        results_data = {
            'network_sensitivity': network_sensitivity,
            'msi_summary': {s: {'mean': float(np.mean(msi_results[s])), 
                               'std': float(np.std(msi_results[s]))} 
                          for s in msi_results}
        }
        
        np.save(os.path.join(self.results_dir, 'modality_contribution_results.npy'), 
                results_data)
        
        print("\n" + "=" * 60)
        print("模态贡献度分析完成！")
        print(f"结果保存目录: {self.results_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='模态贡献度分析')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition',
                        help='项目根目录')
    parser.add_argument('--subjects', default='1,2,3,5', help='被试列表')
    parser.add_argument('--input_dir', default=None, help='单模态模型输入目录')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    
    args = parser.parse_args()
    subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    analyzer = ModalityContributionAnalyzer(args.project_dir, subjects, args.input_dir, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

