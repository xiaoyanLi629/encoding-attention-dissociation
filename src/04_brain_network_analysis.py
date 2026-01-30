"""
Step 4: 功能网络分析
Brain Network Analysis for Cross-Modal Integration Study

基于Schaefer 1000分区的7个功能网络进行分析：
1. 网络级别的模态敏感性
2. 网络内和网络间的跨模态整合
3. 多模态增益的网络分布
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, f_oneway
from scipy.cluster.hierarchy import dendrogram, linkage
import json
from tqdm import tqdm

# Schaefer 1000分区的7个功能网络定义
# 基于Schaefer 2018论文的分区方案
# Scientific colormap based on BuGn (Blue-Green) sequential colormap
# Colors ordered by MII hierarchy: Default > Limbic > Somatomotor > Frontoparietal > VentralAttention > DorsalAttention > Visual
SCHAEFER_7_NETWORKS = {
    'Visual': {
        'indices': list(range(0, 81)) + list(range(500, 581)),  # 162 regions
        'color': '#EDF8FB',  # BuGn lightest (lowest MII)
        'description': 'Primary and secondary visual cortex'
    },
    'Somatomotor': {
        'indices': list(range(81, 172)) + list(range(581, 684)),  # 194 regions
        'color': '#41AE76',  # BuGn medium-dark
        'description': 'Somatomotor and premotor cortex'
    },
    'DorsalAttention': {
        'indices': list(range(172, 233)) + list(range(684, 745)),  # 122 regions
        'color': '#CCECE6',  # BuGn light
        'description': 'Dorsal attention network (intraparietal sulcus, frontal eye fields)'
    },
    'VentralAttention': {
        'indices': list(range(233, 288)) + list(range(745, 811)),  # 121 regions
        'color': '#99D8C9',  # BuGn light-medium
        'description': 'Ventral attention/salience network'
    },
    'Limbic': {
        'indices': list(range(288, 317)) + list(range(811, 842)),  # 60 regions
        'color': '#238B45',  # BuGn dark (high MII)
        'description': 'Limbic system (orbitofrontal cortex, temporal pole)'
    },
    'Frontoparietal': {
        'indices': list(range(317, 374)) + list(range(842, 912)) + [998, 999],  # 129 regions
        'color': '#66C2A4',  # BuGn medium
        'description': 'Frontoparietal control network'
    },
    'Default': {
        'indices': list(range(374, 500)) + list(range(912, 998)),  # 212 regions
        'color': '#005824',  # BuGn darkest (highest MII)
        'description': 'Default mode network (mPFC, PCC, angular gyrus)'
    }
}

# Scientific sequential colormap for modalities (ColorBrewer-inspired)
MODALITY_COLORS = {
    'visual': '#D7301F',   # Reds sequential - dark red
    'audio': '#0570B0',    # Blues sequential - medium blue  
    'language': '#238443'  # Greens sequential - forest green
}

# Configurable figure filenames - adjust these to change output file names
FIGURE_NAMES = {
    'network_modality_profile': 'fig_network_modality_profile',
    'network_integration_hierarchy': 'fig_network_integration_hierarchy',
}


class BrainNetworkAnalyzer:
    """功能网络分析器"""
    
    def __init__(self, project_dir, subjects=[1, 2, 3, 5], input_dir=None, output_dir=None):
        self.project_dir = project_dir
        self.subjects = subjects
        self.modalities = ['visual', 'audio', 'language']
        self.networks = SCHAEFER_7_NETWORKS
        
        self.unimodal_dir = input_dir or os.path.join(project_dir, 'analysis', 'results', 'unimodal_models')
        self.results_dir = output_dir or os.path.join(project_dir, 'analysis', 'results', 'brain_networks')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.num_regions = 1000
    
    def load_unimodal_correlations(self):
        """加载单模态模型的相关系数"""
        correlations = {}
        
        for subject in self.subjects:
            correlations[f'sub-0{subject}'] = {}
            for modality in self.modalities:
                model_path = os.path.join(self.unimodal_dir,
                                          f'ridge_model_sub-0{subject}_modality-{modality}.npy')
                if os.path.exists(model_path):
                    model_data = np.load(model_path, allow_pickle=True).item()
                    correlations[f'sub-0{subject}'][modality] = model_data['correlations']
                else:
                    print(f"警告: 未找到 {model_path}")
                    correlations[f'sub-0{subject}'][modality] = np.zeros(self.num_regions)
        
        return correlations
    
    def compute_network_statistics(self, correlations):
        """计算各网络的统计信息"""
        network_stats = {}
        
        for subject_key in correlations:
            network_stats[subject_key] = {}
            
            for network_name, network_info in self.networks.items():
                indices = [i for i in network_info['indices'] if i < self.num_regions]
                network_stats[subject_key][network_name] = {}
                
                for modality in self.modalities:
                    region_corrs = correlations[subject_key][modality][indices]
                    
                    network_stats[subject_key][network_name][modality] = {
                        'mean': float(np.mean(region_corrs)),
                        'std': float(np.std(region_corrs)),
                        'max': float(np.max(region_corrs)),
                        'min': float(np.min(region_corrs)),
                        'n_high': int(np.sum(region_corrs > 0.3)),
                        'n_regions': len(indices)
                    }
        
        return network_stats
    
    def compute_network_modality_preference(self, network_stats):
        """计算各网络的模态偏好"""
        preferences = {}
        
        for subject_key in network_stats:
            preferences[subject_key] = {}
            
            for network_name in self.networks:
                modality_means = {m: network_stats[subject_key][network_name][m]['mean']
                                 for m in self.modalities}
                
                # 找出最佳模态
                best_modality = max(modality_means, key=modality_means.get)
                best_score = modality_means[best_modality]
                
                # 计算偏好强度（最佳模态与其他模态均值之差）
                other_mean = np.mean([v for k, v in modality_means.items() if k != best_modality])
                preference_strength = best_score - other_mean
                
                preferences[subject_key][network_name] = {
                    'best_modality': best_modality,
                    'best_score': best_score,
                    'preference_strength': preference_strength,
                    'all_scores': modality_means
                }
        
        return preferences
    
    def compute_multimodal_integration_index(self, correlations):
        """
        计算多模态整合指数 (MII)
        衡量每个网络中各模态表现的均衡程度
        MII = 1 - CV (变异系数越小，整合程度越高)
        """
        mii_results = {}
        
        for subject_key in correlations:
            mii_results[subject_key] = {}
            
            for network_name, network_info in self.networks.items():
                indices = [i for i in network_info['indices'] if i < self.num_regions]
                
                # 计算每个脑区的模态变异系数
                region_cv = []
                for idx in indices:
                    modality_corrs = [correlations[subject_key][m][idx] for m in self.modalities]
                    modality_corrs = np.maximum(modality_corrs, 0)  # 确保非负
                    
                    if np.mean(modality_corrs) > 0:
                        cv = np.std(modality_corrs) / np.mean(modality_corrs)
                    else:
                        cv = 0
                    region_cv.append(cv)
                
                # MII = 1 - 平均CV
                mean_cv = np.mean(region_cv)
                mii = 1 - min(mean_cv, 1)  # 限制在0-1之间
                
                mii_results[subject_key][network_name] = {
                    'mii': float(mii),
                    'mean_cv': float(mean_cv),
                    'interpretation': 'High integration' if mii > 0.7 else ('Moderate' if mii > 0.4 else 'Modality-specific')
                }
        
        return mii_results
    
    def statistical_analysis(self, network_stats):
        """进行统计检验"""
        stats_results = {}
        
        # 对每个网络，检验模态之间的差异
        for network_name in self.networks:
            stats_results[network_name] = {}
            
            # 收集所有被试的数据
            modality_data = {m: [] for m in self.modalities}
            for subject_key in network_stats:
                for modality in self.modalities:
                    modality_data[modality].append(
                        network_stats[subject_key][network_name][modality]['mean']
                    )
            
            # ANOVA检验
            f_stat, p_value = f_oneway(*[modality_data[m] for m in self.modalities])
            stats_results[network_name]['anova'] = {
                'F': float(f_stat),
                'p': float(p_value),
                'significant': p_value < 0.05
            }
            
            # 成对t检验
            pairs = [('visual', 'audio'), ('visual', 'language'), ('audio', 'language')]
            for m1, m2 in pairs:
                if len(modality_data[m1]) >= 2:
                    t_stat, p_val = ttest_ind(modality_data[m1], modality_data[m2])
                    stats_results[network_name][f'{m1}_vs_{m2}'] = {
                        't': float(t_stat),
                        'p': float(p_val),
                        'significant': p_val < 0.05
                    }
        
        return stats_results
    
    def plot_network_modality_profile(self, network_stats):
        """绘制网络-模态响应轮廓图 - Enhanced scientific visualization"""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec
        
        networks = list(self.networks.keys())
        
        # 跨被试平均
        avg_data = np.zeros((len(networks), len(self.modalities)))
        std_data = np.zeros((len(networks), len(self.modalities)))
        
        for i, network in enumerate(networks):
            for j, modality in enumerate(self.modalities):
                values = [network_stats[s][network][modality]['mean'] 
                         for s in network_stats]
                avg_data[i, j] = np.mean(values)
                std_data[i, j] = np.std(values)
        
        # Create enhanced figure with 2 panels
        fig = plt.figure(figsize=(16, 7), facecolor='#FAFAFA')
        gs = GridSpec(1, 2, figure=fig, wspace=0.3, width_ratios=[1.2, 1])
        
        # BuGn-based modality colors for consistency with project colormap
        modality_colors_bugn = {
            'visual': '#005824',    # Darkest BuGn
            'audio': '#41AE76',     # Medium BuGn
            'language': '#99D8C9'   # Light BuGn
        }
        
        # Panel A: Enhanced grouped bar chart with error bars
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor('#F8F9FA')
        
        x = np.arange(len(networks))
        width = 0.25
        
        for i, modality in enumerate(self.modalities):
            bars = ax1.bar(x + i*width - width, avg_data[:, i], width, 
                          yerr=std_data[:, i], capsize=4,
                          label=modality.capitalize(),
                          color=modality_colors_bugn[modality], alpha=0.85,
                          edgecolor='white', linewidth=1.5,
                          error_kw={'linewidth': 1.5, 'capthick': 1.5})
        
        ax1.set_xlabel('Brain Network', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_ylabel('Encoding Accuracy (Pearson r)', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_title('A. Network-Specific Modality Sensitivity', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n[:4] for n in networks], rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, fancybox=True)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylim(0, max(avg_data.max() + std_data.max() + 0.05, 0.35))
        
        # Panel B: Enhanced heatmap with BuGn sequential colormap
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('#F8F9FA')
        
        # Use raw data with BuGn sequential colormap for consistency
        # BuGn colormap from ColorBrewer (same as network colors)
        bugn_colors = ['#F7FCFD', '#E5F5F9', '#CCECE6', '#99D8C9', '#66C2A4', '#41AE76', '#238B45', '#005824']
        cmap_bugn = LinearSegmentedColormap.from_list('BuGn', bugn_colors, N=256)
        
        im = ax2.imshow(avg_data, cmap=cmap_bugn, aspect='auto', vmin=0, vmax=0.35)
        
        ax2.set_xticks(range(len(self.modalities)))
        ax2.set_xticklabels([m.capitalize() for m in self.modalities], fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(networks)))
        ax2.set_yticklabels([n[:5] for n in networks], fontsize=10)
        
        # Add value annotations with adaptive colors
        for i in range(len(networks)):
            for j in range(len(self.modalities)):
                val = avg_data[i, j]
                color = 'white' if val > 0.18 else '#1a1a2e'
                text = ax2.text(j, i, f'{val:.3f}',
                              ha='center', va='center', fontsize=10,
                              fontweight='bold', color=color)
        
        ax2.set_xlabel('Stimulus Modality', fontsize=12, fontweight='bold', labelpad=10)
        ax2.set_ylabel('Brain Network', fontsize=12, fontweight='bold', labelpad=15)
        ax2.set_title('B. Network-Modality Encoding', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        
        cbar = plt.colorbar(im, ax=ax2, shrink=0.7, aspect=20, pad=0.02)
        cbar.set_label('Encoding Accuracy (r)', fontsize=10, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Main title
        fig.suptitle('Network-Modality Sensitivity Analysis',
                    fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e')
        
        fig_path_png = os.path.join(self.results_dir, f"{FIGURE_NAMES['network_modality_profile']}.png")
        fig_path_svg = os.path.join(self.results_dir, f"{FIGURE_NAMES['network_modality_profile']}.svg")
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
        plt.savefig(fig_path_svg, format='svg', bbox_inches='tight', facecolor='#FAFAFA')
        plt.close()
        
        print(f"网络-模态轮廓图保存至: {fig_path_png}, {fig_path_svg}")
    
    def plot_network_hierarchy(self, mii_results):
        """绘制网络层级结构图 - Enhanced scientific visualization"""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as mpatches
        
        networks = list(self.networks.keys())
        
        # 跨被试平均
        avg_mii = []
        std_mii = []
        all_subject_mii = {n: [] for n in networks}
        
        for network in networks:
            values = [mii_results[s][network]['mii'] for s in mii_results]
            avg_mii.append(np.mean(values))
            std_mii.append(np.std(values))
            all_subject_mii[network] = values
        
        # Create enhanced figure with 2 panels
        fig = plt.figure(figsize=(14, 7), facecolor='#FAFAFA')
        gs = GridSpec(1, 2, figure=fig, wspace=0.3, width_ratios=[1, 1])
        
        # Panel A: Horizontal bar chart sorted by MII (more elegant)
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor('#F8F9FA')
        
        # Sort by MII value
        sorted_indices = np.argsort(avg_mii)
        sorted_networks = [networks[i] for i in sorted_indices]
        sorted_mii = [avg_mii[i] for i in sorted_indices]
        sorted_std = [std_mii[i] for i in sorted_indices]
        sorted_colors = [self.networks[n]['color'] for n in sorted_networks]
        
        y_pos = np.arange(len(networks))
        
        # Draw bars with gradient effect
        bars = ax1.barh(y_pos, sorted_mii, xerr=sorted_std, 
                       color=sorted_colors, alpha=0.85,
                       edgecolor='white', linewidth=1.5, height=0.7,
                       capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})
        
        # Add value labels
        for i, (bar, val, std) in enumerate(zip(bars, sorted_mii, sorted_std)):
            ax1.text(val + std + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10, fontweight='bold',
                    color='#1a1a2e')
        
        # Add threshold lines
        ax1.axvline(0.5, color='#E74C3C', linestyle='--', linewidth=2.5, 
                   alpha=0.8, label='Integration threshold (0.5)')
        ax1.axvline(0.7, color='#27AE60', linestyle=':', linewidth=2, 
                   alpha=0.8, label='High integration (0.7)')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_networks, fontsize=11)
        ax1.set_xlabel('Multimodal Integration Index (MII)', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_title('A. Integration Hierarchy (Ranked)', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax1.set_xlim(0, 1.0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, 
                  bbox_to_anchor=(0.96, 0.20))
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Panel B: Circular hierarchy diagram
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor('#F8F9FA')
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        
        # Create circular layout with MII as radius
        n_networks = len(networks)
        angles = np.linspace(0, 2*np.pi, n_networks, endpoint=False)
        
        # Draw concentric circles for reference
        for r in [0.3, 0.5, 0.7, 0.9]:
            circle = plt.Circle((0, 0), r, fill=False, color='#CCCCCC', 
                                linestyle='--', linewidth=1, alpha=0.5)
            ax2.add_patch(circle)
            ax2.text(0.02, r + 0.02, f'{r:.1f}', fontsize=8, color='#888888')
        
        # Draw center point
        ax2.scatter([0], [0], s=50, c='#1a1a2e', zorder=10)
        
        # Draw network nodes
        for i, network in enumerate(networks):
            mii = avg_mii[i]
            angle = angles[i]
            
            x = mii * np.cos(angle)
            y = mii * np.sin(angle)
            
            # Draw line from center to node
            ax2.plot([0, x], [0, y], color=self.networks[network]['color'], 
                    linewidth=2, alpha=0.6)
            
            # Draw node
            node_size = 150 + mii * 200
            ax2.scatter([x], [y], s=node_size, c=self.networks[network]['color'],
                       edgecolors='white', linewidth=2, zorder=5, alpha=0.85)
            
            # Add label outside
            label_r = max(mii + 0.15, 0.55)
            label_x = label_r * np.cos(angle)
            label_y = label_r * np.sin(angle)
            
            ha = 'left' if np.cos(angle) >= 0 else 'right'
            ax2.text(label_x, label_y, network[:5], fontsize=10, fontweight='bold',
                    ha=ha, va='center', color='#1a1a2e')
        
        ax2.axis('off')
        ax2.set_title('B. Integration Topology', fontsize=14, 
                     fontweight='bold', color='#1a1a2e', pad=15)
        
        # Add explanation text
        ax2.text(0, -1.35, 'Distance from center = MII value',
                ha='center', fontsize=9, style='italic', color='#666666')
        
        # Main title
        fig.suptitle('Multimodal Integration Hierarchy Across Brain Networks',
                    fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e')
        
        fig_path_png = os.path.join(self.results_dir, f"{FIGURE_NAMES['network_integration_hierarchy']}.png")
        fig_path_svg = os.path.join(self.results_dir, f"{FIGURE_NAMES['network_integration_hierarchy']}.svg")
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
        plt.savefig(fig_path_svg, format='svg', bbox_inches='tight', facecolor='#FAFAFA')
        plt.close()
        
        print(f"网络层级结构图保存至: {fig_path_png}, {fig_path_svg}")
    
    def plot_brain_network_map(self, preferences):
        """绘制脑网络图谱 - Enhanced scientific network visualization"""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, ConnectionPatch
        
        networks = list(self.networks.keys())
        n_networks = len(networks)
        
        # 跨被试汇总偏好和统计
        avg_preferences = {}
        modality_strengths = {}
        
        for network in networks:
            modality_counts = {m: 0 for m in self.modalities}
            modality_values = {m: [] for m in self.modalities}
            
            for subject_key in preferences:
                best_m = preferences[subject_key][network]['best_modality']
                modality_counts[best_m] += 1
                for m in self.modalities:
                    modality_values[m].append(preferences[subject_key][network].get('correlations', {}).get(m, 0))
            
            avg_preferences[network] = max(modality_counts, key=modality_counts.get)
            modality_strengths[network] = {m: np.mean(modality_values[m]) if modality_values[m] else 0 
                                           for m in self.modalities}
        
        # Create enhanced figure with multiple panels
        fig = plt.figure(figsize=(20, 10), facecolor='#FAFAFA')
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25,
                     height_ratios=[1.2, 1], width_ratios=[1.2, 1, 1])
        
        # Panel A: Enhanced network graph
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#F8F9FA')
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.set_aspect('equal')
        
        # Create circular layout
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_networks, endpoint=False)
        radius = 3
        
        x_pos = radius * np.cos(angles)
        y_pos = radius * np.sin(angles)
        
        # Draw connections between similar networks (based on modality preference)
        for i in range(n_networks):
            for j in range(i+1, n_networks):
                if avg_preferences[networks[i]] == avg_preferences[networks[j]]:
                    ax1.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]], 
                            color=MODALITY_COLORS[avg_preferences[networks[i]]],
                            alpha=0.2, linewidth=2, linestyle='--')
        
        # Draw center hub
        hub = plt.Circle((0, 0), 0.6, facecolor='#F8F9FA', edgecolor='#1a1a2e',
                         linewidth=2, zorder=5)
        ax1.add_patch(hub)
        ax1.text(0, 0, 'Brain\nNetworks', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='#1a1a2e')
        
        # Draw network nodes with pie charts showing modality distribution
        for i, network in enumerate(networks):
            best_modality = avg_preferences[network]
            node_color = MODALITY_COLORS[best_modality]
            edge_color = self.networks[network]['color']
            
            # Draw outer ring (network color)
            outer_circle = plt.Circle((x_pos[i], y_pos[i]), 0.7,
                                      facecolor='white', edgecolor=edge_color,
                                      linewidth=4, alpha=0.95, zorder=3)
            ax1.add_patch(outer_circle)
            
            # Draw inner circle (modality color)
            inner_circle = plt.Circle((x_pos[i], y_pos[i]), 0.55,
                                      facecolor=node_color, edgecolor='white',
                                      linewidth=2, alpha=0.85, zorder=4)
            ax1.add_patch(inner_circle)
            
            # Draw connection to center
            ax1.plot([0, x_pos[i]*0.7], [0, y_pos[i]*0.7], 
                    color=edge_color, linewidth=1.5, alpha=0.4, zorder=1)
            
            # Add network label with background
            label_r = radius + 0.9
            label_x = label_r * np.cos(angles[i])
            label_y = label_r * np.sin(angles[i])
            
            ha = 'left' if np.cos(angles[i]) >= 0 else 'right'
            ax1.text(label_x, label_y, network, ha=ha, va='center',
                    fontsize=10, fontweight='bold', color='#1a1a2e',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=edge_color, linewidth=1.5, alpha=0.9))
            
            # Add modality letter inside
            ax1.text(x_pos[i], y_pos[i], best_modality[0].upper(),
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white', zorder=5)
        
        ax1.axis('off')
        ax1.set_title('A. Network Modality Preference Graph', fontsize=14, 
                     fontweight='bold', color='#1a1a2e', pad=20)
        
        # Panel B: Stacked bar showing modality preference counts
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#F8F9FA')
        
        # Count modality preferences across networks
        modality_counts = {m: 0 for m in self.modalities}
        for network in networks:
            modality_counts[avg_preferences[network]] += 1
        
        # Create horizontal stacked bar
        left = 0
        for modality in self.modalities:
            count = modality_counts[modality]
            bar = ax2.barh([0], [count], left=[left], height=0.6,
                          color=MODALITY_COLORS[modality], alpha=0.85,
                          edgecolor='white', linewidth=2,
                          label=f'{modality.capitalize()} ({count})')
            if count > 0:
                ax2.text(left + count/2, 0, str(count), ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            left += count
        
        ax2.set_xlim(0, n_networks)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel('Number of Networks', fontsize=12, fontweight='bold')
        ax2.set_yticks([])
        ax2.set_title('B. Modality Dominance Distribution', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=3, fontsize=10, framealpha=0.95)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Panel C: Network-modality heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor('#F8F9FA')
        
        # Build matrix from preferences
        matrix = np.zeros((n_networks, len(self.modalities)))
        for i, network in enumerate(networks):
            for j, modality in enumerate(self.modalities):
                # Use 1 if this modality is dominant for this network
                matrix[i, j] = 1.0 if avg_preferences[network] == modality else 0.3
        
        # Create custom colormap for categorical display
        im = ax3.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Add markers for dominant modality
        for i, network in enumerate(networks):
            for j, modality in enumerate(self.modalities):
                if avg_preferences[network] == modality:
                    ax3.scatter([j], [i], marker='*', s=200, 
                               c='white', edgecolors='#1a1a2e', linewidth=1.5, zorder=5)
        
        ax3.set_xticks(range(len(self.modalities)))
        ax3.set_xticklabels([m.capitalize() for m in self.modalities], fontsize=11, fontweight='bold')
        ax3.set_yticks(range(n_networks))
        ax3.set_yticklabels([n[:5] for n in networks], fontsize=10)
        ax3.set_title('C. Preference Matrix', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15)
        
        # Panel D: Detailed network cards
        ax4 = fig.add_subplot(gs[1, :])
        ax4.set_facecolor('#F8F9FA')
        ax4.set_xlim(0, 14)
        ax4.set_ylim(0, 3)
        
        # Draw network info cards
        card_width = 1.8
        card_height = 2.2
        spacing = 0.1
        
        for i, network in enumerate(networks):
            x_start = i * (card_width + spacing) + 0.3
            y_start = 0.4
            
            best_modality = avg_preferences[network]
            
            # Card background with shadow
            shadow = FancyBboxPatch((x_start+0.05, y_start-0.05), card_width, card_height,
                                    boxstyle="round,pad=0.05", facecolor='#CCCCCC',
                                    alpha=0.3, zorder=1)
            ax4.add_patch(shadow)
            
            card = FancyBboxPatch((x_start, y_start), card_width, card_height,
                                  boxstyle="round,pad=0.05", 
                                  facecolor='white', edgecolor=self.networks[network]['color'],
                                  linewidth=2.5, zorder=2)
            ax4.add_patch(card)
            
            # Network name header with color
            header = FancyBboxPatch((x_start, y_start + card_height - 0.5), card_width, 0.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=self.networks[network]['color'],
                                    edgecolor='none', zorder=3)
            ax4.add_patch(header)
            ax4.text(x_start + card_width/2, y_start + card_height - 0.25,
                    network[:6], ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white', zorder=4)
            
            # Modality indicator circle
            modality_circle = plt.Circle((x_start + card_width/2, y_start + 1.4), 0.35,
                                         facecolor=MODALITY_COLORS[best_modality],
                                         edgecolor='white', linewidth=2, zorder=4)
            ax4.add_patch(modality_circle)
            ax4.text(x_start + card_width/2, y_start + 1.4, best_modality[0].upper(),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white', zorder=5)
            
            # Dominant modality label
            ax4.text(x_start + card_width/2, y_start + 0.7,
                    f'{best_modality.capitalize()}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#1a1a2e', zorder=4)
            ax4.text(x_start + card_width/2, y_start + 0.4,
                    'Dominant', ha='center', va='center',
                    fontsize=8, color='#666666', zorder=4)
        
        ax4.axis('off')
        ax4.set_title('D. Network Summary Cards', fontsize=14, 
                     fontweight='bold', loc='left', color='#1a1a2e', pad=15, x=0.02)
        
        # Add legend at bottom
        legend_elements = [mpatches.Patch(facecolor=MODALITY_COLORS[m], edgecolor='white',
                          label=f'{m.capitalize()}') for m in self.modalities]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                  fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, 0.02))
        
        # Main title
        fig.suptitle('Brain Network Modality Preference Analysis',
                    fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e')
        
        fig_path = os.path.join(self.results_dir, 'brain_network_map.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
        plt.close()
        
        print(f"脑网络图谱保存至: {fig_path}")
    
    def generate_report(self, network_stats, preferences, mii_results, stats_results):
        """生成分析报告"""
        report = []
        report.append("=" * 70)
        report.append("功能网络分析报告 - Brain Network Analysis Report")
        report.append("=" * 70)
        report.append("")
        
        # 1. 网络-模态敏感性汇总
        report.append("1. 网络-模态敏感性 (跨被试平均)")
        report.append("-" * 50)
        
        networks = list(self.networks.keys())
        header = f"{'Network':<20}" + "".join([f"{m.capitalize():<12}" for m in self.modalities]) + "Best"
        report.append(header)
        report.append("-" * 60)
        
        for network in networks:
            values = []
            for modality in self.modalities:
                mean_val = np.mean([network_stats[s][network][modality]['mean'] 
                                   for s in network_stats])
                values.append(mean_val)
            
            best_idx = np.argmax(values)
            best_modality = self.modalities[best_idx]
            
            row = f"{network:<20}" + "".join([f"{v:.4f}{'*' if i==best_idx else ' ':<11}" 
                                               for i, v in enumerate(values)])
            row += best_modality.capitalize()
            report.append(row)
        
        # 2. 多模态整合指数
        report.append("\n\n2. 多模态整合指数 (MII)")
        report.append("-" * 50)
        
        for network in networks:
            avg_mii = np.mean([mii_results[s][network]['mii'] for s in mii_results])
            interpretation = 'High integration' if avg_mii > 0.7 else ('Moderate' if avg_mii > 0.4 else 'Modality-specific')
            report.append(f"{network:<20}: MII = {avg_mii:.3f} ({interpretation})")
        
        # 3. 统计检验结果
        report.append("\n\n3. 统计检验结果 (ANOVA)")
        report.append("-" * 50)
        
        for network in networks:
            anova = stats_results[network]['anova']
            sig = "***" if anova['p'] < 0.001 else ("**" if anova['p'] < 0.01 else ("*" if anova['p'] < 0.05 else ""))
            report.append(f"{network:<20}: F = {anova['F']:.3f}, p = {anova['p']:.4f} {sig}")
        
        # 4. 理论解释
        report.append("\n\n4. 关键发现与解释")
        report.append("-" * 50)
        report.append("")
        report.append("基于Schaefer 2018分区方案的7功能网络分析揭示：")
        report.append("")
        report.append("• Visual Network: 预期对视觉刺激高度敏感")
        report.append("• Somatomotor Network: 可能响应动作相关的视听内容")
        report.append("• Attention Networks: 跨模态注意力调控")
        report.append("• Default Mode Network: 语言/叙事整合中心")
        report.append("• Frontoparietal Network: 认知控制与多模态整合")
        
        report_text = "\n".join(report)
        report_path = os.path.join(self.results_dir, 'brain_network_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n报告保存至: {report_path}")
    
    def run_full_analysis(self):
        """运行完整的功能网络分析"""
        print("=" * 60)
        print("功能网络分析")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n1. 加载单模态相关系数...")
        correlations = self.load_unimodal_correlations()
        
        # 2. 计算网络统计
        print("2. 计算网络统计信息...")
        network_stats = self.compute_network_statistics(correlations)
        
        # 3. 计算模态偏好
        print("3. 计算模态偏好...")
        preferences = self.compute_network_modality_preference(network_stats)
        
        # 4. 计算多模态整合指数
        print("4. 计算多模态整合指数...")
        mii_results = self.compute_multimodal_integration_index(correlations)
        
        # 5. 统计检验
        print("5. 进行统计检验...")
        stats_results = self.statistical_analysis(network_stats)
        
        # 6. 绘制图表
        print("6. 绘制分析图表...")
        self.plot_network_modality_profile(network_stats)
        self.plot_network_hierarchy(mii_results)
        
        # 7. 生成报告
        print("7. 生成分析报告...")
        self.generate_report(network_stats, preferences, mii_results, stats_results)
        
        # 保存完整结果
        results = {
            'network_stats': network_stats,
            'preferences': preferences,
            'mii_results': mii_results,
            'stats_results': stats_results
        }
        
        np.save(os.path.join(self.results_dir, 'brain_network_results.npy'), results)
        
        print("\n" + "=" * 60)
        print("功能网络分析完成！")
        print(f"结果保存目录: {self.results_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='功能网络分析')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition',
                        help='项目根目录')
    parser.add_argument('--subjects', default='1,2,3,5', help='被试列表')
    parser.add_argument('--input_dir', default=None, help='单模态模型输入目录')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    
    args = parser.parse_args()
    subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    analyzer = BrainNetworkAnalyzer(args.project_dir, subjects, args.input_dir, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

