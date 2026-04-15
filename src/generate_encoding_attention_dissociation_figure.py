"""
Generate Figure 3B: Encoding-Attention Dissociation Visualization
This is the core figure demonstrating that attention does not track encoding strength.

生成编码-注意力分离的核心可视化图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import json

# Set up paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(PROJECT_DIR, 'runs', 'run_revised')
OUTPUT_DIR = os.path.join(RUN_DIR, 'figures')

# Load data
def load_data():
    """Load encoding and attention data from previous analyses"""
    
    # Load unimodal training summary (encoding performance)
    unimodal_path = os.path.join(RUN_DIR, 'unimodal_models', 'unimodal_training_summary.json')
    with open(unimodal_path, 'r') as f:
        unimodal_data = json.load(f)
    
    # Load attention weights
    attention_path = os.path.join(RUN_DIR, 'crossmodal_attention', 'crossmodal_attention_results.json')
    with open(attention_path, 'r') as f:
        attention_data = json.load(f)
    
    return unimodal_data, attention_data

def compute_statistics(unimodal_data, attention_data):
    """Compute encoding and attention statistics"""
    
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
    modalities = ['visual', 'audio', 'language']
    
    # Encoding performance (mean correlation per modality)
    encoding = {m: [] for m in modalities}
    for subj in subjects:
        for mod in modalities:
            encoding[mod].append(unimodal_data[subj][mod]['mean_correlation'])
    
    encoding_mean = {m: np.mean(encoding[m]) for m in modalities}
    encoding_std = {m: np.std(encoding[m]) for m in modalities}
    
    # Attention weights
    attention = {m: [] for m in modalities}
    for subj in subjects:
        weights = attention_data['modality_weights'][subj]
        for i, mod in enumerate(modalities):
            attention[mod].append(weights[i])
    
    attention_mean = {m: np.mean(attention[m]) for m in modalities}
    attention_std = {m: np.std(attention[m]) for m in modalities}
    
    # Compute "efficient" allocation (proportional to encoding strength)
    total_encoding = sum(encoding_mean.values())
    efficient_allocation = {m: encoding_mean[m] / total_encoding for m in modalities}
    
    return {
        'encoding_mean': encoding_mean,
        'encoding_std': encoding_std,
        'attention_mean': attention_mean,
        'attention_std': attention_std,
        'efficient_allocation': efficient_allocation,
        'subjects': subjects,
        'modalities': modalities,
        'encoding_by_subject': encoding,
        'attention_by_subject': attention
    }


def generate_figure_3b(stats, output_path):
    """
    Generate the core Encoding-Attention Dissociation figure
    Shows encoding strength vs attention weight with efficiency line
    """
    
    # BuGn-based colors
    MODALITY_COLORS = {
        'visual': '#005824',    # Darkest BuGn
        'audio': '#41AE76',     # Medium BuGn  
        'language': '#99D8C9'   # Light BuGn
    }
    
    modalities = stats['modalities']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#FAFAFA')
    
    # ============================================
    # Panel A: Encoding Strength vs Attention Weight (THE KEY FIGURE)
    # ============================================
    ax1 = axes[0]
    ax1.set_facecolor('#F8F9FA')
    
    # Data points
    encoding_vals = [stats['encoding_mean'][m] for m in modalities]
    attention_vals = [stats['attention_mean'][m] for m in modalities]
    efficient_vals = [stats['efficient_allocation'][m] for m in modalities]
    
    # Plot efficiency line (y = x normalized)
    x_line = np.linspace(0.08, 0.25, 100)
    # If attention tracked encoding, attention = encoding / sum(encoding)
    total_enc = sum(encoding_vals)
    y_efficient = x_line / total_enc
    ax1.plot(x_line, y_efficient, '--', color='#E74C3C', linewidth=2.5, 
             label='Efficient Allocation\n(Attention ∝ Encoding)', alpha=0.8, zorder=1)
    
    # Fill area showing dissociation
    ax1.fill_between(x_line, 1/3, y_efficient, alpha=0.15, color='#3498DB',
                     label='Dissociation Region')
    
    # Uniform attention line
    ax1.axhline(y=1/3, color='#2ECC71', linestyle=':', linewidth=2, 
                label='Observed (~33%)', alpha=0.9, zorder=2)
    
    # Plot actual data points with error bars
    for i, mod in enumerate(modalities):
        enc_std = stats['encoding_std'][mod]
        att_std = stats['attention_std'][mod]
        
        # Error bars
        ax1.errorbar(encoding_vals[i], attention_vals[i], 
                    xerr=enc_std, yerr=att_std,
                    fmt='none', ecolor=MODALITY_COLORS[mod], 
                    elinewidth=2, capsize=5, capthick=2, alpha=0.6, zorder=3)
        
        # Main scatter point
        ax1.scatter(encoding_vals[i], attention_vals[i], 
                   s=400, c=MODALITY_COLORS[mod], 
                   edgecolors='white', linewidths=3,
                   label=f'{mod.capitalize()}', zorder=4)
        
        # Add modality label
        offset_x = 0.008 if mod == 'visual' else -0.005
        offset_y = 0.025 if mod != 'audio' else -0.03
        ax1.annotate(mod.capitalize(), 
                    (encoding_vals[i] + offset_x, attention_vals[i] + offset_y),
                    fontsize=11, fontweight='bold', color=MODALITY_COLORS[mod],
                    ha='center')
        
        # Draw arrow from observed to "efficient" point
        if mod == 'visual':  # Only for visual to show the gap
            ax1.annotate('', xy=(encoding_vals[i], efficient_vals[i]),
                        xytext=(encoding_vals[i], attention_vals[i]),
                        arrowprops=dict(arrowstyle='->', color='#E74C3C', 
                                       lw=2, ls='--', alpha=0.7))
            
            # Add gap annotation
            gap = efficient_vals[i] - attention_vals[i]
            ax1.text(encoding_vals[i] + 0.012, (efficient_vals[i] + attention_vals[i])/2,
                    f'Gap: {gap*100:.1f}%\npoints',
                    fontsize=9, color='#E74C3C', fontweight='bold',
                    ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='#E74C3C', alpha=0.9))
    
    ax1.set_xlabel('Encoding Strength (Pearson r)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Attention Weight', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('A. Encoding-Attention Dissociation', fontsize=14, 
                 fontweight='bold', loc='left', color='#1a1a2e', pad=15)
    ax1.set_xlim(0.08, 0.25)
    ax1.set_ylim(0.2, 0.55)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, fancybox=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ============================================
    # Panel B: Bar comparison - Expected vs Observed
    # ============================================
    ax2 = axes[1]
    ax2.set_facecolor('#F8F9FA')
    
    x = np.arange(len(modalities))
    width = 0.35
    
    # Expected (efficient) bars
    efficient_bars = ax2.bar(x - width/2, [stats['efficient_allocation'][m] for m in modalities],
                            width, label='Efficient Allocation', color='#E74C3C', alpha=0.7,
                            edgecolor='white', linewidth=2)
    
    # Observed bars
    observed_bars = ax2.bar(x + width/2, [stats['attention_mean'][m] for m in modalities],
                           width, label='Observed Attention', 
                           color=[MODALITY_COLORS[m] for m in modalities], alpha=0.85,
                           edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar in efficient_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='#E74C3C')
    
    for bar in observed_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Uniform reference line
    ax2.axhline(y=1/3, color='#2ECC71', linestyle=':', linewidth=2, alpha=0.8)
    ax2.text(2.6, 1/3 + 0.01, 'Uniform\n(33.3%)', fontsize=8, color='#2ECC71', 
             fontweight='bold', va='bottom')
    
    ax2.set_xlabel('Modality', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Attention Weight', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('B. Expected vs Observed Allocation', fontsize=14,
                 fontweight='bold', loc='left', color='#1a1a2e', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in modalities], fontsize=11)
    ax2.set_ylim(0, 0.6)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ============================================
    # Panel C: Subject-level consistency
    # ============================================
    ax3 = axes[2]
    ax3.set_facecolor('#F8F9FA')
    
    subjects = stats['subjects']
    x_subj = np.arange(len(subjects))
    width_subj = 0.25
    
    for i, mod in enumerate(modalities):
        att_by_subj = stats['attention_by_subject'][mod]
        bars = ax3.bar(x_subj + (i-1)*width_subj, att_by_subj, width_subj,
                      label=mod.capitalize(), color=MODALITY_COLORS[mod], 
                      alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Uniform reference
    ax3.axhline(y=1/3, color='#2ECC71', linestyle=':', linewidth=2, 
               label='Uniform (33.3%)', alpha=0.8)
    
    ax3.set_xlabel('Subject', fontsize=12, fontweight='bold', labelpad=10)
    ax3.set_ylabel('Attention Weight', fontsize=12, fontweight='bold', labelpad=10)
    ax3.set_title('C. Cross-Subject Consistency', fontsize=14,
                 fontweight='bold', loc='left', color='#1a1a2e', pad=15)
    ax3.set_xticks(x_subj)
    ax3.set_xticklabels([s.replace('sub-0', 'S') for s in subjects], fontsize=11)
    ax3.set_ylim(0.25, 0.45)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Main title
    fig.suptitle('The Encoding-Attention Dissociation: Brain Maintains Balanced Attention Despite Unequal Encoding',
                fontsize=14, fontweight='bold', y=1.02, color='#1a1a2e')
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(output_path + '.png', dpi=600, bbox_inches='tight', facecolor='#FAFAFA')
    plt.savefig(output_path + '.svg', format='svg', bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    
    print(f"Figure saved to: {output_path}.png and .svg")


def generate_figure_3_complete(stats, output_path):
    """
    Generate complete Figure 3 with all panels for the paper
    A more compact version suitable for the manuscript
    """
    
    MODALITY_COLORS = {
        'visual': '#005824',
        'audio': '#41AE76',
        'language': '#99D8C9'
    }
    
    modalities = stats['modalities']
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ============================================
    # Panel A: Encoding Strength vs Attention Weight (TOP LEFT)
    # ============================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('white')
    
    encoding_vals = [stats['encoding_mean'][m] for m in modalities]
    attention_vals = [stats['attention_mean'][m] for m in modalities]
    efficient_vals = [stats['efficient_allocation'][m] for m in modalities]
    
    # Efficiency line
    x_line = np.linspace(0.08, 0.25, 100)
    total_enc = sum(encoding_vals)
    y_efficient = x_line / total_enc
    ax1.plot(x_line, y_efficient, '--', color='#E74C3C', linewidth=2.5, 
             label='Efficient (Att ∝ Enc)', alpha=0.8, zorder=1)
    
    # Dissociation region
    ax1.fill_between(x_line, 1/3, y_efficient, alpha=0.12, color='#3498DB')
    
    # Uniform line
    ax1.axhline(y=1/3, color='#2ECC71', linestyle=':', linewidth=2.5, 
                label='Observed (~33%)', alpha=0.9, zorder=2)
    
    # Data points
    for i, mod in enumerate(modalities):
        ax1.errorbar(encoding_vals[i], attention_vals[i], 
                    xerr=stats['encoding_std'][mod], yerr=stats['attention_std'][mod],
                    fmt='none', ecolor=MODALITY_COLORS[mod], 
                    elinewidth=2, capsize=4, capthick=2, alpha=0.5, zorder=3)
        
        ax1.scatter(encoding_vals[i], attention_vals[i], 
                   s=350, c=MODALITY_COLORS[mod], 
                   edgecolors='white', linewidths=2.5,
                   label=f'{mod.capitalize()}', zorder=4)
        
        # Label
        offset_y = 0.022 if mod != 'audio' else -0.028
        ax1.annotate(mod.capitalize()[:3].upper(), 
                    (encoding_vals[i], attention_vals[i] + offset_y),
                    fontsize=10, fontweight='bold', color=MODALITY_COLORS[mod],
                    ha='center')
    
    ax1.set_xlabel('Encoding Strength (r)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax1.set_title('A. Encoding vs Attention', fontsize=12, fontweight='bold', loc='left', pad=10)
    ax1.set_xlim(0.08, 0.24)
    ax1.set_ylim(0.22, 0.52)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ============================================
    # Panel B: Bar comparison (TOP RIGHT)
    # ============================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('white')
    
    x = np.arange(len(modalities))
    width = 0.35
    
    efficient_bars = ax2.bar(x - width/2, [stats['efficient_allocation'][m] for m in modalities],
                            width, label='If Efficient', color='#E74C3C', alpha=0.7,
                            edgecolor='white', linewidth=1.5)
    
    observed_bars = ax2.bar(x + width/2, [stats['attention_mean'][m] for m in modalities],
                           width, label='Observed', 
                           color=[MODALITY_COLORS[m] for m in modalities], alpha=0.85,
                           edgecolor='white', linewidth=1.5)
    
    for bar in efficient_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f'{height:.0%}', ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='#E74C3C')
    
    for bar in observed_bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f'{height:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.axhline(y=1/3, color='#2ECC71', linestyle=':', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Modality', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax2.set_title('B. Expected vs Observed', fontsize=12, fontweight='bold', loc='left', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in modalities], fontsize=10)
    ax2.set_ylim(0, 0.58)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ============================================
    # Panel C: Subject consistency (BOTTOM LEFT)
    # ============================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('white')
    
    subjects = stats['subjects']
    x_subj = np.arange(len(subjects))
    width_subj = 0.25
    
    for i, mod in enumerate(modalities):
        att_by_subj = stats['attention_by_subject'][mod]
        ax3.bar(x_subj + (i-1)*width_subj, att_by_subj, width_subj,
               label=mod.capitalize(), color=MODALITY_COLORS[mod], 
               alpha=0.85, edgecolor='white', linewidth=1.5)
    
    ax3.axhline(y=1/3, color='#2ECC71', linestyle=':', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Subject', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax3.set_title('C. Cross-Subject Consistency', fontsize=12, fontweight='bold', loc='left', pad=10)
    ax3.set_xticks(x_subj)
    ax3.set_xticklabels([s.replace('sub-0', 'S') for s in subjects], fontsize=10)
    ax3.set_ylim(0.25, 0.42)
    ax3.legend(loc='upper right', fontsize=9, ncol=3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ============================================
    # Panel D: Key insight summary (BOTTOM RIGHT)
    # ============================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('white')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Title
    ax4.text(5, 9.2, 'D. Key Insight', fontsize=12, fontweight='bold', 
             ha='center', color='#1a1a2e')
    
    # Core message box
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='#005824', linewidth=2, alpha=0.95)
    
    core_message = (
        "ENCODING-ATTENTION\n"
        "DISSOCIATION\n\n"
        "Visual encoding: 2× stronger\n"
        "Visual attention: same as others\n\n"
        "→ Brain decouples\n"
        "   'what to extract' from\n"
        "   'how to integrate'"
    )
    ax4.text(5, 5.5, core_message, fontsize=11, ha='center', va='center',
             fontweight='bold', color='#005824', bbox=box_props,
             linespacing=1.4)
    
    # Implication
    impl_props = dict(boxstyle='round,pad=0.4', facecolor='#E8F6F3', 
                     edgecolor='#41AE76', linewidth=1.5)
    ax4.text(5, 1.3, 'VLM Design: Add explicit\nattention balancing', 
             fontsize=10, ha='center', va='center',
             fontweight='bold', color='#238B45', bbox=impl_props)
    
    # Main title
    fig.suptitle('Figure 3: The Encoding-Attention Dissociation',
                fontsize=14, fontweight='bold', y=0.98, color='#1a1a2e')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path + '.png', dpi=600, bbox_inches='tight', facecolor='#FAFAFA')
    plt.savefig(output_path + '.svg', format='svg', bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    
    print(f"Complete Figure 3 saved to: {output_path}.png and .svg")


def main():
    print("=" * 60)
    print("Generating Encoding-Attention Dissociation Figures")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    unimodal_data, attention_data = load_data()
    
    # Compute statistics
    print("2. Computing statistics...")
    stats = compute_statistics(unimodal_data, attention_data)
    
    # Print summary
    print("\n--- Summary Statistics ---")
    print("\nEncoding Performance (mean r):")
    for mod in stats['modalities']:
        print(f"  {mod.capitalize():12s}: {stats['encoding_mean'][mod]:.4f} ± {stats['encoding_std'][mod]:.4f}")
    
    print("\nAttention Weights:")
    for mod in stats['modalities']:
        print(f"  {mod.capitalize():12s}: {stats['attention_mean'][mod]:.3f} ± {stats['attention_std'][mod]:.3f}")
    
    print("\nEfficient Allocation (if attention ∝ encoding):")
    for mod in stats['modalities']:
        print(f"  {mod.capitalize():12s}: {stats['efficient_allocation'][mod]:.3f}")
    
    print("\nDissociation Gap (Efficient - Observed):")
    for mod in stats['modalities']:
        gap = stats['efficient_allocation'][mod] - stats['attention_mean'][mod]
        print(f"  {mod.capitalize():12s}: {gap:+.3f} ({gap*100:+.1f}%)")
    
    # Generate figures
    print("\n3. Generating Figure 3B (horizontal layout)...")
    output_3b = os.path.join(OUTPUT_DIR, 'fig03b_encoding_attention_dissociation')
    generate_figure_3b(stats, output_3b)
    
    print("\n4. Generating Complete Figure 3 (2x2 layout)...")
    output_3 = os.path.join(OUTPUT_DIR, 'fig03_encoding_attention_complete')
    generate_figure_3_complete(stats, output_3)
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

