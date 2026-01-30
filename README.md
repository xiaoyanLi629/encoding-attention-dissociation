# The Encoding-Attention Dissociation

## Apparent Inefficiency, Hidden Optimality: Why the Brain's Multimodal Attention Does Not Track Feature Strength

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete analysis pipeline for the CogSci 2026 submission on cognitive efficiency and multimodal brain encoding.

**Conference Theme**: Cognitive Efficiency and Inefficiency

**Key Finding**: Despite visual features dominating brain encoding (91% of regions), attention weights remain balanced (~33% per modality)—revealing an apparent "inefficiency" that enables robust multimodal integration.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Analysis Pipeline](#analysis-pipeline)
- [Key Results](#key-results)
- [Paper](#paper)
- [Data Requirements](#data-requirements)
- [Citation](#citation)

---

## Project Structure

```
project_1/
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
│
├── src/                                # Core analysis scripts
│   ├── 01_train_unimodal_models.py     # Step 1: Train unimodal encoding models
│   ├── 02_modality_contribution_analysis.py  # Step 2: Modality contribution analysis
│   ├── 03_crossmodal_attention_analysis.py   # Step 3: Cross-modal attention analysis
│   ├── 04_brain_network_analysis.py    # Step 4: Brain network analysis
│   ├── generate_all_figures.py         # Step 5: Publication figure generator
│   ├── generate_encoding_attention_dissociation_figure.py  # Core dissociation figure
│   └── brain_region_mapping.py         # Schaefer 1000 parcellation utilities
│
├── scripts/                            # Utility scripts
│   ├── run_full_analysis.py            # Complete pipeline runner
│   ├── convert_to_pdf.py               # Markdown to PDF converter
│   └── generate_images.py              # AI image generation (DALL-E 3/Gemini)
│
├── paper/                              # Manuscript files
│   ├── main.tex                        # LaTeX manuscript
│   ├── main.pdf                        # Compiled PDF
│   ├── CogSci_Template.bib             # Bibliography
│   ├── cogsci.sty                      # CogSci style file
│   └── figures/                        # Paper figures
│       └── figures1/                   # Additional figures
│
├── results/                            # Example results
│   └── example_run/                    # Pre-computed example outputs
│
└── runs/                               # Analysis outputs (timestamped)
    └── run_YYYYMMDD_HHMMSS/
        ├── run_config.json             # Run configuration
        ├── unimodal_models/            # Trained models and correlations
        ├── modality_contribution/      # Modality specificity analysis
        ├── crossmodal_attention/       # Attention weight analysis
        ├── brain_networks/             # Network-level analysis
        └── figures/                    # Publication figures (600 DPI)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- NiBabel (for neuroimaging data)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/encoding-attention-dissociation.git
cd encoding-attention-dissociation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Run Complete Analysis Pipeline

Each run creates a timestamped output directory: `runs/run_YYYYMMDD_HHMMSS/`

```bash
# Run full analysis (creates new timestamped directory)
python scripts/run_full_analysis.py --project_dir /path/to/data

# Skip training (if models already exist)
python scripts/run_full_analysis.py --skip_training

# Run only specific step
python scripts/run_full_analysis.py --only figures
```

### Run Steps Individually

```bash
# Step 1: Train unimodal encoding models
python src/01_train_unimodal_models.py \
    --project_dir /path/to/data \
    --output_dir ./runs/my_run/unimodal_models

# Step 2: Modality contribution analysis
python src/02_modality_contribution_analysis.py \
    --project_dir /path/to/data \
    --input_dir ./runs/my_run/unimodal_models \
    --output_dir ./runs/my_run/modality_contribution

# Step 3: Cross-modal attention analysis
python src/03_crossmodal_attention_analysis.py \
    --project_dir /path/to/data \
    --output_dir ./runs/my_run/crossmodal_attention

# Step 4: Brain network analysis
python src/04_brain_network_analysis.py \
    --project_dir /path/to/data \
    --input_dir ./runs/my_run/unimodal_models \
    --output_dir ./runs/my_run/brain_networks

# Step 5: Generate publication figures
python src/generate_all_figures.py \
    --project_dir /path/to/data \
    --input_dir ./runs/my_run \
    --output_dir ./runs/my_run/figures
```

---

## Analysis Pipeline

### Step 1: Unimodal Encoding Models
Trains Ridge regression models to predict fMRI responses from each modality (visual, audio, language) independently.

### Step 2: Modality Contribution Analysis
Quantifies the contribution of each modality to brain activity across 1000 cortical parcels using the Schaefer 2018 atlas.

### Step 3: Cross-Modal Attention Analysis
Trains a personalized multimodal neural network that learns explicit modality fusion weights, revealing the brain's attention allocation strategy.

### Step 4: Brain Network Analysis
Analyzes modality integration patterns across the 7 canonical functional networks (Visual, Somatomotor, Dorsal Attention, Ventral Attention, Limbic, Frontoparietal, Default Mode).

### Step 5: Figure Generation
Generates publication-quality figures (600 DPI) for the manuscript.

---

## Key Results

### The Encoding-Attention Dissociation

| Metric | Visual | Audio | Language |
|--------|--------|-------|----------|
| **Encoding Accuracy** | r = 0.204 | r = 0.107 | r = 0.122 |
| **Dominant Regions** | 91% | 5% | 4% |
| **Attention Weights** | 33% | 35% | 32% |

**Key Insight**: Despite visual features providing 2× higher encoding accuracy and dominating 91% of brain regions, attention weights remain balanced (~33% per modality). This "inefficiency" enables robust multimodal integration.

### Hierarchical Integration Architecture

1. **Early Sensory Cortex**: Modality-specific processing (high MSI)
2. **Association Cortex**: Multimodal integration (high MII)
3. **Default Mode Network**: Integration hub with balanced attention

---

## Paper

The manuscript is located in `paper/main.tex`. To compile:

```bash
cd paper
pdflatex main
biber main
pdflatex main
pdflatex main
```

The compiled PDF is available at `paper/main.pdf`.

---

## Data Requirements

This analysis requires:

1. **fMRI Data**: Preprocessed BOLD signals parcellated using Schaefer 1000 atlas
2. **Feature Embeddings**: 
   - Visual: CLIP embeddings
   - Audio: Wav2Vec2 embeddings
   - Language: GPT-2 embeddings

Data should be organized as:
```
/path/to/data/
├── fmri/
│   └── subject_X/
│       └── video_segment.npy
└── features/
    ├── visual/
    ├── audio/
    └── language/
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ding2026encoding,
  title={Apparent Inefficiency, Hidden Optimality: Why the Brain's Multimodal Attention Does Not Track Feature Strength},
  author={Ding, Xiaoyue},
  booktitle={Proceedings of the 48th Annual Conference of the Cognitive Science Society},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Schaefer 2018 parcellation atlas
- Friends and Movie10 naturalistic viewing datasets
- CogSci 2026 Conference on Cognitive Efficiency
