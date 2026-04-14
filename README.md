# The Encoding-Attention Dissociation

## Apparent Inefficiency, Hidden Optimality: Why the Brain's Multimodal Attention Does Not Track Feature Strength

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete analysis pipeline for a study on cognitive efficiency and multimodal brain encoding. Originally submitted to CogSci 2026, now revised for IEEE BIBM 2026 with additional multi-model validation and control analyses.

**Key Finding**: Despite visual features dominating brain encoding (91% of regions), attention weights remain balanced (~33% per modality) — an *Encoding-Attention Dissociation* that is robust across encoding thresholds, training strategies, and feature extraction models.

---

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/                                    # Analysis scripts
│   ├── 01_train_unimodal_models.py         # Step 1: Train unimodal Ridge encoding models
│   ├── 02_modality_contribution_analysis.py # Step 2: Modality contribution analysis
│   ├── 03_crossmodal_attention_analysis.py  # Step 3: Cross-modal attention analysis
│   ├── 04_brain_network_analysis.py         # Step 4: Brain network analysis
│   ├── 05_extract_additional_features.py    # Step 5: Multi-model feature extraction (NEW)
│   ├── 06_train_multimodal_model.py         # Step 6: Train multimodal attention network (NEW)
│   ├── 07_control_analyses.py              # Step 7: Control analyses (NEW)
│   ├── generate_all_figures.py             # Publication figure generator
│   ├── generate_encoding_attention_dissociation_figure.py
│   └── brain_region_mapping.py             # Schaefer 1000 parcellation utilities
│
├── scripts/                                # Utility scripts
│   ├── run_full_analysis.py                # Complete pipeline runner
│   ├── convert_to_pdf.py
│   └── generate_images.py
│
├── IEEE_manuscript/                        # IEEE BIBM 2026 manuscript (revised)
│   ├── IEEE-conference-template-062824.tex # Main LaTeX source
│   ├── IEEE-conference-template-062824.pdf # Compiled PDF
│   ├── IEEEtran.cls                        # IEEE style class
│   ├── figure1.png                         # Unimodal encoding figure
│   ├── figure2.png                         # Hierarchical integration figure
│   └── figure3_control_analyses.png        # Control analyses figure (NEW)
│
├── cog_sci_latex2026/                      # Original CogSci 2026 submission
│   ├── cogsci_full_paper_template.tex
│   ├── cogsci_full_paper_template.pdf
│   ├── cogsci_bibliography_template.bib
│   └── ...
│
├── results/example_run/                    # Pre-computed example outputs
│
└── runs/
    ├── run_20251210_142057/                # Original analysis run
    └── run_revised/                        # Revised analysis (NEW)
        ├── unimodal_models/                # 12 Ridge models (4 subjects × 3 modalities)
        ├── trained_models/                 # 4 multimodal .pth models
        ├── multi_model/                    # CLIP/Wav2Vec2/GPT-2 encoding comparison
        └── control_analyses/               # Subset + end-to-end control results
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Additional dependencies for multi-model feature extraction** (Step 5):
```bash
pip install transformers opencv-python-headless soundfile
```

---

## Analysis Pipeline

### Core Analysis (Steps 1–4)

| Step | Script | Description | GPU? |
|------|--------|-------------|------|
| 1 | `01_train_unimodal_models.py` | Ridge regression encoding models per modality | No |
| 2 | `02_modality_contribution_analysis.py` | Modality specificity/dominance analysis | No |
| 3 | `03_crossmodal_attention_analysis.py` | Extract attention weights from trained models | No |
| 4 | `04_brain_network_analysis.py` | Network-level integration analysis | No |

### New Analyses for Revision (Steps 5–7)

| Step | Script | Description | GPU? |
|------|--------|-------------|------|
| 5 | `05_extract_additional_features.py` | Extract CLIP/Wav2Vec2/GPT-2 features from stimuli | **Yes** |
| 6 | `06_train_multimodal_model.py` | Train PersonalizedMultiModalNetwork with learnable modality weights | **Yes** |
| 7 | `07_control_analyses.py` | High-encoding subset, end-to-end, permutation controls | **Yes** |

### Quick Start

```bash
# Run unimodal encoding (CPU, ~30 min)
python src/01_train_unimodal_models.py \
    --project_dir /path/to/data \
    --output_dir runs/run_revised/unimodal_models

# Train multimodal attention model (GPU recommended, ~1 hr)
python src/06_train_multimodal_model.py \
    --project_dir /path/to/data \
    --output_dir runs/run_revised/trained_models

# Extract additional model features (GPU required, ~2 hrs)
python src/05_extract_additional_features.py \
    --project_dir /path/to/data \
    --output_dir /path/to/data/data/features/additional_features

# Run control analyses (GPU recommended, ~1 hr)
python src/07_control_analyses.py \
    --project_dir /path/to/data \
    --output_dir runs/run_revised/control_analyses \
    --unimodal_results_dir runs/run_revised/unimodal_models \
    --controls subset,e2e
```

---

## Key Results

### Encoding-Attention Dissociation

| Metric | Visual | Audio | Language |
|--------|--------|-------|----------|
| **Encoding Accuracy** | r = 0.204 | r = 0.107 | r = 0.122 |
| **Dominant Regions** | 91% | 5% | 4% |
| **Observed Attention** | 32.3% | 34.6% | 33.1% |
| **Efficient Attention** | 47.1% | 24.7% | 28.2% |

### Control Analyses

| Control | Result | Rules Out |
|---------|--------|-----------|
| High-encoding subset (r > 0.2) | V=0.330, A=0.336, L=0.334 | Poor model fit |
| End-to-end training | V=0.328, A=0.343, L=0.329 | Training artifact |
| Multi-model (CLIP/Wav2Vec2/GPT-2) | Visual still dominant | Model-specific artifact |

### Multi-Model Validation

| Modality | Primary Model (r) | Alternative Model (r) |
|----------|-------------------|----------------------|
| Visual | SlowFast = 0.204 | CLIP = 0.133 |
| Audio | MFCC = 0.107 | Wav2Vec2 = 0.097 |
| Language | BERT = 0.122 | GPT-2 = 0.125 |

Visual > Language > Audio ordering preserved across both model sets.

---

## Data Requirements

This analysis requires data from the [Algonauts 2025 Challenge](https://algonautsproject.com/):

1. **fMRI Data**: Preprocessed BOLD signals parcellated using Schaefer 1000 atlas
2. **Feature Embeddings**: Official stimulus features (PCA-reduced)
3. **Stimuli** (for multi-model extraction): Original .mkv video files and .tsv transcripts

---

## Manuscripts

- **IEEE BIBM 2026** (revised): `IEEE_manuscript/IEEE-conference-template-062824.pdf`
- **CogSci 2026** (original): `cog_sci_latex2026/cogsci_full_paper_template.pdf`

---

## Citation

```bibtex
@inproceedings{anonymous2026encoding,
  title={Apparent Inefficiency, Hidden Optimality: Why the Brain's Multimodal
         Attention Does Not Track Feature Strength},
  author={Anonymous},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
