# Collapse Index: SST-2 Public Validation

<div align="center">

[![Version](https://img.shields.io/badge/version-2.0.0-blue?style=flat-square)](https://github.com/collapseindex/ci-sst2)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

**[collapseindex.org](https://collapseindex.org)** • **[ask@collapseindex.org](mailto:ask@collapseindex.org)**

</div>

> **Public Validation #2:** Reproducible demonstration of **Collapse Index (CI)** and **Structural Retention Index (SRI)** on binary sentiment classification.

> 📊 **Also Available:** [AG News Validation (ci-sri)](https://github.com/collapseindex/ci-sri) - Multi-class text classification (contrast case where confidence works)

**Why SST-2?** Binary sentiment classification is a standard benchmark. This validation shows CI/SRI detecting brittleness that confidence-based monitoring completely misses.

## 🎯 Results

**Reproducible Metrics (Public):**

| Metric | Value | Notes |
|--------|-------|-------|
| **Model** | DistilBERT-SST2 | HuggingFace public model |
| **Benchmark Accuracy** | 90.8% | Base examples (clean text) |
| **Flip Rate** | 43.4% | 217/500 base examples flip |
| **Dataset Size** | 2,000 rows | 500 base × 4 variants each |

**Advanced Diagnostics (Commercial Implementation):**

*Metric definitions are published in the referenced papers; this repository demonstrates behavior, not full reimplementation.*

| Metric | Value | Notes |
|--------|-------|-------|
| **CI Score (avg)** | 0.275 | Moderate prediction instability |
| **SRI Score (avg)** | 0.725 | Structural retention metric |
| **CI + SRI** | 1.000 | Perfect complementarity* |
| **AUC(CI)** | 0.698 | Error discrimination via instability |
| **AUC(SRI)** | 0.698 | Error discrimination via retention |
| **AUC(Conf)** | 0.515 | Near-random on perturbed variants |
| **AUC(Conf) base** | 0.866 | Works on clean text only |
| **Δ CI-Conf** | +0.183 | CI is 18% better than confidence |
| **Confidence Status** | ⚠️ Unreliable | Degrades under perturbation |
| **SRI Grade** | B | Good structural retention |
| **Trinity Verdict** | 🟡 Overconfident Stable | Low drift + good retention + broken confidence |
| **CSI Error Distribution** | 13/6/17/10/0 | Type I/II/III/IV/V error counts |

*\*CI + SRI = 1.0 is empirical for this validation, not a theoretical identity.*

*Note: Advanced metrics require commercial licensing. Contact ask@collapseindex.org or visit [collapseindex.org/evals.html](https://collapseindex.org/evals.html)*

## 📊 The SRI Story

**Important:** In this SST-2 validation, confidence works on clean base examples (AUC=0.866) but degrades to near-random under perturbations (AUC=0.515 on all variants). This makes SST-2 the failure case CI/SRI were designed for: detecting brittleness that emerges only under real-world input variation.

**Standard benchmarks say:** "Ship it! 90.8% accuracy."

**What confidence tells you on clean text:** Errors are lower confidence (AUC=0.866 on base examples). Looks fine.

**What happens under perturbation:** Confidence collapses to near-random (AUC=0.515 on all variants). The model loses its ability to distinguish errors from correct predictions when users make typos or rephrase.

**What CI/SRI reveal:** The model has moderate instability (CI=0.275) with 43% of predictions flipping under benign perturbations. CI achieves AUC=0.698—18 percentage points better than confidence at predicting errors.

**Failure Mode Classification (CSI):**
- **Type I (13):** Stable Collapse - Confidently wrong, no flips
- **Type II (6):** Hidden Instability - Internal shifts, same label
- **Type III (17):** Moderate Flip - Clear label flips under stress
- **Type IV (10):** High Flip - Frequent flips and instability
- **Type V (0):** Extreme Flip - Chaotic breakdown

**Why Trinity matters:** This is exactly the scenario where you need CI/SRI:
- **Confidence** → Useless for error detection (AUC ≈ 0.5)
- **CI (instability)** → Catches 18% more errors than confidence
- **SRI (structure)** → Grade B retention despite high flip rate
- **CSI (failure type)** → 17 Type III + 10 Type IV = behavioral instability you can catch

**Key Insight:** Compare to AG News where confidence works (AUC=0.829). Here, confidence fails completely. Same framework, different result—CI/SRI adapt to the model's actual behavior.

**SST-2 Results:**
- **Trinity Verdict:** 🟡 Overconfident Stable (moderate drift + good retention + broken confidence)
- **46 total errors** across all CSI types, with Type III (17) and Type IV (10) dominating
- **43.4% flip rate:** Nearly half of predictions change under perturbation
- **The confidence gap:** Errors and correct predictions have nearly identical confidence distributions

*Operational implication:* Confidence-based rejection thresholds will not work for this model. Use CI thresholds instead.

## 🔬 Dataset

- **Base:** 500 examples from SST-2 validation set (binary sentiment classification)
- **Perturbations:** 3 variants per base using:
  - Character-level typos (keyboard distance)
  - Synonym substitution (WordNet)
  - Natural paraphrasing patterns
- **Total:** 2,000 rows (500 × 4 variants)
- **Format:** CSV with columns: `id`, `variant_id`, `text`, `true_label`, `pred_label`, `confidence`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional)

The `sst2_ci_demo.csv` is included, but you can regenerate:

```bash
python generate_sst2_demo.py
```

This will:
- Download SST-2 validation set (500 examples)
- Generate 3 perturbations per example
- Run DistilBERT-SST2 inference on all 2,000 rows
- Save to `sst2_ci_demo.csv`

Takes ~3-5 minutes on CPU.

### 3. Verify Basic Metrics (Optional)

Validate flip rate and accuracy independently:

```bash
python validate_metrics.py
```

This verifies metrics that don't require the full CI pipeline.

### 4. Analyze with Collapse Index

For complete analysis (AUC, CI scores, high-confidence errors):

```bash
# Request evaluation from Collapse Index Labs
# https://collapseindex.org/evals.html
# Email: ask@collapseindex.org
```

## 📁 Files

- `README.md` - This file
- `requirements.txt` - Python dependencies
- `generate_sst2_demo.py` - Dataset generation script
- `validate_metrics.py` - Independent metric verification script
- `sst2_ci_demo.csv` - Full 2,000-row dataset with predictions

## 🔗 Links

**CI Framework & Validations:**
- **Main CI Repository:** [github.com/collapseindex/collapseindex](https://github.com/collapseindex/collapseindex)
- **SST-2 Validation:** [github.com/collapseindex/ci-sst2](https://github.com/collapseindex/ci-sst2) *(you are here)*
- **AG News Validation (SRI):** [github.com/collapseindex/ci-sri](https://github.com/collapseindex/ci-sri)
- **Collapse Index Labs:** [collapseindex.org](https://collapseindex.org)

**Data & Models:**
- **Model Used:** [huggingface.co/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- **SST-2 Dataset:** [huggingface.co/datasets/sst2](https://huggingface.co/datasets/sst2)

## 📝 Citation

If you use this validation dataset in your research:

```bibtex
@misc{ci-sst2-validation,
  title={Collapse Index: SST-2 Public Validation},
  author={Kwon, Alex},
  year={2025},
  url={https://github.com/collapseindex/ci-sst2},
  note={Collapse Index Labs}
}
```

**Author:** Alex Kwon ([collapseindex.org](https://collapseindex.org)) · ORCID: [0009-0002-2566-5538](https://orcid.org/0009-0002-2566-5538)

Please also cite the original SST-2 dataset:

```bibtex
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
```

## ⚖️ License

- **This Repository (v2.0.0):** MIT License (code only)
- **CI + SRI Methodology:** [Proprietary](https://github.com/collapseindex/collapseindex/blob/main/LICENSE.md) - (c) 2026 Collapse Index Labs - Alex Kwon
- **SST-2 Dataset:** Available via HuggingFace Datasets (cite original paper above)
- **DistilBERT Model:** Apache 2.0

**Copyright © 2026 Collapse Index Labs - Alex Kwon. All rights reserved.**

**Note:** This repository provides reproducible validation code for CI/SRI research. The complete implementation is proprietary. For commercial licensing, contact [ask@collapseindex.org](mailto:ask@collapseindex.org).

**Version History:**
- **v2.0.0** (Jan 2026) - **Major Update:** Added SRI metrics, Trinity framework, CSI breakdown. Updated to match AG News validation format. Key finding: SST-2 shows confidence failure (AUC=0.515) while AG News shows confidence success (AUC=0.829)—demonstrating CI/SRI adapt to actual model behavior.
- **v1.0.0** (Dec 2025) - Initial public release with SST-2 validation

## 📧 Contact

Questions? Email [ask@collapseindex.org](mailto:ask@collapseindex.org)


