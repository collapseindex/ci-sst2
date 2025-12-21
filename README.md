# Collapse Index: SST-2 Public Validation

Reproducible demonstration showing Collapse Index detects brittleness that standard benchmarks miss.

## 🎯 Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Model** | DistilBERT-SST2 | HuggingFace public model |
| **Benchmark Accuracy** | 93%+ | SST-2 validation set |
| **CI Score** | 0.275 | Moderate instability (0-1 scale) |
| **AUC (CI)** | 0.698 | CI predicts flips reliably |
| **AUC (Confidence)** | 0.515 | Confidence barely predicts flips |
| **ΔAUC** | +0.182 | CI is 18% better than confidence |
| **Flip Rate** | 43.4% | 217/500 base cases flip |
| **Dataset Size** | 2,000 rows | 500 base × 4 variants each |

## 📊 The Story

**Standard benchmarks say:** "Ship it! 93% accuracy."

**Reality under perturbations:** Nearly half of predictions silently flip when users make typos or rephrase naturally.

**Why CI matters:** Confidence scores barely predict brittleness (AUC 0.515). Collapse Index catches it reliably (AUC 0.698).

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

### 3. Analyze with Collapse Index

Request evaluation from Collapse Index Labs
https://collapseindex.org/evals.html
Email: ask@collapseindex.org

## 📁 Files

- `README.md` - This file
- `requirements.txt` - Python dependencies
- `generate_sst2_demo.py` - Dataset generation script
- `sst2_ci_demo.csv` - Full 2,000-row dataset with predictions

## 🔗 Links

- **Full Analysis:** [collapseindex.org/evals.html#validation](https://collapseindex.org/evals.html#validation)
- **Collapse Index Labs:** [collapseindex.org](https://collapseindex.org)
- **Model Used:** [huggingface.co/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

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

- **This Repository:** MIT License (code and methodology)
- **SST-2 Dataset:** Available via HuggingFace Datasets (cite original paper above)
- **DistilBERT Model:** Apache 2.0

**Copyright © 2025 Collapse Index Labs - Alex Kwon. All rights reserved.**

## 📧 Contact

Questions? Email [ask@collapseindex.org](mailto:ask@collapseindex.org)

