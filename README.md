# KMR-RAG: Knowledge-guided Multi-document Retrieval-Augmented Generation and Refinement for Automated Cancer Registry Abstraction

<img width="1148" height="1108" alt="image" src="https://github.com/user-attachments/assets/0e72f73c-3d16-4ceb-b7e0-4de86ed09f29" />


<p align="center">
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-IJMI%20(Under%20Review)-blue"/></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"/></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-yellow"/></a>
  <a href="#"><img alt="Code" src="https://img.shields.io/badge/Code-Coming%20Soon-orange"/></a>
</p>

---

## 📋 Overview

**KMR-RAG** is a knowledge-guided multi-document Retrieval-Augmented Generation and Refinement framework for automated cancer registry abstraction from longitudinal clinical records. It extracts eight structured colorectal cancer registry fields (histological type, grade, tumor size, examined/positive nodes, and pathological TNM staging) from heterogeneous electronic medical records — without any site-specific model retraining.

The framework integrates three core components:

- **Knowledge-guided prompt engineering** — AJCC 8th Edition and Taiwan Cancer Registry (TCR) coding rules are embedded directly as prompt constraints, enforcing valid code outputs and evidence grounding.
- **Hybrid retrieval strategy** — BM25 for numerical and categorical fields; dense semantic retrieval for contextual fields. Validated across 12 embedding models.
- **Two-pass iterative refinement** — sequentially aggregates evidence across retrieved document chunks, contributing up to **+22.1 percentage points** of performance improvement.

---

## 📊 Results

Evaluated on **403 colorectal cancer patients** and **2,594 longitudinal clinical reports** from four hospital systems in Taiwan (KMUH, CMUH, KAFGH, PPH):

| Hospital | Language | Overall F1 | vs. HAN |
|----------|----------|-----------|---------|
| KMUH | English | **0.842** | +6.3 pp |
| CMUH | English | **0.705** | +9.7 pp |
| KAFGH | Chinese / Mixed | **0.556** | +52.4 pp |
| PPH | Chinese | **0.666** | +56.2 pp |

> KMR-RAG outperforms all baselines at CMUH and surpasses HAN by up to **+46.2 pp** at KAFGH — where the supervised model collapses to near-zero performance.

---

## 🏗️ Architecture

```
Patient Records (n reports, ≥18 months)
        │
        ▼
┌─────────────────────┐
│  Phase 1: Indexing  │  Per-patient vector database
│  chunk → encode     │  (prevents cross-patient contamination)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 2: Retrieval │  BM25 (numerical/categorical fields)
│  field-specific     │  Dense embeddings (semantic fields)
│  query q_f → top-k  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 3: Prompt    │  M_R: AJCC/TCR coding rules
│  construction       │  M_T: evidence grounding constraints
│                     │  M_C: retrieved chunks
│                     │  M_O: JSON output schema
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 4: Two-pass  │  Pass 1: initial extraction
│  LLM inference      │  Pass 2..k: iterative refinement
│                     │  with hard modification constraints
└─────────────────────┘
```

---

## 🔧 System Requirements

| Component | Specification |
|-----------|-------------|
| GPU | NVIDIA RTX 3090 (24 GB VRAM) or equivalent |
| LLM | LLaMA-3.1-8B-Instruct (recommended) |
| Inference server | vLLM / LMStudio |
| Embedding | bge-large-en-v1.5 (KMUH), BM25 (CMUH), BGE-fr-en (KAFGH), mE5-large (PPH) |
| Python | 3.10+ |
| Deployment | Fully on-premise — no external API calls |

---

## 📦 Code Release

> **Code, prompt templates, and evaluation scripts will be released upon paper acceptance.**

The release will include:

- [ ] Complete KMR-RAG inference pipeline
- [ ] Prompt templates for all 8 registry fields (AJCC/TCR rules)
- [ ] Hybrid retrieval module (BM25 + dense)
- [ ] Two-pass refinement implementation
- [ ] Evaluation scripts (micro-F1, bootstrap CI)
- [ ] Per-hospital retrieval model selection guide
- [ ] Example configuration files

To be notified of the release, please **Watch** or **Star** this repository.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{kmrrag2025,
  title   = {Knowledge-guided Multi-document Retrieval-Augmented
             Generation and Refinement ({KMR-RAG}) for Automated
             Cancer Registration: A Multi-center Evaluation},
  author  = {[Authors]},
  journal = {International Journal of Medical Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## 📁 Repository Structure (Preview)

```
KMR-RAG/
├── README.md
├── requirements.txt
├── config/
│   └── hospital_configs/        # per-hospital retrieval settings
├── src/
│   ├── indexing/                # document chunking & embedding
│   ├── retrieval/               # BM25 + dense hybrid retrieval
│   ├── prompts/                 # M_R, M_T, M_C, M_O modules
│   ├── refinement/              # two-pass refinement engine
│   └── evaluation/              # F1, bootstrap CI
├── prompts/
│   ├── original_template.txt    # Pass 1 prompt
│   ├── refinement_template.txt  # Pass 2..k prompt
│   └── field_rules/             # per-field AJCC/TCR rules
│       ├── H_histological.txt
│       ├── G_grade.txt
│       ├── TS_tumor_size.txt
│       ├── EN_examined_nodes.txt
│       ├── PN_positive_nodes.txt
│       └── TNM_staging.txt
└── supplementary/               # full prompt templates (paper)
```

---

## 🏥 Dataset

This study used a retrospective multi-center colorectal cancer dataset from four hospitals in Taiwan. Due to patient privacy regulations, the raw clinical data **cannot be shared**. However, the following are provided:

- Anonymized evaluation statistics
- Gold-standard field distribution
- Inter-hospital documentation heterogeneity descriptions

---

## 📬 Contact

For questions regarding the paper or early access to code, please open an **Issue** or contact the corresponding author at `[hjdai@nkust.edu.tw]`.

---

<p align="center">
<sub>© 2025 · Submitted to International Journal of Medical Informatics · Code releasing upon acceptance</sub>
</p>
