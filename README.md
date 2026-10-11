# KMR-RAG: Knowledge-guided Multi-document Retrieval-Augmented Generation and Refinement for Cancer Registry Abstraction

This repository provides the source code accompanying the paper:

> **KMR-RAG: Knowledge-guided Multi-document Retrieval-Augmented Generation and Refinement for Automated Cancer Registry Abstraction**  
> *PLOS ONE* (under review)

---

## Overview

KMR-RAG is a training-free framework for automated cancer registry abstraction from longitudinal clinical records. It combines three components:

- **Field-specific retrieval**: dense semantic retrieval (for narrative and categorical fields) and BM25 sparse retrieval (for numerical fields), guided by a field taxonomy.
- **Knowledge-augmented prompting**: each LLM inference call receives registry coding rules and a structured prompt template alongside the retrieved passages.
- **Iterative refinement**: an initial extraction pass (Pass 1) is followed by one or more refinement passes (Passes 2–*k*) in which the LLM revises its answer using updated retrieved context.

The framework is evaluated on colorectal cancer records from four hospitals in Taiwan (KMUH, CMUH, KAFGH, PPH) using four locally deployable LLMs: Mistral-7B-v0.3, LLaMA-3.1-8B-Instruct, DeepSeek-R1-Distill-LLaMA-8B, and DeepSeek-R1-Distill-Qwen-7B, deployed via [vLLM](https://github.com/vllm-project/vllm) on an NVIDIA RTX 3090.

---

## Repository Structure

```
KMR-RAG/
├── kmr_rag/
│   ├── agents/
│   │   ├── agent_system.py       # Core CancerRegistryAgentSystem (LlamaIndex-based)
│   │   └── toolspecs.py          # Retrieval tool specs and prompt templates
│   └── datasets/
│       └── utils.py              # CRCPlainTextReader: multi-hospital text loader
├── notebooks/
│   ├── kmr_rag_main.ipynb             # Main experiment notebook (KMR-RAG evaluation)
│   ├── experiment_runner.ipynb        # Experiment runner with ablation configurations
│   ├── llamaindex_crc_multi_doc.ipynb # LlamaIndex multi-document CRC exploration
│   └── llamaindex_multi_doc_exploration.ipynb  # LlamaIndex multi-document baseline exploration
├── demo_app.py                   # Streamlit demo application
└── README.md
```

---

## Requirements

- Python 3.10+
- [LlamaIndex](https://github.com/run-llama/llama_index) (llama-index-core, llama-index-llms-huggingface, llama-index-embeddings-huggingface)
- [vLLM](https://github.com/vllm-project/vllm) (for local LLM serving)
- [Streamlit](https://streamlit.io) (for the demo application)
- PyTorch with CUDA

Install dependencies:

```bash
pip install llama-index llama-index-llms-openai-like llama-index-embeddings-huggingface \
            streamlit streamlit-timeline streamlit-markmap \
            transformers sentence-transformers rank_bm25
```

---

## LLM Serving with vLLM

The framework calls LLMs through a local OpenAI-compatible endpoint served by vLLM. Example commands for each supported backbone:

**Mistral-7B-Instruct-v0.3**
```bash
docker run --gpus all --rm -it \
  -v $PWD/models/LLM/mistralai/Mistral-7B-Instruct-v0.3:/model \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name Mistral-7B-Instruct-v0.3 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --load-format safetensors \
  --disable-log-stats
```

**LLaMA-3.1-8B-Instruct**
```bash
docker run --gpus all --rm -it \
  -v $PWD/models/LLM/meta-llama/Meta-Llama-3.1-8B-Instruct:/model \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --load-format safetensors \
  --disable-log-stats \
  --enable-prefix-caching
```

**DeepSeek-R1-Distill-LLaMA-8B**
```bash
docker run --gpus all --rm -it \
  -v $PWD/models/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B:/model \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name DeepSeek-R1-Distill-Llama-8B \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --load-format safetensors \
  --disable-log-stats \
  --enable-prefix-caching
```

**DeepSeek-R1-Distill-Qwen-7B**
```bash
docker run --gpus all --rm -it \
  -v $PWD/models/LLM/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:/model \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name DeepSeek-R1-Distill-Qwen-7B \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --load-format safetensors \
  --disable-log-stats \
  --enable-prefix-caching
```

By default, the notebooks connect to `http://localhost:8000/v1`. Update the `api_base` parameter to match your server address.

---

## Embedding Models

Retrieval embedding models are loaded from a local `.models/Retriever/` directory. The experiments evaluated a range of multilingual and biomedical embedding models. Place each model under:

```
.models/Retriever/<model-name>/
```

Models used in the paper include multilingual-e5-large-instruct, GIST-large-Embedding-v0, bge-large-en-v1.5, mxbai-embed-large-v1, and others. Selection criteria are described in Section 2.2 of the paper.

---

## Dataset

The clinical datasets used in this study contain protected health information and are not publicly available due to patient privacy regulations and institutional data governance requirements. Researchers who wish to access the data for verification or collaborative purposes may contact the corresponding author directly.

The `kmr_rag/datasets/utils.py` module provides a `CRCPlainTextReader` class that loads pre-processed, de-identified clinical reports in plain text format. The expected directory structure for each hospital site is:

```
<hospital_id>/
  testset/
    <patient_id_1>.txt
    <patient_id_2>.txt
    ...
```

Each `.txt` file contains one patient's longitudinal clinical reports, with individual reports separated by a delimiter line (`############################################`).

---

## Running the Demo Application

```bash
streamlit run demo_app.py
```

Place the test set under `datasets/crc/testset/` and ensure the vLLM endpoint is running before launching.

---

## Running Experiments

See `notebooks/kmr_rag_main.ipynb` and `notebooks/experiment_runner.ipynb` for the full experiment pipeline, including:

- Hospital and retrieval model selection
- Ablation over retrieval depth (*k* ∈ {1, 3, 6}), chunk size (*c* ∈ {256, 512, 1024}), and refinement passes
- Per-field and overall macro-average F1 evaluation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{KMR-RAG2026,
  title   = {{KMR-RAG}: Knowledge-guided Multi-document Retrieval-Augmented Generation
             and Refinement for Automated Cancer Registry Abstraction},
  author  = {[Author list to be completed upon publication]},
  journal = {PLOS ONE},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

The source code in this repository is released under the [MIT License](LICENSE).

---

## Contact

For questions about the code or requests regarding the clinical datasets, please contact the corresponding author at the affiliation listed in the paper.
