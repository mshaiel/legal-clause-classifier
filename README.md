# Legal Clause Classifier (Phi-3-mini + QLoRA)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

I fine-tuned Microsoft's Phi-3-mini-4k-instruct (3.8B parameters) with QLoRA to classify raw contract clauses into 100 provision types from the LexGLUE/LEDGAR benchmark. Updating only 8 million of the model's 3.8 billion parameters, it reaches 62.60% accuracy and 53.52% Macro-F1 on a 500-sample test evaluation. Fully trained encoder models (BERT, LegalBERT) score around 80-82% Macro-F1 on the same task, so there is a real gap -- but the comparison is not apples-to-apples: those models use a fixed 100-way classifier head trained to convergence on the full dataset, while this treats classification as text generation.

The adapter is published on [Hugging Face Hub](https://huggingface.co/mshaiel2004/phi3-legal-clause-qlora).

---

## Results

Evaluated on 500 samples from the official LexGLUE/LEDGAR test split. Encoder baselines are from Chalkidis et al., ACL 2022.

| Model | Macro-F1 | Accuracy | Parameters trained | Precision |
| :--- | :---: | :---: | :---: | :---: |
| LegalBERT | 82.5% | 88.4% | 110M (all) | FP32 |
| RoBERTa-base | 81.5% | 88.2% | 125M (all) | FP32 |
| BERT-base | 80.0% | 87.5% | 110M (all) | FP32 |
| **Phi-3-mini QLoRA (this work)** | **53.52%** | **62.60%** | **8M of 3.8B** | **4-bit NF4** |
| Random baseline | 1.0% | 1.0% | -- | -- |

Mean inference latency: 2,570 ms per clause (0.39 clauses/sec). The accuracy and F1 figures are verified against `data/benchmark_results.json`. The encoder latency figures from the original LexGLUE paper are not reproduced here since I did not run them myself.

---

## Problem and motivation

Contract review is slow and expensive. Automatically tagging clauses by type (indemnification, governing law, termination, and so on) is a useful first step toward automated contract analysis. LexGLUE provides a clean benchmark for this: the LEDGAR subset contains clauses extracted from SEC EDGAR filings, labeled across 100 provision categories.

I picked this task partly because the class imbalance is severe. The most common class has roughly 3,167 training examples; the rarest has 23 (a 137.7x disparity). That forced real engineering decisions rather than just dropping a dataset into a training script.

---

## Data

**LexGLUE/LEDGAR** (Chalkidis et al., ACL 2022): approximately 60,000 training clauses across 100 provision types, sourced from SEC EDGAR contracts. Accessed via HuggingFace Datasets as `coastalcph/lex_glue`, subset `ledgar`. The test evaluation in this repo uses 500 samples from the official test split.

The distribution is heavily skewed toward a handful of common clause types. I computed inverse-frequency weights to keep the rare classes from being overwhelmed during training.

---

## Approach

### Why generative instead of discriminative?

The natural choice for multi-class classification is an encoder (BERT, RoBERTa, LegalBERT) with a linear classification head. I went the other direction: I prompt Phi-3-mini to generate the provision category name as plain text. The motivation was to explore whether an instruction-tuned causal LLM can generalize to legal classification with minimal task-specific supervision, and to produce a model that can, in principle, explain its predictions or be adapted for related tasks (clause extraction, structured JSON output) without retraining the head.

The tradeoffs are real. Generative classification is slower, harder to optimize for class imbalance, and introduces a string-matching step to reconcile the generated text with the label set. On accuracy, it loses to full fine-tuning of much smaller encoders.

### QLoRA

Rather than updating all 3.8B parameters, I used QLoRA (Dettmers et al., NeurIPS 2023): the base model is loaded in 4-bit NF4 quantization and kept frozen; a rank-16 LoRA adapter (alpha=32, dropout=0.05) is injected into the attention and MLP projection layers (`o_proj`, `qkv_proj`, `gate_up_proj`, `down_proj`). Only the 8M adapter parameters are trained.

This brings peak training VRAM down to roughly 4.8 GB, which fits in a free Colab T4 session. Full FP32 fine-tuning of the same model would require over 50 GB.

### Handling class imbalance

I computed inverse-frequency weights:

```
w_c = N / (C * N_c)
```

where N is total training examples (~60,000), C is the number of classes (100), and N_c is the count for class c. Weights are clipped at 15.0 to avoid gradient instability from the long tail. A custom `WeightedTrainer` applies these per sample during training.

Loss is computed only on the assistant's completion tokens, not the prompt. Without this masking, the model learns to copy the input clause rather than classify it.

### Training setup

- 250 steps, effective batch size 32 (4 per device, 8 gradient accumulation steps)
- Learning rate: 2e-4, cosine schedule, 50 warmup steps
- Optimizer: paged AdamW 8-bit
- Gradient checkpointing enabled
- Trained on a Colab T4 GPU

---

## Results and analysis

62.60% accuracy on a 100-class problem is meaningfully above chance (random is 1%), but the roughly 29-point Macro-F1 gap against LegalBERT is real. A few factors explain most of it.

Training budget is the largest one. 250 steps covers a small fraction of the full 60k-sample training set. The encoder baselines trained to convergence on the complete set. Running a full epoch, or several, would likely recover significant F1.

Generative overhead adds a small but nonzero parsing failure rate. When the model produces a string that does not exactly match a label, fuzzy matching is applied. A discriminative argmax classifier never has this problem.

Architecture fit matters too. Encoder models attend bidirectionally, which suits classification well. A causal decoder's inductive bias is predicting the next token, which is the right prior for generation but not obviously for labeling.

The model does best on high-frequency classes like "Governing Laws" and "Indemnification." Performance on rare classes (fewer than ~50 training examples) is poor.

---

## Limitations and what I would do next

The 250-step budget was a practical constraint imposed by Colab session time, not a deliberate design choice. The most useful next step would be running a full epoch with early stopping and seeing where accuracy plateaus.

Other things I would try:

- A frozen Phi-3-mini base with a linear probe, to separate the contribution of the generative framing from the model's base representations.
- Evaluating on the full test split rather than 500 samples for a cleaner published comparison.
- A smaller, faster generative model (Phi-3.5-mini or similar) to test whether scale or architecture choice matters more here.
- Batching inference to improve throughput. At 0.39 clauses/sec, this is not production-ready as-is.

The latency gap against discriminative classifiers (2,570 ms vs. roughly 25 ms for BERT) is fundamental to the generative approach and would persist even with optimization.

---

## How to run it

Clone and install:

```bash
git clone https://github.com/mshaiel/legal-clause-classifier.git
cd legal-clause-classifier
pip install -r requirements-local.txt
```

Run the test suite:

```bash
pytest tests/
```

Classify a single clause (downloads the adapter from the Hub on first run):

```python
from src.inference.pipeline import LegalClauseClassifier

classifier = LegalClauseClassifier(adapter_model_id="mshaiel2004/phi3-legal-clause-qlora")

clause = (
    "This Agreement and all claims arising hereunder shall be governed by, "
    "and construed in accordance with, the laws of the State of Delaware."
)

result = classifier.classify_clause(clause)
print(result["predicted_category"])  # -> "Governing Laws"
print(result["latency_ms"])
```

Reproduce the benchmark evaluation:

```bash
python scripts/evaluate.py --adapter_id mshaiel2004/phi3-legal-clause-qlora --num_samples 500
```

Launch the Gradio demo locally:

```bash
python spaces_app/app.py
```

For GPU training, use `notebooks/02_train_colab.ipynb` in Google Colab, or `scripts/train.py` with `requirements-colab.txt`.

---

## Project structure

```
project2/
├── configs/              # qlora_config.yaml, inference_config.yaml
├── data/                 # Benchmark results and baseline comparisons (JSON)
├── notebooks/            # EDA (01) and Colab training notebook (02)
├── scripts/              # CLI: train.py, evaluate.py, export_adapter.py, eda_ledgar.py
├── spaces_app/           # Gradio web demo (app.py + requirements.txt)
├── src/
│   ├── data/             # Dataset loader, prompt formatting, class weights
│   ├── inference/        # Quantized inference pipeline with CPU fallback
│   ├── model/            # Macro-F1, Micro-F1, fuzzy label matcher
│   └── training/         # WeightedTrainer with completion-token loss masking
├── tests/                # Unit tests for all core components
├── requirements-local.txt
└── requirements-colab.txt
```

---

## References

1. Chalkidis et al. "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English." ACL 2022. [arXiv:2110.00976](https://arxiv.org/abs/2110.00976)
2. Tuggener et al. "LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts." LREC 2020.
3. Abdin et al. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." Microsoft Research, 2024. [arXiv:2404.14219](https://arxiv.org/abs/2404.14219)
4. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

---

MIT License.
