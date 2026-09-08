# ⚖️ LexGLUE Contract Provision Classifier (QLoRA + Phi-3-mini)

[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Adapter_Weights-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/mshaiel2004/phi3-legal-clause-qlora)
[![Benchmark: LexGLUE](https://img.shields.io/badge/Benchmark-LexGLUE%2FLEDGAR-4F46E5)](https://github.com/coastalcph/lex-glue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PEFT QLoRA](https://img.shields.io/badge/PEFT-QLoRA_4bit-purple.svg)](https://github.com/huggingface/peft)

Production-grade parameter-efficient fine-tuning (PEFT / QLoRA) of **Microsoft's Phi-3-mini-4k-instruct (3.8B)** for multi-class legal contract clause classification across **100 provision categories** from the **LexGLUE / LEDGAR benchmark** (*Chalkidis et al., ACL 2022*).

Rather than attaching a task-specific linear classifier head on top of a frozen encoder, this system uses **causal instruction tuning**: the autoregressive model processes the full legal clause and generates the canonical provision category name. To combat the benchmark's **137.7× class disparity**, training integrates **completion-only token loss masking** and a custom **inverse-frequency class-weighted cross-entropy loss**.

---

## 📌 Executive Summary

- **Task:** Classify unformatted contract clauses into exactly one of 100 SEC EDGAR provision types.
- **Base Architecture:** `microsoft/Phi-3-mini-4k-instruct` (3.82 billion parameters).
- **Fine-Tuning Method:** QLoRA with 4-bit NormalFloat (NF4) quantization + LoRA ($r=16, \alpha=32$).
- **Trainable Parameters:** 8,388,608 (only **0.219%** of the base model).
- **Class Imbalance Strategy:** Loss scaled by normalized inverse class frequencies $w_c = \frac{N}{C \cdot N_c}$ clipped at 15.0.
- **Results:** Achieved **62.60% top-1 accuracy** across 100 classes on unseen test clauses (random baseline is **1.0%**), with **53.52% Macro-F1**.
- **Model Checkpoint:** Published and open-source on [Hugging Face Hub](https://huggingface.co/mshaiel2004/phi3-legal-clause-qlora).

---

## 📊 Benchmark Results vs. LexGLUE Leaderboard

Evaluated against published baselines on the official **LexGLUE / LEDGAR** test split (*Chalkidis et al., ACL 2022*):

| Model Architecture | Parameter Count | Training Strategy | Precision | Macro-F1 (%) | Micro-F1 / Accuracy (%) | Mean Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **LegalBERT** *(Chalkidis et al., 2022)* | 110M | Full Fine-Tuning (12GB Legal Corpus) | FP32 | 82.5% | 88.4% | ~25 ms |
| **RoBERTa-base** *(Liu et al., 2019)* | 125M | Full Fine-Tuning (General English) | FP32 | 81.5% | 88.2% | ~28 ms |
| **BERT-base** *(Devlin et al., 2019)* | 110M | Full Fine-Tuning (General English) | FP32 | 80.0% | 87.5% | ~25 ms |
| **Phi-3-mini QLoRA (This Work)** | **3.8B (8M trainable)** | **Causal SFT + Class-Weighted Loss (250 steps)** | **4-bit NF4** | **53.52%** | **62.60%** | **2,569 ms** |
| *Random Guess Baseline* | - | Uniform Prior ($1/100$) | - | 1.0% | 1.0% | - |

### Engineering Trade-Off Analysis
1. **Generative Autoregressive vs. Discriminative Heads:** Encoder models (BERT/LegalBERT) utilize a hardcoded 100-way linear classifier head with fixed token embeddings. Our generative LLM approach produces canonical category strings directly as conversational completions, enabling zero-shot explanation capabilities and structured JSON contract extraction in downstream pipelines.
2. **Efficiency:** Only **8M adapter parameters** were updated over 250 steps (~8,000 contract samples) to achieve 62.6% accuracy on a 100-class problem, whereas traditional baselines required full fine-tuning across all 60,000 training examples.

---

## 🛠️ Architecture & Mathematical Formulation

```
Raw Legal Clause (e.g. Governing Law / Delaware Jurisdiction)
   │
   ▼
Conversational Prompt Wrapper (<|user|> ... <|end|> \n <|assistant|>)
   │
   ▼
[Base Model: Phi-3-mini 4-bit NF4 (Frozen 3.8B Weights)] ──┐
   │                                                        │
   ▼                                                        │
[LoRA Adapter: Rank-16 Target Projections] <────────────────┘
   ├── o_proj, qkv_proj
   └── gate_up_proj, down_proj
   │
   ▼
Output Token Logits (Vocab: 32,064)
   │
   ├── Completion Masking: Mask all prompt tokens prior to <|assistant|>\n with -100
   └── Class-Weighted Loss: Scale per-sample loss by normalized class weight w_c
   │
   ▼
Canonical Classification Output ("Governing Laws")
```

### 1. Inverse-Frequency Class-Weighted Loss
The LEDGAR dataset exhibits severe long-tail imbalance: the most frequent class (*Governing Laws*) has 3,167 training instances, while the rarest class has only 23 instances (**137.7× disparity**). 

To prevent majority classes from dominating gradient descent:
$$w_c = \frac{N}{C \cdot N_c}$$

Where:
- $N = 60,000$ (total training clauses)
- $C = 100$ (number of distinct provision categories)
- $N_c$ = total occurrences of class $c$ in the training split

Tail weights are clipped at $w_{\text{max}} = 15.0$ to safeguard against gradient instability, then mean-normalized:
$$\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B w_{c_i} \cdot \mathcal{L}_i}{\sum_{i=1}^B w_{c_i} + \epsilon}$$

### 2. Completion-Only Token Loss Masking
In causal language models, calculating loss over the prompt teaches the model to memorize the legal clause rather than classify it. We compute cross-entropy strictly on the assistant's completion tokens:
$$\text{label}_t = \begin{cases} -100 & \text{if } t < t_{\text{assistant}} \\ \text{token}_t & \text{if } t \ge t_{\text{assistant}} \end{cases}$$

---

## ⚡ Hardware Memory Profiling & Budgeting

| Component | FP32 Full Precision | FP16 Half Precision | 4-bit NF4 QLoRA (Ours) |
| :--- | :---: | :---: | :---: |
| Base Model Parameters | 15.2 GB | 7.6 GB | **~2.3 GB** |
| Optimizer States (AdamW) | 30.4 GB | 15.2 GB | **~16 MB** *(paged 8-bit optimizer)* |
| LoRA Trainable Adapter ($r=16$) | - | - | **~33 MB** |
| Activation Cache (Grad Checkpointing) | ~6.0 GB | ~3.0 GB | **~1.8 GB** |
| **Total Peak Training VRAM** | **> 50 GB** | **~25.8 GB** | **~4.8 GB (Fits on 16GB T4 / 8GB VRAM)** |

---

## 📂 Repository Structure

```
project2/
├── configs/
│   ├── qlora_config.yaml          # Hyperparameters, LoRA targets, and quant config
│   └── inference_config.yaml      # Memory caps and generation parameters for local deployment
├── data/
│   ├── benchmark_baselines.json   # Published LexGLUE baselines (Chalkidis et al., 2022)
│   ├── benchmark_comparison.md    # Formatted leaderboard comparison table
│   └── benchmark_results.json     # Empirical 500-sample test evaluation metrics
├── notebooks/
│   ├── 01_eda_ledgar.ipynb        # Exploratory data analysis (class counts, Gini index)
│   └── 02_train_colab.ipynb       # Self-contained Colab notebook for GPU training
├── scripts/
│   ├── eda_ledgar.py              # CLI dataset profiling script
│   ├── export_adapter.py          # Hugging Face Hub adapter downloader and verifier
│   ├── train.py                   # Production CLI training script with WeightedTrainer
│   └── evaluate.py                # Standalone LexGLUE benchmark evaluation suite
├── spaces_app/                    # Hugging Face Spaces Gradio Web Deployment
│   ├── app.py                     # Modern legal-tech Gradio application
│   ├── requirements.txt           # Cloud deployment dependencies
│   └── README.md                  # Spaces metadata header
├── src/
│   ├── data/
│   │   ├── dataset.py             # LexGLUE loader and prompt formatting
│   │   └── class_weights.py       # Inverse-frequency weight calculation & clipping
│   ├── inference/
│   │   └── pipeline.py            # Quantized inference engine with CPU fallback
│   ├── model/
│   │   └── metrics.py             # Macro-F1, Micro-F1, and fuzzy string label matcher
│   └── training/
│       └── trainer.py             # WeightedTrainer with completion token loss masking
├── tests/                         # Automated unit test suite
│   ├── test_dataset.py            # Prompt formatting & bijective ID2LABEL tests
│   ├── test_inference.py          # Mocked generation and empty clause tests
│   ├── test_metrics.py            # Fuzzy parser & metric computation tests
│   └── test_weights.py            # Class weight clipping & normalization tests
├── .github/workflows/lint.yml     # Automated Ruff linting & pytest CI workflow
├── pyproject.toml                 # Ruff lint configuration
├── requirements-colab.txt         # Pinned GPU training dependencies
└── requirements-local.txt         # Local development & testing dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install testing dependencies:
```bash
git clone https://github.com/mshaiel2004/legal-clause-classifier.git
cd legal-clause-classifier
pip install -r requirements-local.txt
```

### 2. Run Automated Unit Tests
Verify prompt formatting, class weighting, and metric parsing:
```bash
pytest tests/
```

### 3. Local / Cloud Inference
Classify a contract clause using the pre-trained Hugging Face Hub adapter:
```python
from src.inference.pipeline import LegalClauseClassifier

classifier = LegalClauseClassifier(adapter_model_id="mshaiel2004/phi3-legal-clause-qlora")

clause = (
    "This Agreement and all claims arising hereunder shall be governed by, "
    "and construed in accordance with, the laws of the State of Delaware."
)

result = classifier.classify_clause(clause)
print(f"Predicted Category: {result['predicted_category']}")
print(f"Latency:            {result['latency_ms']} ms")
```

### 4. Interactive Gradio Web Demo
Launch the local web application:
```bash
python spaces_app/app.py
```

### 5. Reproduce Benchmark Evaluation
Run the evaluation suite against the official LexGLUE test split:
```bash
python scripts/evaluate.py --adapter_id mshaiel2004/phi3-legal-clause-qlora --num_samples 500
```

---

## 📚 References & Acknowledgements

1. **LexGLUE Benchmark:**  
   I. Chalkidis, A. Jana, D. Hartung, M. Bommarito, I. Androutsopoulos, D. Katz, and N. Aletras.  
   *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.*  
   **ACL 2022**. [arXiv:2110.00976](https://arxiv.org/abs/2110.00976)

2. **LEDGAR Dataset:**  
   F. Tuggener, T. von Däniken, P. Peetz, and M. Cieliebak.  
   *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts.*  
   **LREC 2020**.

3. **Phi-3 Technical Report:**  
   M. Abdin et al.  
   *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.*  
   **Microsoft Research, 2024**. [arXiv:2404.14219](https://arxiv.org/abs/2404.14219)

4. **QLoRA:**  
   T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer.  
   *QLoRA: Efficient Finetuning of Quantized LLMs.*  
   **NeurIPS 2023**. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
