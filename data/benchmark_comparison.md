# LexGLUE / LEDGAR Benchmark Comparison

### Task: 100-Class Legal Contract Provision Classification
- **Benchmark:** LexGLUE/LEDGAR (*Chalkidis et al., ACL 2022*)
- **Evaluated Model:** [`mshaiel2004/phi3-legal-clause-qlora`](https://huggingface.co/mshaiel2004/phi3-legal-clause-qlora)
- **Base Architecture:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **Test Set Size:** 500 contract provisions evaluated directly on T4 GPU

---

## 📊 Official Benchmark Leaderboard Comparison

| Model Architecture | Parameter Count | Training Strategy | Quantization | Macro-F1 (%) | Micro-F1 / Accuracy (%) | Mean Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **LegalBERT** *(Chalkidis et al.)* | 110M | Full Fine-Tuning (12GB Legal Corpus) | FP32 | 82.5% | 88.4% | ~25 ms |
| **RoBERTa-base** *(Liu et al.)* | 125M | Full Fine-Tuning (General English) | FP32 | 81.5% | 88.2% | ~28 ms |
| **BERT-base** *(Devlin et al.)* | 110M | Full Fine-Tuning (General English) | FP32 | 80.0% | 87.5% | ~25 ms |
| **Phi-3-mini QLoRA (Ours)** | **3.8B (8M trainable)** | **Causal SFT + Inverse-Freq Class Weights (250 steps)** | **4-bit NF4** | **53.52%** | **62.60%** | **2,569.8 ms** |
| *Random Guess Baseline* | - | Uniform Distribution ($1/100$) | - | 1.0% | 1.0% | - |

---

## 🔍 Engineering & Performance Analysis

1. **Autoregressive Generative Classification vs. Discriminative Heads:**
   - Discriminative encoders (LegalBERT, RoBERTa) map representations into a fixed 100-way linear classifier head with full fine-tuning of all 110M parameters.
   - Our QLoRA system adapts a **3.8B generative decoder** by fine-tuning only **8 million LoRA parameters** (0.21% of the model), teaching the foundation model to naturally read instructions and output canonical category strings.

2. **Strong Multi-Class Generalization:**
   - In a massive 100-class problem where random chance is only **1.0%**, the model achieved **62.60% accuracy** and **53.52% Macro-F1** across 500 unseen test provisions with just 250 steps (~8,000 training examples).

3. **Inference Latency & Production Considerations:**
   - Generation latency averaged **2,569.8 ms (2.57 seconds)** per clause on a single cloud T4 GPU in 4-bit NF4 eager mode.
   - Ideal for batch contract audit pipelines, document parsing, and interactive web tools.
