# LexGLUE-LEDGAR Legal Clause Classifier: QLoRA Fine-Tuning & Deployment Blueprint

### Industry-Grade Portfolio Project — Legal Contract Provision Classification
**Base Model:** `microsoft/Phi-3-mini-4k-instruct` · **Technique:** QLoRA (4-bit NF4) · **Dataset:** LexGLUE/LEDGAR (100 classes)  
**Training Compute:** Google Colab Free Tier (T4 GPU, 16GB VRAM) · **Local Testing:** NVIDIA GTX 1050 (4GB VRAM) / Intel i7-7700HQ  
**Checkpoints:** Direct Hugging Face Hub push (`push_to_hub=True`) · **Demo Hosting:** Hugging Face Spaces (Gradio)

---

## 1. Architecture Decisions & Mathematical Formulation

### 1.1 Causal Instruction Tuning with `SFTTrainer`
Rather than attaching an external linear sequence classification head, the model is trained via causal instruction tuning to natively understand instruction prompts and generate the exact provision category name.

#### Prompt Template (Phi-3-mini Instruct format)
```text
<|user|>
You are an expert legal AI assistant. Classify the following contract provision into exactly one of the 100 standard LEDGAR provision categories. Respond with only the exact category name.

Contract Provision:
"""{provision_text}"""<|end|>
<|assistant|>
{category_name}<|end|>
```

#### Training Loss Masking
Using TRL's `DataCollatorForCompletionOnlyLM` with response template `<|assistant|>\n`, tokens prior to the assistant response are masked out with label `-100`. Loss is computed **strictly on the generated category tokens**.

### 1.2 Class-Weighted Cross-Entropy Loss Formulation
LEDGAR exhibits severe class imbalance across its 100 categories (ranging from thousands of samples in majority classes like "Governing Laws" to under 50 samples in tail classes).

To inject class weights into `SFTTrainer`:
1. **Precompute normalized inverse class frequency weights** $W \in \mathbb{R}^{100}$ from the training set labels:
   $$w_c = \frac{N}{C \cdot N_c}$$
   Where $N$ is total training samples (60,000), $C=100$, and $N_c$ is the frequency count of class $c$.
2. **In our custom `WeightedSFTTrainer` (subclass of `trl.SFTTrainer`)**, override `compute_loss`:
   - Map each batch sample to its ground-truth target class index $c_i \in \{0, \dots, 99\}$.
   - Compute unreduced per-token cross-entropy loss $\mathcal{L}_{\text{token}}$.
   - Average token loss across the completion tokens for each sample $i$ to obtain $\mathcal{L}_i$.
   - Weight each sample's loss by its class weight $w_{c_i}$ and normalize across the batch:
     $$\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B w_{c_i} \cdot \mathcal{L}_i}{\sum_{i=1}^B w_{c_i}}$$

---

## 2. Project Directory Layout

```
project2/
├── .github/
│   └── workflows/
│       └── lint.yml                  # Ruff lint & format check on push
├── configs/
│   ├── qlora_config.yaml             # BitsAndBytes, LoRA, and Training hyperparams
│   └── inference_config.yaml         # Local GTX 1050 4-bit inference settings
├── notebooks/
│   ├── 01_eda_ledgar.ipynb           # EDA, label mapping, class distribution analysis
│   ├── 02_train_colab.ipynb          # End-to-end Colab training notebook
│   └── 03_evaluation.ipynb           # Post-training analysis & confusion matrix
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # LEDGAR loader, prompt formatting, train/val/test
│   │   └── class_weights.py          # Balanced inverse-frequency class weights
│   ├── model/
│   │   ├── __init__.py
│   │   ├── builder.py                # Model loader (4-bit NF4) & LoRA setup
│   │   └── metrics.py                # Generation parser, Macro/Micro-F1, Accuracy
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                # WeightedSFTTrainer with sample-level class weights
│   └── inference/
│       ├── __init__.py
│       └── pipeline.py               # Memory-efficient local pipeline (GTX 1050 / CPU)
├── scripts/
│   ├── train.py                      # Modular training CLI script
│   ├── evaluate.py                   # Standalone test-set evaluation CLI script
│   └── export_adapter.py             # Adapter verification and HF Hub synchronization
├── spaces_app/
│   ├── app.py                       # Hugging Face Spaces Gradio app entrypoint
│   ├── requirements.txt              # Cloud deployment dependencies
│   └── README.md                     # Spaces configuration metadata
├── tests/
│   ├── test_dataset.py               # Data formatting & prompt tests
│   └── test_weights.py               # Class weight calculation tests
├── .env.example                      # Template for WANDB_API_KEY, HF_TOKEN
├── .gitignore                        # Comprehensive ignores for models, cache, envs
├── pyproject.toml                    # Ruff & code formatting configuration
├── requirements-colab.txt            # Pinned requirements for Colab T4
├── requirements-local.txt            # Local dev, EDA, Gradio preview, inference
└── README.md                         # Recruiter-grade project documentation
```

---

## 3. Phased Execution Roadmap

We will proceed strictly **phase by phase in collaborative sessions**:

### Phase 1: Repository Architecture & Environment Setup (Local Antigravity)
- Initialize clean modular repository structure.
- Setup `.gitignore`, `pyproject.toml`, `.env.example`, `requirements-local.txt`, `requirements-colab.txt`.
- Build dataset verification and Exploratory Data Analysis module (`notebooks/01_eda_ledgar.ipynb`):
  - 100-class label distribution, class imbalance statistics, Gini index.
  - Verification of label mapping (integer IDs $\leftrightarrow$ category names).

### Phase 2: Core Data Pipeline & Custom Weighted SFTTrainer (Local Antigravity)
- Implement `src/data/dataset.py`: prompt formatter for Phi-3 instruct syntax.
- Implement `src/data/class_weights.py`: compute balanced weights across LEDGAR 100 classes.
- Implement `src/model/metrics.py`: string extraction, category matching, and Macro/Micro/Weighted F1 calculation.
- Implement `src/training/trainer.py`: `WeightedSFTTrainer` with sample-level class weighting.
- Run local unit tests (`tests/`) to ensure prompt collation and loss computation logic are 100% sound before touching GPU compute.

### Phase 3: Colab Training Artifacts & Execution (Google Colab T4)
- Create `configs/qlora_config.yaml` (BitsAndBytes 4-bit NF4, LoRA rank 16, alpha 32, target modules for Phi-3).
- Create `notebooks/02_train_colab.ipynb` and `scripts/train.py` with:
  - Hugging Face Hub direct checkpoint push (`push_to_hub=True`, private repo).
  - Weights & Biases (`wandb`) real-time loss and metric tracking.
  - T4 memory optimizations: `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` (effective batch size 32), `optim="paged_adamw_8bit"`, `bf16=True`.
- **Interactive Checkpoint:** Hand off to you to execute on Google Colab T4. You will report back with training logs, W&B run link, and adapter repo status.

### Phase 4: Evaluation & Post-Training Analysis
- Evaluate the fine-tuned adapter on the full LEDGAR test set (10,000 samples).
- Compute benchmark comparison table vs official LexGLUE baselines (BERT-base, RoBERTa-base, LegalBERT).
- Generate confusion matrix and tail-class recall analysis in `notebooks/03_evaluation.ipynb`.

### Phase 5: Local Quantized Inference (GTX 1050 4GB)
- Implement `src/inference/pipeline.py`:
  - 4-bit NF4 quantization configured with `bnb_4bit_compute_dtype=torch.float16` for Pascal architecture.
  - Local memory footprint strictly constrained within 2.5GB of the 4GB VRAM limit.
  - Clean CPU fallback option.
  - Verification script on sample contract clauses.

### Phase 6: Hugging Face Spaces Gradio Demo & Recruiter-Grade Portfolio
- Build `spaces_app/app.py`, `spaces_app/requirements.txt`, and Spaces `README.md`.
- Deploy/test Gradio interface with confidence scores, provision category explanations, and pre-loaded clauses.
- Author recruiter-grade GitHub `README.md` with architecture diagrams, W&B curves, LexGLUE benchmark comparisons, and live Spaces links.

---

## 4. Verification Plan

### Automated Tests (Local Antigravity)
- `tests/test_dataset.py`: Verify prompt tokenization, completion mask creation, label decoding.
- `tests/test_weights.py`: Verify class weight dimensions $(100,)$ and weight normalization.
- `ruff check .`: Enforce PEP8 and import cleanliness.

### Colab & Cloud Verification
- Colab training loss convergence on W&B.
- Successful push of adapter weights and tokenizer to Hugging Face Hub.
- Gradio app live test on Hugging Face Spaces.
