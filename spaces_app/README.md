---
title: LexGLUE Legal Contract Provision Classifier
emoji: ⚖️
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# ⚖️ LexGLUE Legal Contract Provision Classifier
### Instruction-Tuned Phi-3-mini QLoRA on 100 SEC EDGAR Provision Categories

Interactive demonstration for contract clause classification across the 100 canonical LEDGAR provision categories from the **LexGLUE benchmark** (*Chalkidis et al., ACL 2022*).

- **Base Model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **Quantization:** 4-bit NormalFloat (NF4) with double quantization
- **Trained Adapter:** [`mshaiel2004/phi3-legal-clause-qlora`](https://huggingface.co/mshaiel2004/phi3-legal-clause-qlora)
- **Dataset:** LexGLUE / LEDGAR (60,000 training provisions, 100 categories)
- **Evaluation Accuracy:** **62.60%** across 100 classes (vs. 1.0% random baseline)
