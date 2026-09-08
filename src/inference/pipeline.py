"""
Local Quantized Inference Pipeline for Contract Provision Classification.

Optimized for 4-bit NF4 quantization on budget consumer hardware:
- NVIDIA GTX 1050 (4GB VRAM, Pascal Architecture)
- Automatic CPU fallback if GPU VRAM is constrained
- Robust output parsing into the 100 canonical LexGLUE/LEDGAR categories
"""

import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import yaml

# Internal imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset import LEDGAR_PROVISION_CLASSES, SYSTEM_PROMPT
from src.model.metrics import clean_generated_label, map_text_to_label_id


class LegalClauseClassifier:
    """Production-grade legal provision classifier using 4-bit fine-tuned Phi-3-mini."""

    def __init__(
        self,
        config_path: str = "configs/inference_config.yaml",
        adapter_model_id: str | None = None,
        base_model_id: str | None = None,
        device: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.cfg = self._load_config(self.config_path) if self.config_path.exists() else {}

        self.base_model_id = (
            base_model_id or self.cfg.get("base_model_id") or "microsoft/Phi-3-mini-4k-instruct"
        )
        self.adapter_model_id = (
            adapter_model_id
            or self.cfg.get("adapter_model_id")
            or "mshaiel2004/phi3-legal-clause-qlora"
        )

        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.model = None
        self.tokenizer = None
        self.gen_config = None

        self._init_pipeline()

    def _has_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_config(self, path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_pipeline(self):
        """Load tokenizer and 4-bit or CPU quantized model with compatibility patches."""
        import torch
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DynamicCache,
            GenerationConfig,
        )
        from transformers.generation import GenerationMixin

        # Compatibility patch for Transformers 5.x
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

        print(f"[INFERENCE] Initializing LegalClauseClassifier on {self.device.upper()}...")
        print(f"  Base Model:    {self.base_model_id}")
        print(f"  LoRA Adapter:  {self.adapter_model_id}")

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_model_id if Path(self.adapter_model_id).exists() else self.base_model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Base Model Config
        config = AutoConfig.from_pretrained(self.base_model_id, trust_remote_code=True)
        config.rope_scaling = None

        # 3. Load Model (4-bit NF4 if CUDA available, else CPU float32)
        if self.device == "cuda" and torch.cuda.is_available():
            q_cfg = self.cfg.get("quantization", {})
            compute_dtype = getattr(torch, q_cfg.get("bnb_4bit_compute_dtype", "float16"))

            # Memory budgeting for 4GB Pascal GPU
            max_memory = self.cfg.get("hardware", {}).get(
                "max_memory", {0: "3.2GiB", "cpu": "12GiB"}
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=q_cfg.get("load_in_4bit", True),
                bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype,
            )

            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    config=config,
                    quantization_config=bnb_config,
                    device_map=self.cfg.get("hardware", {}).get("device_map", "auto"),
                    max_memory=max_memory,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    torch_dtype=compute_dtype,
                )
            except Exception as e:
                print(f"[WARN] Failed to load on GPU ({e}). Falling back to CPU...")
                self.device = "cpu"
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    config=config,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                config=config,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        # Ensure GenerationMixin is attached
        if not hasattr(base_model, "generate"):
            base_model.__class__ = type(
                base_model.__class__.__name__,
                (base_model.__class__, GenerationMixin),
                {},
            )

        # Ensure GenerationConfig exists
        try:
            self.gen_config = GenerationConfig.from_pretrained(self.base_model_id)
        except Exception:
            self.gen_config = GenerationConfig(max_new_tokens=32, do_sample=False)
        base_model.generation_config = self.gen_config

        # 4. Attach Adapter
        print(f"[INFERENCE] Attaching fine-tuned adapter: {self.adapter_model_id}...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_model_id)
        self.model.generation_config = self.gen_config
        self.model.eval()

        print("[INFERENCE] Pipeline successfully loaded and ready for contract classification.")

    def classify_clause(
        self,
        clause_text: str,
        max_new_tokens: int = 32,
    ) -> dict[str, Any]:
        """
        Classify a single contract provision into one of the 100 LEDGAR categories.

        Args:
            clause_text: Raw legal contract provision string.
            max_new_tokens: Maximum tokens for generated category name.

        Returns:
            Dictionary containing:
            - predicted_category: Standardized LEDGAR provision category name
            - category_id: Integer index (0-99)
            - raw_response: Raw text generated by model
            - latency_ms: Inference execution time in milliseconds
            - device_used: 'cuda' or 'cpu'
        """
        clause_clean = clause_text.strip()
        if not clause_clean:
            return {
                "predicted_category": "Unknown",
                "category_id": -1,
                "raw_response": "",
                "latency_ms": 0.0,
                "device_used": self.device,
            }

        import torch

        prompt = (
            f"<|user|>\n"
            f"{SYSTEM_PROMPT}\n\n"
            f'Contract Provision:\n"{clause_clean}"\n<|end|>\n'
            f"<|assistant|>\n"
        )

        inputs = self.tokenizer(
            prompt,
            max_length=self.cfg.get("max_seq_length", 256),
            truncation=True,
            return_tensors="pt",
        )

        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.gen_config,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        cleaned_text = clean_generated_label(raw_text)
        cat_id = map_text_to_label_id(raw_text)
        canonical_category = LEDGAR_PROVISION_CLASSES[cat_id]

        return {
            "predicted_category": canonical_category,
            "category_id": cat_id,
            "cleaned_label": cleaned_text,
            "raw_response": raw_text.strip(),
            "latency_ms": round(elapsed_ms, 2),
            "device_used": self.device,
            "clause_preview": clause_clean[:120] + "..."
            if len(clause_clean) > 120
            else clause_clean,
        }

    def batch_classify(self, clauses: list[str]) -> list[dict[str, Any]]:
        """Classify multiple contract clauses sequentially."""
        return [self.classify_clause(c) for c in clauses]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classify contract clauses using fine-tuned Phi-3")
    parser.add_argument(
        "--clause",
        type=str,
        default=(
            "This Agreement shall be construed, interpreted, and governed by and in accordance "
            "with the laws of the State of Delaware, without giving effect to conflict of laws."
        ),
        help="Contract provision text to classify",
    )
    parser.add_argument(
        "--adapter_id",
        type=str,
        default="mshaiel2004/phi3-legal-clause-qlora",
        help="Hugging Face adapter repository ID or local checkpoint path",
    )
    args = parser.parse_args()

    classifier = LegalClauseClassifier(adapter_model_id=args.adapter_id)
    result = classifier.classify_clause(args.clause)

    print("\n" + "=" * 70)
    print("⚖️ LEGAL PROVISION CLASSIFICATION RESULT")
    print("=" * 70)
    print(f'Clause Preview:      "{result["clause_preview"]}"')
    print(f"Predicted Category:  {result['predicted_category']} (ID: {result['category_id']})")
    print(f'Raw Generation:      "{result["raw_response"]}"')
    print(f"Inference Latency:   {result['latency_ms']:.2f} ms")
    print(f"Hardware Device:     {result['device_used'].upper()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
