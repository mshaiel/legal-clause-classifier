"""Unit tests for the LegalClauseClassifier inference pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.inference.pipeline import LegalClauseClassifier

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_empty_clause_handling():
    """Verify empty or whitespace-only clauses are handled gracefully."""
    with patch.object(LegalClauseClassifier, "_init_pipeline", return_value=None):
        classifier = LegalClauseClassifier(config_path="configs/inference_config.yaml")
        result = classifier.classify_clause("   ")
        assert result["predicted_category"] == "Unknown"
        assert result["category_id"] == -1
        assert result["latency_ms"] == 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="Requires torch for tensor generation mocking")
def test_classify_clause_mocked_generation():
    """Verify that simulated model output is correctly parsed and mapped to canonical classes."""
    with patch.object(LegalClauseClassifier, "_init_pipeline", return_value=None):
        classifier = LegalClauseClassifier(config_path="configs/inference_config.yaml")
        classifier.device = "cpu"

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.eos_token_id = 32000
        mock_tokenizer.decode.return_value = "Governing Laws<|end|>"
        classifier.tokenizer = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100, 101]])
        classifier.model = mock_model
        classifier.gen_config = MagicMock()

        sample_clause = "This Agreement shall be governed by Delaware law."
        result = classifier.classify_clause(sample_clause)

        assert result["predicted_category"] == "Governing Laws"
        assert result["category_id"] in range(100)
        assert result["latency_ms"] >= 0.0
        assert "Delaware" in result["clause_preview"]


@pytest.mark.skipif(not HAS_TORCH, reason="Requires torch for tensor generation mocking")
def test_batch_classification():
    """Verify batch classify processes multiple clauses sequentially."""
    with patch.object(LegalClauseClassifier, "_init_pipeline", return_value=None):
        classifier = LegalClauseClassifier(config_path="configs/inference_config.yaml")
        classifier.device = "cpu"

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.eos_token_id = 32000
        mock_tokenizer.decode.return_value = "Severability"
        classifier.tokenizer = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 99]])
        classifier.model = mock_model
        classifier.gen_config = MagicMock()

        clauses = [
            "If any provision is held invalid, the remainder shall continue in effect.",
            "Any notice required hereunder shall be in writing.",
        ]
        results = classifier.batch_classify(clauses)
        assert len(results) == 2
        for r in results:
            assert r["predicted_category"] == "Severability"
