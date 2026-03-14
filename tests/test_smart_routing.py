"""Tests for smart routing / task classification in quality gate."""

import pytest

from avp.rosetta.quality import (
    TaskClassification,
    TransferQualityConfig,
    TransferQualityResult,
    assess_transfer,
    classify_task,
)


class TestClassifyTask:
    """Tests for classify_task() prompt classification."""

    def test_math_problem_classified_as_structured(self):
        prompt = "Solve: 24 * 17 + 3 = ?"
        result = classify_task(prompt, prompt_tokens=50)
        assert result.task_type == "structured"
        assert result.score > 0

    def test_code_problem_classified_as_structured(self):
        prompt = "def fibonacci(n):\n    # implement this function\n    return result"
        result = classify_task(prompt, prompt_tokens=30)
        assert result.task_type == "structured"
        assert result.score > 0

    def test_comprehension_classified_as_comprehension(self):
        prompt = (
            "The industrial revolution began in Britain in the late 18th century. "
            "It was a period of great change in manufacturing, mining, and transport.\n\n"
            "The textile industry was the first to adopt new methods of production.\n\n"
            "Steam power was central to the changes.\n\n"
            "Who started the industrial revolution?"
        )
        result = classify_task(prompt, prompt_tokens=800)
        assert result.task_type == "comprehension"
        assert result.score < 0

    def test_high_digit_density_is_structured(self):
        prompt = "Calculate 123 + 456 * 789 - 321 / 654 = ?"
        result = classify_task(prompt, prompt_tokens=20)
        assert result.features["digit_density"] > 0
        assert result.task_type == "structured"

    def test_no_digits_no_digit_bonus(self):
        prompt = "Tell me about the history of France"
        result = classify_task(prompt, prompt_tokens=10)
        assert result.features["digit_density"] == 0

    def test_long_prompt_penalized(self):
        prompt = "Some text"
        result = classify_task(prompt, prompt_tokens=600)
        assert result.features["token_count"] == -2

    def test_short_prompt_rewarded(self):
        prompt = "Solve: 2 + 2"
        result = classify_task(prompt, prompt_tokens=50)
        assert result.features["token_count"] == 1

    def test_math_markers_detected(self):
        prompt = r"Find \frac{x}{y} where \sqrt{x} = 5"
        result = classify_task(prompt, prompt_tokens=20)
        assert result.features["markers"] > 0

    def test_code_markers_detected(self):
        prompt = "```python\ndef foo():\n    return 42\n```"
        result = classify_task(prompt, prompt_tokens=20)
        assert result.features["markers"] > 0

    def test_comprehension_questions_detected(self):
        prompt = "Based on the passage above, who was the first president?"
        result = classify_task(prompt, prompt_tokens=100)
        assert result.features["question_type"] < 0

    def test_structured_starters_detected(self):
        prompt = "Solve the following equation: 3x + 5 = 20"
        result = classify_task(prompt, prompt_tokens=50)
        assert result.features["question_type"] > 0

    def test_multi_paragraph_penalized(self):
        prompt = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nQuestion?"
        result = classify_task(prompt, prompt_tokens=100)
        assert result.features["paragraphs"] < 0

    def test_returns_task_classification_dataclass(self):
        result = classify_task("test prompt", prompt_tokens=50)
        assert isinstance(result, TaskClassification)
        assert result.task_type in ("structured", "comprehension")
        assert isinstance(result.score, int)
        assert isinstance(result.features, dict)

    def test_gsm8k_style_prompt(self):
        """GSM8K-style math prompt should be structured."""
        prompt = (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with four. "
            "She sells the remainder at the farmers' market daily for $2 per "
            "fresh duck egg. How much in dollars does she make every day at the "
            "farmers' market?"
        )
        result = classify_task(prompt, prompt_tokens=80)
        assert result.task_type == "structured"

    def test_hotpotqa_style_prompt(self):
        """HotpotQA-style multi-paragraph QA should be comprehension."""
        prompt = (
            "Walter Elias Disney was an American entrepreneur, animator, voice "
            "actor and film producer. A pioneer of the American animation "
            "industry, he introduced several developments in the production of "
            "cartoons.\n\n"
            "As a film producer, Disney holds the record for most Academy Awards "
            "earned by an individual, having won 22 Oscars from 59 nominations.\n\n"
            "He was presented with two Golden Globe Special Achievement Awards "
            "and an Emmy Award, among other honors.\n\n"
            "Several of his films are included in the National Film Registry by "
            "the Library of Congress.\n\n"
            "Based on the passage, how many Academy Awards did Disney win?"
        )
        result = classify_task(prompt, prompt_tokens=1200)
        assert result.task_type == "comprehension"

    def test_empty_prompt(self):
        result = classify_task("", prompt_tokens=0)
        assert isinstance(result, TaskClassification)

    def test_zero_tokens_no_token_bonus(self):
        result = classify_task("test", prompt_tokens=0)
        assert result.features["token_count"] == 0


class TestAssessTransferWithPromptText:
    """Tests for enhanced assess_transfer() with prompt text."""

    def test_backward_compatible_no_prompt_text(self):
        """Without prompt_text, behaves exactly like v1."""
        result = assess_transfer(prompt_tokens=200)
        assert result.recommend_latent is True
        assert result.task_classification is None

    def test_backward_compatible_over_limit(self):
        result = assess_transfer(prompt_tokens=400)
        assert result.recommend_latent is False
        assert result.task_classification is None

    def test_structured_prompt_recommended_latent(self):
        result = assess_transfer(
            prompt_tokens=100,
            prompt_text="Solve: 24 * 17 + 3 = ?",
        )
        assert result.recommend_latent is True
        assert result.task_classification is not None
        assert result.task_classification.task_type == "structured"

    def test_comprehension_prompt_blocks_latent(self):
        """Strong comprehension signal blocks latent even under token limit."""
        prompt = (
            "The Mona Lisa is a half-length portrait painting by Italian "
            "artist Leonardo da Vinci.\n\n"
            "It has been described as the best known painting in the world.\n\n"
            "The painting is thought to be a portrait of Lisa Gherardini.\n\n"
            "According to the passage, who painted the Mona Lisa?"
        )
        result = assess_transfer(prompt_tokens=200, prompt_text=prompt)
        assert result.recommend_latent is False
        assert result.task_classification is not None
        assert result.task_classification.task_type == "comprehension"

    def test_structured_extends_token_limit(self):
        """Strong structured signal allows prompts up to 1.5x limit."""
        prompt = (
            "```python\n"
            "def calculate(x, y):\n"
            "    result = x * y + 100 / 50 - 25\n"
            "    return result\n"
            "```\n"
            "Solve: compute calculate(10, 20) step by step. "
            "Find the value of 10 * 20 + 100 / 50 - 25"
        )
        # 350 tokens > 300 limit, but strong structured signals
        result = assess_transfer(prompt_tokens=350, prompt_text=prompt)
        assert result.recommend_latent is True

    def test_structured_still_blocked_beyond_extended_limit(self):
        """Even structured, blocked if way over limit (>450 for default 300)."""
        prompt = "Solve: 2 + 2 = ?"
        result = assess_transfer(prompt_tokens=500, prompt_text=prompt)
        assert result.recommend_latent is False

    def test_task_classification_disabled(self):
        config = TransferQualityConfig(use_task_classification=False)
        result = assess_transfer(
            prompt_tokens=200,
            prompt_text="Who painted the Mona Lisa?",
            config=config,
        )
        assert result.task_classification is None
        assert result.recommend_latent is True  # under token limit

    def test_result_includes_classification(self):
        result = assess_transfer(
            prompt_tokens=100,
            prompt_text="Solve: 2 + 2",
        )
        assert isinstance(result, TransferQualityResult)
        assert result.task_classification is not None
        assert isinstance(result.task_classification, TaskClassification)

    def test_effective_rank_still_works_with_prompt_text(self):
        """Effective rank check still works alongside task classification."""
        import torch

        config = TransferQualityConfig(check_effective_rank=True)
        # Identity-like matrix = low effective rank
        hidden = torch.eye(10, 64)
        result = assess_transfer(
            prompt_tokens=100,
            prompt_text="Solve: 2 + 2",
            hidden_states=hidden,
            config=config,
        )
        assert isinstance(result, TransferQualityResult)


class TestAssessTransferBackwardCompat:
    """Ensure all existing behavior is preserved."""

    def test_short_prompt_recommends_latent(self):
        result = assess_transfer(prompt_tokens=200)
        assert result.recommend_latent is True

    def test_long_prompt_recommends_text(self):
        result = assess_transfer(prompt_tokens=1500)
        assert result.recommend_latent is False

    def test_boundary_at_300(self):
        assert assess_transfer(prompt_tokens=300).recommend_latent is True
        assert assess_transfer(prompt_tokens=301).recommend_latent is False

    def test_custom_config_threshold(self):
        config = TransferQualityConfig(max_prompt_tokens=100)
        assert assess_transfer(prompt_tokens=100, config=config).recommend_latent is True
        assert assess_transfer(prompt_tokens=101, config=config).recommend_latent is False

    def test_zero_tokens(self):
        result = assess_transfer(prompt_tokens=0)
        assert result.recommend_latent is True
