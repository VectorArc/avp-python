"""Per-transfer quality gate for cross-model rosetta projection.

Assesses whether a given transfer is likely to produce useful results
via latent projection, or whether JSON fallback is recommended.

Primary signal: source prompt token count. Single-embedding rosetta works
for short structured prompts (<300 tokens) but degrades significantly for
longer prompts. Validated across 4 benchmarks x 3 rosetta configurations:
  - GSM8K cross-family: 65% at <300 tokens, 41% at 300-500 tokens
  - HumanEval same-family: 61% at <300, 40% at 300-500, 19% at 500+
  - Even reverse rosetta (strong config): 87%, 84%, 55% at 500+
Prompt length is a zero-cost proxy for information density.

Enhanced signal (v2): task-type classification via prompt features. Detects
math/code markers (structured tasks where rosetta works) vs comprehension
patterns (multi-paragraph context + question where text is needed). Zero
latency overhead, zero dependencies. See classify_task().

Secondary signal (opt-in): effective rank ratio of hidden states. High
effective rank means information is spread across many dimensions, which
a single embedding cannot capture. Off by default; included for future
validation.

Usage::

    from avp.rosetta.quality import assess_transfer

    # Token-count only (backward compatible)
    result = assess_transfer(prompt_tokens=len(input_ids[0]))

    # Enhanced: with prompt text for task-type classification
    result = assess_transfer(prompt_tokens=len(input_ids[0]), prompt_text=prompt)

    if result.recommend_latent:
        # proceed with rosetta projection
        ...
    else:
        # fall back to JSON text transfer
        ...
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Compiled patterns for task classification (module-level for zero overhead)
_MATH_MARKERS = re.compile(
    r'\\boxed|\\frac|\\sqrt|\\sum|\\int|\\cdot|'
    r'\d+\s*[+\-*/=]\s*\d+|'
    r'\$[^$]+\$'
)
_CODE_MARKERS = re.compile(
    r'\bdef\s+\w+|'
    r'\bclass\s+\w+|'
    r'\bimport\s+\w+|'
    r'\bfunction\s+\w+|'
    r'\breturn\s+|'
    r'```|'
    r'>>>|'
    r'\bfor\s+\w+\s+in\s+'
)
_COMPREHENSION_STARTERS = re.compile(
    r'(?:^|[\n.?!]\s*)(?:who\s+(?:is|was|were|are|did)|'
    r'what\s+(?:is|was|were|are|did|does)|'
    r'when\s+(?:did|was|were|is)|'
    r'where\s+(?:did|was|were|is)|'
    r'why\s+(?:did|was|were|is)|'
    r'how\s+did|'
    r'according\s+to|based\s+on|which\s+of\s+the\s+following|'
    r'in\s+the\s+(?:passage|text|article|context))',
    re.IGNORECASE,
)
_STRUCTURED_STARTERS = re.compile(
    r'(?:^|[\n.]\s*)(?:solve|compute|calculate|find\s+the|evaluate|simplify|'
    r'how\s+much|how\s+many|'
    r'implement|write\s+(?:a\s+)?(?:function|program|code|class)|'
    r'debug|fix\s+(?:the|this)|refactor)',
    re.IGNORECASE,
)


@dataclass
class TransferQualityConfig:
    """Configuration for the per-transfer quality gate.

    Attributes:
        max_prompt_tokens: Maximum prompt token count for which latent
            transfer is recommended. Default 300, validated across GSM8K,
            HumanEval, and HotpotQA rosetta benchmarks. Above 300 tokens,
            accuracy drops 20-30pp across all configurations.
        use_task_classification: Whether to use prompt text features
            for task-type classification. Requires prompt_text to be
            passed to assess_transfer(). On by default.
        check_effective_rank: Whether to compute effective rank ratio
            of hidden states as a secondary signal. Requires torch and
            a hidden_states tensor. Off by default.
        max_effective_rank_ratio: Threshold for effective rank ratio.
            Above this, JSON fallback is recommended. Only used when
            check_effective_rank=True.
    """

    max_prompt_tokens: int = 300
    use_task_classification: bool = True
    check_effective_rank: bool = False
    max_effective_rank_ratio: float = 0.8


@dataclass
class TaskClassification:
    """Result of prompt task-type classification.

    Attributes:
        task_type: 'structured' or 'comprehension'.
        score: Numeric score (positive=structured, negative=comprehension).
        features: Dict of individual feature contributions.
    """

    task_type: str
    score: int
    features: dict = field(default_factory=dict)


@dataclass
class TransferQualityResult:
    """Result of a per-transfer quality assessment.

    Attributes:
        recommend_latent: True if latent transfer is recommended,
            False if JSON fallback is recommended.
        prompt_tokens: The prompt token count that was assessed.
        reason: Human-readable explanation of the recommendation.
        effective_rank_ratio: Effective rank ratio of hidden states,
            or None if not computed.
        task_classification: Task-type classification result, or None
            if prompt_text was not provided.
    """

    recommend_latent: bool
    prompt_tokens: int
    reason: str
    effective_rank_ratio: Optional[float] = None
    task_classification: Optional[TaskClassification] = None


def classify_task(prompt_text: str, prompt_tokens: int = 0) -> TaskClassification:
    """Classify a prompt as 'structured' or 'comprehension'.

    Uses lexical and structural features of the prompt text to determine
    whether rosetta projection is likely to work (structured tasks like
    math/code) or whether text mode is needed (comprehension tasks with
    long contexts).

    Scoring: positive = structured, negative = comprehension.
    Threshold: score > 0 = structured.

    Features (5 signals, validated against AVP benchmark data):
      1. Token count: >500 penalizes, <200 rewards
      2. Digit density: high digit ratio = math/structured
      3. Math/code markers: regex patterns for code/math syntax
      4. Comprehension question patterns: who/what/when/according-to
      5. Multi-paragraph context: 3+ paragraphs = comprehension

    Args:
        prompt_text: The raw prompt text.
        prompt_tokens: Token count (if known, used for length signal).

    Returns:
        TaskClassification with task_type, score, and feature breakdown.
    """
    score = 0
    features = {}

    # Feature 1: Token count (existing signal, enhanced)
    if prompt_tokens > 500:
        features["token_count"] = -2
        score -= 2
    elif prompt_tokens > 300:
        features["token_count"] = -1
        score -= 1
    elif 0 < prompt_tokens < 200:
        features["token_count"] = 1
        score += 1
    else:
        features["token_count"] = 0

    # Feature 2: Digit density
    digits = sum(c.isdigit() for c in prompt_text)
    text_len = max(len(prompt_text), 1)
    digit_ratio = digits / text_len
    if digit_ratio > 0.05:
        features["digit_density"] = 2
        score += 2
    elif digit_ratio > 0.02:
        features["digit_density"] = 1
        score += 1
    else:
        features["digit_density"] = 0

    # Feature 3: Math/code markers
    math_hits = len(_MATH_MARKERS.findall(prompt_text))
    code_hits = len(_CODE_MARKERS.findall(prompt_text))
    marker_score = 0
    if math_hits >= 2:
        marker_score += 2
    elif math_hits >= 1:
        marker_score += 1
    if code_hits >= 2:
        marker_score += 2
    elif code_hits >= 1:
        marker_score += 1
    marker_score = min(marker_score, 3)  # cap at 3
    features["markers"] = marker_score
    score += marker_score

    # Feature 4: Comprehension question patterns
    comp_hits = len(_COMPREHENSION_STARTERS.findall(prompt_text))
    struct_hits = len(_STRUCTURED_STARTERS.findall(prompt_text))
    if comp_hits > 0 and struct_hits == 0:
        features["question_type"] = -2
        score -= 2
    elif struct_hits > 0 and comp_hits == 0:
        features["question_type"] = 1
        score += 1
    else:
        features["question_type"] = 0

    # Feature 5: Multi-paragraph context (3+ double-newlines)
    paragraphs = prompt_text.count("\n\n")
    if paragraphs >= 3:
        features["paragraphs"] = -2
        score -= 2
    elif paragraphs >= 2:
        features["paragraphs"] = -1
        score -= 1
    else:
        features["paragraphs"] = 0

    task_type = "structured" if score > 0 else "comprehension"
    return TaskClassification(task_type=task_type, score=score, features=features)


def _compute_effective_rank_ratio(hidden_states: Any) -> float:
    """Compute effective rank ratio of hidden states via SVD.

    The effective rank (Roy & Vetterli, 2007) measures how spread the
    information is across singular values. A ratio near 1.0 means
    information is uniformly spread (hard to compress into 1 embedding);
    near 0.0 means it's concentrated in few dimensions.

    Args:
        hidden_states: Tensor of shape [seq_len, D] or [1, seq_len, D].

    Returns:
        Ratio in [0, 1] where higher means more spread.
    """
    import torch

    t = hidden_states
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)

    # Squeeze batch dim if present: [1, seq, D] -> [seq, D]
    if t.ndim == 3:
        t = t.squeeze(0)

    if t.ndim != 2 or t.shape[0] < 2:
        return 0.0

    # SVD on [seq_len, D]
    s = torch.linalg.svdvals(t.float())

    # Avoid division by zero
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0

    # Normalized singular values -> probability distribution
    p = s / s_sum

    # Shannon entropy of the distribution
    log_p = torch.log(p + 1e-12)
    entropy = -(p * log_p).sum().item()

    # Maximum entropy = log(min(seq_len, D))
    max_entropy = float(torch.log(torch.tensor(min(t.shape[0], t.shape[1]), dtype=torch.float32)).item())

    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy


def assess_transfer(
    prompt_tokens: int,
    prompt_text: Optional[str] = None,
    hidden_states: Any = None,
    config: Optional[TransferQualityConfig] = None,
) -> TransferQualityResult:
    """Assess whether a cross-model transfer should use latent or JSON.

    This is advisory -- the caller decides how to act on the result.

    When prompt_text is provided and config.use_task_classification is True,
    task-type classification enhances the token-count heuristic. A prompt
    classified as 'comprehension' recommends text even if under the token
    limit; a prompt classified as 'structured' with strong signals may
    recommend latent even if slightly over the token limit.

    Args:
        prompt_tokens: Number of tokens in the source prompt.
        prompt_text: Optional raw prompt text for task-type classification.
        hidden_states: Optional tensor of hidden states, shape
            [seq_len, D] or [1, seq_len, D]. Only used when
            config.check_effective_rank=True.
        config: Quality gate configuration. Uses defaults if None.

    Returns:
        TransferQualityResult with recommendation and reasoning.
    """
    if config is None:
        config = TransferQualityConfig()

    effective_rank_ratio: Optional[float] = None
    task_cls: Optional[TaskClassification] = None

    # Task classification (if prompt text available)
    if prompt_text is not None and config.use_task_classification:
        task_cls = classify_task(prompt_text, prompt_tokens)

    # Combined gate: token count + task classification
    if task_cls is not None:
        # Strong comprehension signal overrides even short prompts
        if task_cls.task_type == "comprehension" and task_cls.score <= -3:
            return TransferQualityResult(
                recommend_latent=False,
                prompt_tokens=prompt_tokens,
                reason=(
                    f"task classified as comprehension (score={task_cls.score}); "
                    f"single embedding unlikely to capture context"
                ),
                task_classification=task_cls,
            )

        # Strong structured signal allows slightly longer prompts
        if task_cls.task_type == "structured" and task_cls.score >= 3:
            # Allow up to 1.5x the normal token limit for strongly structured
            extended_limit = int(config.max_prompt_tokens * 1.5)
            if prompt_tokens > extended_limit:
                return TransferQualityResult(
                    recommend_latent=False,
                    prompt_tokens=prompt_tokens,
                    reason=(
                        f"prompt_tokens={prompt_tokens} exceeds extended limit "
                        f"{extended_limit} (structured task, score={task_cls.score})"
                    ),
                    task_classification=task_cls,
                )
            # Within extended limit -- recommend latent
        else:
            # Moderate signals: use standard token limit
            if prompt_tokens > config.max_prompt_tokens:
                return TransferQualityResult(
                    recommend_latent=False,
                    prompt_tokens=prompt_tokens,
                    reason=(
                        f"prompt_tokens={prompt_tokens} exceeds "
                        f"max_prompt_tokens={config.max_prompt_tokens}; "
                        f"task_type={task_cls.task_type} (score={task_cls.score})"
                    ),
                    task_classification=task_cls,
                )
    else:
        # No task classification -- fall back to token count only
        if prompt_tokens > config.max_prompt_tokens:
            return TransferQualityResult(
                recommend_latent=False,
                prompt_tokens=prompt_tokens,
                reason=(
                    f"prompt_tokens={prompt_tokens} exceeds "
                    f"max_prompt_tokens={config.max_prompt_tokens}; "
                    f"single embedding unlikely to capture sufficient information"
                ),
                effective_rank_ratio=None,
            )

    # Secondary gate: effective rank (opt-in)
    if config.check_effective_rank and hidden_states is not None:
        effective_rank_ratio = _compute_effective_rank_ratio(hidden_states)
        if effective_rank_ratio > config.max_effective_rank_ratio:
            return TransferQualityResult(
                recommend_latent=False,
                prompt_tokens=prompt_tokens,
                reason=(
                    f"effective_rank_ratio={effective_rank_ratio:.3f} exceeds "
                    f"max={config.max_effective_rank_ratio}; "
                    f"information too spread for single-embedding transfer"
                ),
                effective_rank_ratio=effective_rank_ratio,
                task_classification=task_cls,
            )

    return TransferQualityResult(
        recommend_latent=True,
        prompt_tokens=prompt_tokens,
        reason="transfer within quality thresholds",
        effective_rank_ratio=effective_rank_ratio,
        task_classification=task_cls,
    )
