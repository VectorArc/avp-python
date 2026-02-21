"""Calibrate a Rosetta Stone projection matrix between two models.

Runs anchor texts through both models, collects last-layer hidden states,
and fits a linear map via ridge regression or orthogonal Procrustes.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from ..errors import RealignmentError
from ..types import ProjectionMethod
from .._torch_compat import require_torch as _require_torch


@dataclass
class AVPMap:
    """A learned or vocabulary-mediated projection map between two models.

    The ``w_map`` field holds different data depending on the method:
      - RIDGE / PROCRUSTES: projection matrix of shape [D_src, D_tgt].
      - VOCAB_MEDIATED: target model's input embedding weights [vocab_size, D_tgt].
    """

    source_model_id: str
    source_hash: str
    source_dim: int
    target_model_id: str
    target_hash: str
    target_dim: int
    w_map: Any           # torch.Tensor — see class docstring for shape
    bias: Optional[Any]  # torch.Tensor [D_tgt] or None
    target_norm: Any     # torch.Tensor scalar
    method: Union[ProjectionMethod, str]  # ProjectionMethod enum (str accepted for compat)
    anchor_count: int
    validation_score: float

    def __post_init__(self) -> None:
        if isinstance(self.method, str):
            self.method = ProjectionMethod(self.method)


# Diverse anchor sentences for calibration.
# Covers: English prose, instructions, math, code, multilingual, short/long.
DEFAULT_ANCHORS: List[str] = [
    # English prose
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "The only thing we have to fear is fear itself.",
    "To be or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
    "Call me Ishmael.",
    "The sun rose over the mountains, casting long shadows across the valley below.",
    "She walked through the garden, noting how the roses had bloomed overnight.",
    # Instructions / technical
    "Please summarize the following document in three bullet points.",
    "Explain the difference between a stack and a queue in computer science.",
    "Write a Python function that computes the Fibonacci sequence.",
    "List the steps needed to deploy a web application to production.",
    "Describe how gradient descent works in machine learning.",
    # Math and reasoning
    "If x plus three equals seven, then x equals four.",
    "The area of a circle is pi times the radius squared.",
    "Calculate the derivative of f(x) = 3x^2 + 2x - 5.",
    "A train travels 120 kilometers in 2 hours. What is its average speed?",
    "Solve the equation: 2x + 5 = 15.",
    # Code-like
    "def hello_world(): print('Hello, world!')",
    "for i in range(10): result += data[i] * weights[i]",
    "SELECT name, age FROM users WHERE age > 18 ORDER BY name;",
    "import numpy as np; x = np.array([1, 2, 3, 4, 5])",
    "class Node: def __init__(self, value): self.value = value",
    # Multilingual
    "Bonjour, comment allez-vous aujourd'hui?",
    "Die Katze sitzt auf der Matte.",
    "El sol brilla sobre el mar azul.",
    "La vita e bella quando si e felici.",
    "Konnichiwa, kyou wa ii tenki desu ne.",
    # Short and long
    "Yes.",
    "No, I don't think that's correct.",
    "The implementation of the attention mechanism in transformer models allows each position in the sequence to attend to all other positions.",
    "Machine learning models require careful hyperparameter tuning, regularization, and cross-validation to achieve optimal generalization performance on unseen data.",
    "One.",
    "Two plus two equals four.",
    "In the beginning, there was nothing. Then, there was everything.",
    "The cat sat on the mat and looked out the window at the birds flying by.",
    "According to recent studies, the global temperature has risen by approximately 1.1 degrees Celsius since pre-industrial times.",
    "What is the meaning of life?",
    # More diverse topics
    "The stock market experienced significant volatility during the trading session.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Renaissance was a period of cultural rebirth in European history.",
    "Quantum computers use qubits instead of classical bits for computation.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Democracy is a form of government in which power is vested in the people.",
    "The human genome contains approximately three billion base pairs of DNA.",
    "Climate change affects ecosystems, weather patterns, and sea levels globally.",
    "Artificial intelligence is transforming industries from healthcare to finance.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
]


def _extract_hidden_states(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: str,
) -> Any:
    """Run texts through a model and collect last-layer hidden states.

    Returns tensor of shape [N, hidden_dim] — one vector per text.
    """
    torch = _require_torch()

    vectors = []
    for text in texts:
        encoded = tokenizer(
            text, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=128,
        )
        input_ids = encoded["input_ids"].to(device)
        if "attention_mask" in encoded:
            attention_mask = encoded["attention_mask"].to(device)
        else:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last hidden state of last token: [hidden_dim]
        last_hidden = outputs.hidden_states[-1][0, -1, :]
        vectors.append(last_hidden)

    return torch.stack(vectors).to(torch.float32)  # [N, D]


def _ridge_regression(
    X: Any, Y: Any, lambda_reg: float
) -> Any:
    """Fit W via ridge regression: W = (X^T X + lambda I)^{-1} X^T Y.

    Args:
        X: Source hidden states [N, D_src].
        Y: Target hidden states [N, D_tgt].
        lambda_reg: Regularization coefficient.

    Returns:
        W: Projection matrix [D_src, D_tgt].
    """
    torch = _require_torch()

    gram = torch.matmul(X.T, X)  # [D_src, D_src]
    reg = lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    gram = gram + reg

    rhs = torch.matmul(X.T, Y)  # [D_src, D_tgt]
    W = torch.linalg.solve(gram, rhs)  # [D_src, D_tgt]
    return W


def _orthogonal_procrustes(X: Any, Y: Any) -> Any:
    """Fit W via orthogonal Procrustes: W = V @ U^T from SVD of Y^T X.

    Finds the orthogonal matrix that best maps X to Y.

    Args:
        X: Source hidden states [N, D] (must have same D as Y).
        Y: Target hidden states [N, D].

    Returns:
        W: Orthogonal projection matrix [D, D].
    """
    torch = _require_torch()

    M = torch.matmul(Y.T, X)  # [D, D]
    U, _, Vt = torch.linalg.svd(M)
    W = torch.matmul(Vt.T, U.T)  # [D, D]
    return W


def _have_shared_vocab(source_tokenizer: Any, target_tokenizer: Any) -> bool:
    """Check if two tokenizers share the same vocabulary."""
    if not (hasattr(source_tokenizer, "get_vocab") and hasattr(target_tokenizer, "get_vocab")):
        return False
    src_vocab = source_tokenizer.get_vocab()
    tgt_vocab = target_tokenizer.get_vocab()
    return src_vocab == tgt_vocab


def calibrate(
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    anchor_texts: Optional[List[str]] = None,
    method: str = "auto",
    lambda_reg: float = 1e-4,
    validation_split: float = 0.2,
    device: Optional[str] = None,
) -> AVPMap:
    """Calibrate a projection matrix between two models.

    When both tokenizers share the same vocabulary, returns a vocab-mediated
    AVPMap instantly (no calibration needed). Otherwise falls back to
    ridge regression or orthogonal Procrustes.

    Args:
        source_model: Source HuggingFace model.
        target_model: Target HuggingFace model.
        source_tokenizer: Source model's tokenizer.
        target_tokenizer: Target model's tokenizer.
        anchor_texts: Calibration sentences. Defaults to DEFAULT_ANCHORS.
        method: "ridge", "procrustes", "vocab_mediated", or "auto".
        lambda_reg: Ridge regularization coefficient.
        validation_split: Fraction of anchors held out for validation.
        device: Device for computation. Defaults to source model's device.

    Returns:
        AVPMap with the learned or vocabulary-mediated projection.
    """
    torch = _require_torch()
    from ..handshake import compute_model_hash, extract_model_identity
    from ..realign import compute_target_norm

    if device is None:
        device = str(next(source_model.parameters()).device)

    # Extract identities
    src_identity = extract_model_identity(source_model)
    tgt_identity = extract_model_identity(target_model)
    src_hash = compute_model_hash(source_model.config.to_dict())
    tgt_hash = compute_model_hash(target_model.config.to_dict())

    # Resolve method string to enum (accept both strings and enums)
    if isinstance(method, ProjectionMethod):
        resolved = method
    elif method == "auto":
        resolved = None  # decide below
    else:
        resolved = ProjectionMethod(method)

    # Check for shared vocabulary → instant vocab-mediated projection
    shared_vocab = _have_shared_vocab(source_tokenizer, target_tokenizer)
    if resolved is None and shared_vocab:
        resolved = ProjectionMethod.VOCAB_MEDIATED
    if resolved == ProjectionMethod.VOCAB_MEDIATED:
        if not shared_vocab:
            raise ValueError(
                "vocab_mediated method requires tokenizers with the same vocabulary."
            )

        # Store target model's input embedding weights as w_map
        tgt_input_embeds = target_model.get_input_embeddings()
        if tgt_input_embeds is None or not hasattr(tgt_input_embeds, "weight"):
            raise RealignmentError(
                "Cannot get target input embeddings for vocab-mediated projection."
            )
        w_map = tgt_input_embeds.weight.detach().to(torch.float32).cpu()
        target_norm = compute_target_norm(target_model, device=device)

        src_dim = source_model.config.hidden_size
        tgt_dim = target_model.config.hidden_size
        # For models with non-standard config keys (e.g. GPT2 n_embd)
        if src_dim is None or src_dim == 0:
            src_dim = getattr(source_model.config, "n_embd", 0)
        if tgt_dim is None or tgt_dim == 0:
            tgt_dim = getattr(target_model.config, "n_embd", 0)

        return AVPMap(
            source_model_id=src_identity.model_id,
            source_hash=src_hash,
            source_dim=src_dim,
            target_model_id=tgt_identity.model_id,
            target_hash=tgt_hash,
            target_dim=tgt_dim,
            w_map=w_map,
            bias=None,
            target_norm=target_norm.cpu(),
            method=ProjectionMethod.VOCAB_MEDIATED,
            anchor_count=0,
            validation_score=1.0,  # no calibration needed
        )

    # --- Learned projection (ridge / procrustes) ---
    if anchor_texts is None:
        anchor_texts = DEFAULT_ANCHORS

    if len(anchor_texts) < 5:
        raise ValueError("Need at least 5 anchor texts for calibration.")

    # Split train/validation
    n_val = max(1, int(len(anchor_texts) * validation_split))
    n_train = len(anchor_texts) - n_val
    train_texts = anchor_texts[:n_train]
    val_texts = anchor_texts[n_train:]

    # Extract hidden states — get actual dims from tensors (more reliable
    # than config, since some models use non-standard config keys like n_embd)
    X_train = _extract_hidden_states(source_model, source_tokenizer, train_texts, device)

    # Ridge target: use target input embeddings projected from anchor tokens,
    # not hidden states. This maps to what we actually inject via inputs_embeds.
    tgt_input_embeds = target_model.get_input_embeddings()
    if tgt_input_embeds is not None and hasattr(tgt_input_embeds, "weight"):
        tgt_embed_weight = tgt_input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        Y_train_vecs = []
        for text in train_texts:
            encoded = target_tokenizer(
                text, return_tensors="pt", add_special_tokens=True,
                truncation=True, max_length=128,
            )
            ids = encoded["input_ids"].to(device)
            # Mean-pool the input embeddings for this text
            embeds = tgt_embed_weight[ids[0]]  # [seq_len, D_tgt]
            Y_train_vecs.append(embeds.mean(dim=0))
        Y_train = torch.stack(Y_train_vecs).to(torch.float32)
    else:
        # Fallback to hidden states if embeddings unavailable
        Y_train = _extract_hidden_states(target_model, target_tokenizer, train_texts, device)

    d_src = X_train.shape[-1]
    d_tgt = Y_train.shape[-1]

    # Choose method
    if resolved is None:
        chosen = ProjectionMethod.PROCRUSTES if d_src == d_tgt else ProjectionMethod.RIDGE
    else:
        if resolved == ProjectionMethod.PROCRUSTES and d_src != d_tgt:
            raise ValueError(
                f"Procrustes requires same dimensions, got {d_src} vs {d_tgt}. "
                f"Use method='ridge' or method='auto'."
            )
        chosen = resolved

    # Fit projection
    if chosen == ProjectionMethod.PROCRUSTES:
        w_map = _orthogonal_procrustes(X_train, Y_train)
    else:
        w_map = _ridge_regression(X_train, Y_train, lambda_reg)

    # Compute target norm
    target_norm = compute_target_norm(target_model, device=device)

    # Validation: cosine similarity on held-out anchors
    if len(val_texts) > 0:
        X_val = _extract_hidden_states(source_model, source_tokenizer, val_texts, device)

        # Use same Y target type as training (input embeddings or hidden states)
        if tgt_input_embeds is not None and hasattr(tgt_input_embeds, "weight"):
            Y_val_vecs = []
            for text in val_texts:
                encoded = target_tokenizer(
                    text, return_tensors="pt", add_special_tokens=True,
                    truncation=True, max_length=128,
                )
                ids = encoded["input_ids"].to(device)
                embeds = tgt_embed_weight[ids[0]]
                Y_val_vecs.append(embeds.mean(dim=0))
            Y_val = torch.stack(Y_val_vecs).to(torch.float32)
        else:
            Y_val = _extract_hidden_states(target_model, target_tokenizer, val_texts, device)

        projected = torch.matmul(X_val, w_map)  # [N_val, D_tgt]
        # Normalize both for cosine similarity
        proj_norm = projected / projected.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        y_norm = Y_val / Y_val.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_sims = (proj_norm * y_norm).sum(dim=-1)  # [N_val]
        validation_score = cos_sims.mean().item()
    else:
        validation_score = 0.0

    return AVPMap(
        source_model_id=src_identity.model_id,
        source_hash=src_hash,
        source_dim=d_src,
        target_model_id=tgt_identity.model_id,
        target_hash=tgt_hash,
        target_dim=d_tgt,
        w_map=w_map.cpu(),
        bias=None,
        target_norm=target_norm.cpu(),
        method=chosen,
        anchor_count=n_train,
        validation_score=validation_score,
    )
