"""Trained cross-model translator (C2C) for AVP.

Trains per-layer linear projections with learned sigmoid gates to map
source model hidden states to target model activation space at every
transformer layer. Both source and target models are frozen; only the
lightweight projector trains.

Based on:
- C2C (arxiv 2510.03215): Cross-cache fusion, +6-14% over zero-shot
- DroidSpeak: ~11% of layers are critical, gates learn which matter
- Model Stitching (2506.06609): Affine maps between layers, 2K-180K samples

Usage::

    from avp.rosetta.train import LayerProjector, train_projector

    projector = train_projector(
        source_model=src_model,
        target_model=tgt_model,
        source_tokenizer=src_tok,
        target_tokenizer=tgt_tok,
        device="cuda",
    )
    # Produces an AVPMap with method=TRAINED
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("torch is required for training. pip install torch")


@dataclass
class TrainConfig:
    """Configuration for training a cross-model projector.

    Attributes:
        num_samples: Number of text samples to train on.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        num_epochs: Number of training epochs.
        gate_reg_weight: L1 regularization weight for gate sparsity.
        gate_init: Initial gate logit (sigmoid(gate_init) = initial gate value).
        max_seq_len: Maximum sequence length for training samples.
        extraction_layer_ratio: Depth ratio for source hidden state extraction.
        warmup_steps: Number of linear warmup steps.
        seed: Random seed for reproducibility.
    """

    num_samples: int = 5000
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 2
    gate_reg_weight: float = 0.01
    gate_init: float = -5.0  # sigmoid(-5) ~ 0.007 — near zero to avoid corrupting target
    max_seq_len: int = 256
    extraction_layer_ratio: float = 0.75
    warmup_steps: int = 100
    seed: int = 42
    mse_aux_weight: float = 0.1  # Weight for MSE auxiliary loss (0 = NTP only)
    use_ntp_loss: bool = True  # Use next-token prediction as primary loss


class LayerProjector:
    """Per-layer linear projections with learned sigmoid gates.

    Maps a source hidden state to target activation space at each layer.
    Gate values control injection strength per layer (initialized near zero).
    """

    def __init__(self, source_dim: int, target_dim: int, num_layers: int, config: Optional[TrainConfig] = None):
        torch = _require_torch()
        import torch.nn as nn

        if config is None:
            config = TrainConfig()

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.num_layers = num_layers

        # Per-layer projection: source hidden -> target hidden at layer L
        self.layer_projections = nn.ModuleList([
            nn.Linear(source_dim, target_dim, bias=True)
            for _ in range(num_layers)
        ])

        # Per-layer gate: sigmoid scalar controlling injection strength
        self.layer_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(config.gate_init))
            for _ in range(num_layers)
        ])

        # Initialize projections with small weights
        for proj in self.layer_projections:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            nn.init.zeros_(proj.bias)

    def parameters(self):
        """Return all trainable parameters."""
        for proj in self.layer_projections:
            yield from proj.parameters()
        for gate in self.layer_gates:
            yield gate

    def to(self, device):
        """Move all parameters to device."""
        for proj in self.layer_projections:
            proj.to(device)
        for i, gate in enumerate(self.layer_gates):
            self.layer_gates[i] = gate.to(device)
        return self

    def train(self):
        """Set to training mode."""
        for proj in self.layer_projections:
            proj.train()

    def eval(self):
        """Set to evaluation mode."""
        for proj in self.layer_projections:
            proj.eval()

    def forward(self, source_hidden, return_gate_tensors: bool = False):
        """Project source hidden state to each target layer.

        Args:
            source_hidden: [B, D_src] from source model.
            return_gate_tensors: If True, return gate as tensor (for training
                gradient flow). If False, return gate as float (for inference).

        Returns:
            List of (projected_hidden [B, D_tgt], gate_value) per layer.
            gate_value is a float (inference) or tensor (training).
        """
        torch = _require_torch()
        results = []
        for proj, gate_logit in zip(self.layer_projections, self.layer_gates):
            projected = proj(source_hidden)           # [B, D_tgt]
            gate_tensor = torch.sigmoid(gate_logit)   # scalar tensor in [0, 1]
            if return_gate_tensors:
                results.append((projected, gate_tensor))
            else:
                results.append((projected, gate_tensor.item()))
        return results

    def get_active_layers(self, threshold: float = 0.01) -> List[int]:
        """Return indices of layers with gate > threshold."""
        torch = _require_torch()
        active = []
        for i, gate_logit in enumerate(self.layer_gates):
            gate = torch.sigmoid(gate_logit).item()
            if gate > threshold:
                active.append(i)
        return active

    def export_weights(self) -> Tuple[List[Any], List[Any], List[float]]:
        """Export weights for serialization into AVPMap.

        Returns:
            Tuple of (layer_weights, layer_biases, layer_gates)
            where each is a list of length num_layers.
        """
        torch = _require_torch()
        weights = []
        biases = []
        gates = []
        for proj, gate_logit in zip(self.layer_projections, self.layer_gates):
            weights.append(proj.weight.detach().cpu())      # [D_tgt, D_src]
            biases.append(proj.bias.detach().cpu())          # [D_tgt]
            gates.append(torch.sigmoid(gate_logit).item())   # float
        return weights, biases, gates


def _load_training_data(
    tokenizer: Any,
    num_samples: int,
    max_seq_len: int,
    seed: int = 42,
) -> List[str]:
    """Load training text samples from a dataset.

    Tries to use OpenHermes-2.5, falls back to a simple math/code/text mix.
    """
    try:
        from datasets import load_dataset
        logger.info("Loading training data from teknium/OpenHermes-2.5...")
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        texts = []
        for item in ds:
            if len(texts) >= num_samples:
                break
            # Extract conversation text
            convos = item.get("conversations", [])
            text = " ".join(c.get("value", "") for c in convos)
            if len(text) > 50:  # Skip very short entries
                texts.append(text[:max_seq_len * 4])  # Rough char limit
        logger.info(f"Loaded {len(texts)} training samples")
        return texts
    except Exception as e:
        logger.warning(f"Could not load OpenHermes: {e}. Using fallback data.")
        # Fallback: generate simple diverse prompts
        import random
        rng = random.Random(seed)
        templates = [
            "Solve the following math problem step by step: {a} + {b} * {c} = ?",
            "Write a function that computes the {n}th Fibonacci number.",
            "Explain the concept of {topic} in simple terms.",
            "The quick brown fox jumps over the lazy dog. {extra}",
            "In {year}, {person} made a significant discovery about {topic}.",
        ]
        topics = ["recursion", "neural networks", "gravity", "evolution", "democracy",
                  "photosynthesis", "quantum mechanics", "machine learning", "databases"]
        texts = []
        for i in range(num_samples):
            tmpl = rng.choice(templates)
            text = tmpl.format(
                a=rng.randint(1, 1000), b=rng.randint(1, 100), c=rng.randint(1, 50),
                n=rng.randint(5, 50),
                topic=rng.choice(topics),
                extra=" ".join(rng.choices(topics, k=3)),
                year=rng.randint(1900, 2025),
                person=f"Dr. {rng.choice(['Smith', 'Chen', 'Patel', 'Garcia', 'Kim'])}",
            )
            texts.append(text)
        return texts


def _tokenize_batch(
    texts: List[str],
    tokenizer: Any,
    max_seq_len: int,
    device: str,
) -> Any:
    """Tokenize a batch of texts."""
    torch = _require_torch()
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def train_projector(
    source_model: Any,
    target_model: Any,
    source_tokenizer: Any,
    target_tokenizer: Any,
    device: str = "cuda",
    config: Optional[TrainConfig] = None,
    progress_callback: Optional[Any] = None,
) -> "AVPMap":
    """Train a per-layer cross-model projector.

    Both models are frozen. Only the lightweight projector trains.
    Uses MSE loss between projected source hidden states and target
    reference hidden states at each layer.

    Args:
        source_model: Source HuggingFace model (frozen).
        target_model: Target HuggingFace model (frozen).
        source_tokenizer: Source tokenizer.
        target_tokenizer: Target tokenizer (used for training data).
        device: Training device.
        config: Training configuration.
        progress_callback: Optional callback(step, total_steps, loss) for progress.

    Returns:
        AVPMap with method=TRAINED and per-layer projection data.
    """
    torch = _require_torch()
    import torch.nn.functional as F
    from ..handshake import compute_model_hash, extract_model_identity

    if config is None:
        config = TrainConfig()

    # Set seed
    torch.manual_seed(config.seed)

    # Get model info
    src_identity = extract_model_identity(source_model, source_tokenizer)
    tgt_identity = extract_model_identity(target_model, target_tokenizer)
    src_hash = compute_model_hash(source_model.config.to_dict())
    tgt_hash = compute_model_hash(target_model.config.to_dict())

    source_dim = src_identity.hidden_dim
    target_dim = tgt_identity.hidden_dim
    target_num_layers = tgt_identity.num_layers

    logger.info(
        "Training projector: %s (%dd) -> %s (%dd, %d layers)",
        src_identity.model_id, source_dim,
        tgt_identity.model_id, target_dim, target_num_layers,
    )

    # Create projector
    projector = LayerProjector(source_dim, target_dim, target_num_layers, config)
    projector.to(device)
    projector.train()

    # Freeze both models
    source_model.eval()
    target_model.eval()
    for p in source_model.parameters():
        p.requires_grad_(False)
    for p in target_model.parameters():
        p.requires_grad_(False)

    # Load training data
    texts = _load_training_data(
        target_tokenizer, config.num_samples, config.max_seq_len, config.seed
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Training loop
    num_batches = math.ceil(len(texts) / config.batch_size)
    total_steps = num_batches * config.num_epochs
    step = 0
    best_loss = float("inf")

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_start in range(0, len(texts), config.batch_size):
            batch_texts = texts[batch_start:batch_start + config.batch_size]

            # Tokenize with target tokenizer (both models process same text)
            try:
                tgt_encoded = _tokenize_batch(batch_texts, target_tokenizer, config.max_seq_len, device)
                src_encoded = _tokenize_batch(batch_texts, source_tokenizer, config.max_seq_len, device)
            except Exception:
                continue

            # Source forward pass -> extract hidden state
            with torch.no_grad():
                src_out = source_model(
                    **src_encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Extract from extraction layer ratio
                src_hidden_states = src_out.hidden_states
                extract_idx = int(len(src_hidden_states) * config.extraction_layer_ratio)
                extract_idx = min(extract_idx, len(src_hidden_states) - 1)
                src_hidden = src_hidden_states[extract_idx][:, -1, :]  # [B, D_src]

            # Project source hidden state through all layers
            projections = projector.forward(src_hidden.float(), return_gate_tensors=True)

            # Build per-layer projections for hooks
            layer_proj_for_hooks = []
            for proj_h, gate in projections:
                gate_val = gate.item()
                if gate_val < 0.001:
                    layer_proj_for_hooks.append(None)
                else:
                    layer_proj_for_hooks.append((proj_h, gate))

            loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Reference forward pass (unhooked, no grad) for MSE auxiliary
            ref_hidden_states = None
            if config.mse_aux_weight > 0:
                with torch.no_grad():
                    ref_out = target_model(
                        **tgt_encoded,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                ref_hidden_states = ref_out.hidden_states

            # Primary loss: NTP through target model with injected projections
            if config.use_ntp_loss:
                from .trained_hooks import trained_multi_layer_hook
                tgt_input_ids = tgt_encoded["input_ids"]
                labels = tgt_input_ids[:, 1:].contiguous()  # shift right

                with trained_multi_layer_hook(target_model, layer_proj_for_hooks):
                    tgt_out = target_model(
                        **tgt_encoded,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                logits = tgt_out.logits[:, :-1, :].contiguous()  # [B, seq-1, vocab]
                ntp_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=target_tokenizer.pad_token_id or -100,
                )
                loss = loss + ntp_loss
            elif config.mse_aux_weight > 0:
                pass  # ref_hidden_states already computed above
            else:
                # Neither NTP nor MSE — nothing to train on
                with torch.no_grad():
                    ref_out = target_model(
                        **tgt_encoded,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                ref_hidden_states = ref_out.hidden_states

            # Auxiliary loss: MSE between projected and unhooked reference hidden states
            if config.mse_aux_weight > 0 and ref_hidden_states is not None:
                mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for layer_idx, (proj_h, gate) in enumerate(projections):
                    gate_val = gate.item()
                    if gate_val < 0.001:
                        continue
                    ref_idx = min(layer_idx + 1, len(ref_hidden_states) - 1)
                    ref_h = ref_hidden_states[ref_idx][:, -1, :].float().detach()
                    layer_loss = F.mse_loss(proj_h, ref_h)
                    mse_loss = mse_loss + gate * layer_loss
                loss = loss + config.mse_aux_weight * mse_loss

            # Gate sparsity regularization (encourage most gates to be ~0)
            gate_sum = sum(
                torch.sigmoid(g) for g in projector.layer_gates
            ) / target_num_layers
            loss = loss + config.gate_reg_weight * gate_sum

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)

            optimizer.step()

            step += 1
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1

            if step % 50 == 0:
                active = projector.get_active_layers()
                logger.info(
                    "Step %d/%d, loss=%.4f, active_layers=%d/%d",
                    step, total_steps, loss_val, len(active), target_num_layers,
                )

            if progress_callback is not None:
                progress_callback(step, total_steps, loss_val)

            # Free intermediate tensors
            del src_out, projections, loss
            ref_hidden_states = None
            if device == "cuda":
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d/%d complete, avg_loss=%.4f", epoch + 1, config.num_epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Export to AVPMap
    projector.eval()
    layer_weights, layer_biases, layer_gates = projector.export_weights()

    active = projector.get_active_layers()
    logger.info(
        "Training complete. Active layers: %d/%d (gates > 0.01): %s",
        len(active), target_num_layers, active,
    )

    from .calibrate import AVPMap
    from ..types import ProjectionMethod

    # Compute target norm from target model embeddings
    tgt_embed = target_model.get_input_embeddings()
    target_norm = tgt_embed.weight.float().norm(dim=-1).mean().detach().cpu()

    avp_map = AVPMap(
        source_model_id=src_identity.model_id,
        source_hash=src_hash,
        source_dim=source_dim,
        target_model_id=tgt_identity.model_id,
        target_hash=tgt_hash,
        target_dim=target_dim,
        w_map=torch.zeros(1),  # Placeholder (projections stored in layer_weights)
        bias=None,
        target_norm=target_norm,
        method=ProjectionMethod.TRAINED,
        anchor_count=config.num_samples,
        validation_score=1.0 - best_loss,  # Higher is better
        layer_weights=layer_weights,
        layer_biases=layer_biases,
        layer_gates=layer_gates,
    )

    return avp_map
