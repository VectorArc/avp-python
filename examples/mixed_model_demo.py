#!/usr/bin/env python3
"""AVP Mixed-Model Demo: automatic handshake negotiation between different models.

Runs a 4-agent reasoning chain where agents 1-2 use Model A and agents 3-4 use
Model B. The AVP handshake protocol automatically negotiates:
  - Same-model pairs → LATENT mode (KV-cache transfer via AVP codec)
  - Different-model pairs → JSON fallback (text)

Usage:
    # Default (Qwen2.5-1.5B + Qwen2.5-0.5B on CUDA)
    python examples/mixed_model_demo.py

    # With verbose output
    python examples/mixed_model_demo.py --verbose

    # Custom models
    python examples/mixed_model_demo.py --model_a Qwen/Qwen2.5-1.5B-Instruct --model_b gpt2

    # Custom question
    python examples/mixed_model_demo.py --question "What is 15% of 240?"
"""

import argparse
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Fix Windows console encoding for model output containing Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure avp-python root is on sys.path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

DEFAULT_QUESTION = (
    "A factory produces widgets in three shifts. The morning shift produces "
    "150 widgets per hour for 8 hours. The afternoon shift is 20% more "
    "productive than the morning shift and runs for 6 hours. The night shift "
    "produces 80% as many widgets per hour as the morning shift and runs for "
    "10 hours. If 5% of all widgets fail quality inspection, how many widgets "
    "pass inspection in a day?"
)

SYSTEM_MESSAGE = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

@dataclass
class MixedAgent:
    """An agent in the mixed-model chain."""
    name: str
    role: str
    model_key: str  # "model_a" or "model_b"


AGENTS = [
    MixedAgent(name="Planner", role="planner", model_key="model_a"),
    MixedAgent(name="Critic", role="critic", model_key="model_a"),
    MixedAgent(name="Refiner", role="refiner", model_key="model_b"),
    MixedAgent(name="Judger", role="judger", model_key="model_b"),
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_latent_prompt(role: str, question: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode (context carried in KV-cache)."""
    if role == "planner":
        user_prompt = (
            f"You are a Planner Agent. Given an input question, design a clear, "
            f"step-by-step plan for how to solve the question.\n\n"
            f"Question: {question}\n\n"
            f"Your outlined plan should be concise with a few bullet points for each step. "
            f"Do not produce the final answer.\n"
            f"Now output your plan to solve the question below:"
        )
    elif role == "critic":
        user_prompt = (
            f"Question: {question}\n\n"
            f"You are a Critic Agent to evaluate the correctness of the input plan "
            f"for the given question and provide helpful feedback for improving the plan.\n"
            f"The plan information is provided in latent KV representation format. "
            f"Review the plan and question and output:\n"
            f"(1) original plan contents\n"
            f"(2) constructive feedback on the original plan.\n\n"
            f"Format your response as follows:\n"
            f"Original Plan: [Copy the provided Planner Agent's plan here]\n"
            f"Feedback: [Your detailed feedback to improve the plan here]\n\n"
            f"Now, output your response below:"
        )
    elif role == "refiner":
        user_prompt = (
            f"Question: {question}\n\n"
            f"You are a Refiner Agent to provide a refined step-by-step plan "
            f"for solving the given question.\n"
            f"You are provided with:\n"
            f"(1) latent-format information: a previous plan with feedback\n"
            f"(2) text-format information: the input question you need to solve.\n\n"
            f"Based on the input, write a refined and improved plan to solve the question. "
            f"Make sure your output plan is correct and concise.\n\n"
            f"Now, output your refined plan below:"
        )
    elif role == "judger":
        user_prompt = (
            f"Target Question: {question}\n\n"
            f"You are a helpful assistant. You are provided with latent information "
            f"for reference and a target question to solve.\n\n"
            f"The latent information might contain irrelevant contents. "
            f"Ignore it if it is not helpful for solving the target question.\n\n"
            f"You must reason step-by-step to solve the provided Target Question "
            f"without outputting other irrelevant information.\n\n"
            f"Now, reason step by step and output the final answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_text_prompt(
    role: str, question: str, context: str = ""
) -> List[Dict[str, str]]:
    """Build chat messages for text mode (prior agents' output in prompt)."""
    if role == "refiner":
        user_content = (
            f"You are a Refiner Agent. You are provided with:\n"
            f"(1) the original question, and\n"
            f"(2) the previous agents' plan and critique in text format.\n\n"
            f"Your job is to incorporate the feedback and produce an improved, "
            f"refined step-by-step plan.\n\n"
            f"## Input Question:\n{question}\n\n"
            f"## Previous Agents' Output:\n{context}\n\n"
            f"Make sure your output plan is logically correct, concise, and sufficient "
            f"to guide final problem solving.\n"
            f"Now, output your refined plan below:"
        )
    elif role == "judger":
        user_content = (
            f"Target Question: {question}\n\n"
            f"You are the final solver agent in a sequential multi-agent system "
            f"(planner -> critic -> refiner -> solver).\n"
            f"You are provided with the Refiner Agent's plan as reference.\n\n"
            f"Refined Plan from Previous Agents:\n{context}\n\n"
            f"The plan might contain irrelevant or incorrect contents. "
            f"Ignore them if they are not helpful for solving the target question.\n\n"
            f"You must reason step-by-step to solve the provided Target Question "
            f"without outputting other irrelevant information.\n\n"
            f"Now, reason step by step and output the final answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"build_text_prompt not needed for role: {role}")
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_materialize_prompt(question: str) -> List[Dict[str, str]]:
    """Short prompt to generate text from accumulated KV-cache at a JSON boundary."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Based on the plan and critique above, summarize the key points "
            f"of the plan and the feedback concisely:"
        )},
    ]


# ---------------------------------------------------------------------------
# Helpers (adapted from benchmarks/gsm8k/agents.py)
# ---------------------------------------------------------------------------

def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Render chat messages to a string using the tokenizer's chat template."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    segments = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
    segments.append("<|assistant|>")
    return "\n".join(segments)


def tokenize_prompt(
    tokenizer: Any, prompt_text: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a rendered prompt string, returning (input_ids, attention_mask)."""
    encoded = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    past_key_values: Optional[Any] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Tuple[str, Optional[Any]]:
    """Generate text, optionally with a pre-filled KV-cache."""
    prompt_len = attention_mask.sum(dim=1).tolist()[0]

    cache_position = None
    if past_key_values is not None:
        try:
            from transformers.cache_utils import Cache
            if isinstance(past_key_values, Cache):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[0][0].shape[-2]
        except ImportError:
            past_len = past_key_values[0][0].shape[-2]

        cache_position = torch.arange(
            past_len, past_len + input_ids.shape[-1],
            dtype=torch.long, device=device,
        )
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        past_key_values=past_key_values,
        cache_position=cache_position,
    )

    generated_ids = outputs.sequences[0, prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, getattr(outputs, "past_key_values", None)


def get_past_length(past_kv: Any) -> int:
    """Get sequence length from past_key_values."""
    if past_kv is None:
        return 0
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kv, Cache):
            return past_kv.get_seq_length()
    except ImportError:
        pass
    if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
        first = past_kv[0]
        if isinstance(first, (tuple, list)) and len(first) > 0:
            return first[0].shape[-2]
    return 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def auto_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_model(model_name: str, device: str):
    """Load model + tokenizer + connector, return dict with all components."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from avp.connectors.huggingface import HuggingFaceConnector

    print(f"  Loading {model_name} on {device}...")
    t0 = time.perf_counter()

    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    connector = HuggingFaceConnector(model=model, tokenizer=tokenizer, device=device)
    identity = connector.get_model_identity()

    elapsed = time.perf_counter() - t0
    print(f"  Loaded in {elapsed:.1f}s: {identity.model_family}, "
          f"hidden={identity.hidden_dim}, layers={identity.num_layers}, "
          f"kv_heads={identity.num_kv_heads}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "connector": connector,
        "identity": identity,
        "name": model_name,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_mixed_pipeline(
    models: Dict[str, Dict],
    question: str,
    latent_steps: int = 20,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 4-agent mixed-model pipeline on a single question."""
    from avp.codec import decode as avp_decode
    from avp.codec import encode_kv_cache
    from avp.handshake import CompatibilityResolver
    from avp.fallback import JSONMessage
    from avp.kv_cache import (
        deserialize_kv_cache,
        legacy_to_dynamic_cache,
        serialize_kv_cache,
    )
    from avp.types import (
        AVPMetadata,
        CommunicationMode,
        DataType,
        PayloadType,
    )

    t0 = time.perf_counter()
    session_id = str(uuid.uuid4())
    device = next(iter(models.values()))["model"].device
    device_str = str(device)

    past_kv = None
    json_content = None  # text received from JSON fallback
    hop_traces: List[Dict] = []
    total_codec_ms = 0.0
    total_wire_bytes = 0
    total_json_chars = 0
    final_answer = ""

    for idx, agent in enumerate(AGENTS):
        m = models[agent.model_key]
        connector = m["connector"]
        model = m["model"]
        tokenizer = m["tokenizer"]
        identity = m["identity"]

        # --- Handshake with next agent (for display and mode decision) ---
        next_agent = AGENTS[idx + 1] if idx + 1 < len(AGENTS) else None

        if next_agent:
            next_m = models[next_agent.model_key]
            next_identity = next_m["identity"]
            session_info = CompatibilityResolver.resolve(identity, next_identity)
            next_mode = session_info.mode
        else:
            next_mode = None  # last agent

        # --- Determine incoming mode (how this agent received data) ---
        if idx == 0:
            incoming_mode = None  # first agent
        else:
            prev_agent = AGENTS[idx - 1]
            prev_identity = models[prev_agent.model_key]["identity"]
            incoming_info = CompatibilityResolver.resolve(prev_identity, identity)
            incoming_mode = incoming_info.mode

        # --- Build prompt based on incoming mode ---
        if incoming_mode == CommunicationMode.JSON:
            # Received JSON text — include it as context
            messages = build_text_prompt(agent.role, question, json_content or "")
            past_kv = None  # fresh start on new model
        else:
            # First agent or received KV-cache
            messages = build_latent_prompt(agent.role, question)

        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device_str)

        is_final = (idx == len(AGENTS) - 1)

        if is_final:
            # --- Judger: generate final text answer ---
            hop_header = f"Final Answer ({agent.name})"
            print(f"\n{'':─<64}")
            print(f"  {hop_header}")

            text, _ = generate_text(
                model, tokenizer, input_ids, attention_mask, device_str,
                past_key_values=past_kv,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
            final_answer = text

            kv_input_len = get_past_length(past_kv)
            print(f"  KV input seq_len: {kv_input_len}")
            print(f"  Output ({len(text)} chars):")
            preview = text[:300] + ("..." if len(text) > 300 else "")
            for line in preview.split("\n"):
                print(f"    {line}")

        elif next_mode == CommunicationMode.JSON:
            # --- This agent's output goes to a different model via JSON ---
            hop_num = idx + 1
            hop_header = f"Hop {hop_num}: {agent.name} -> {next_agent.name}"
            print(f"\n{'':─<64}")
            print(f"  {hop_header}")
            print(f"  Handshake: {identity.model_family} "
                  f"({identity.hidden_dim}d, {identity.num_layers}L) <-> "
                  f"{next_identity.model_family} "
                  f"({next_identity.hidden_dim}d, {next_identity.num_layers}L) "
                  f"=> JSON")

            # First do latent steps to accumulate "thinking" in KV-cache
            past_kv = connector.generate_latent_steps(
                input_ids, latent_steps=latent_steps,
                attention_mask=attention_mask, past_key_values=past_kv,
            )

            # Then generate text from accumulated KV-cache
            mat_messages = build_materialize_prompt(question)
            mat_text = render_prompt(tokenizer, mat_messages)
            mat_ids, mat_mask = tokenize_prompt(tokenizer, mat_text, device_str)

            text, _ = generate_text(
                model, tokenizer, mat_ids, mat_mask, device_str,
                past_key_values=past_kv,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )

            # Wrap in JSONMessage
            json_msg = JSONMessage(
                session_id=session_id,
                source_agent_id=agent.name,
                target_agent_id=next_agent.name,
                content=text,
            )
            json_wire = json_msg.to_json()
            json_content = text
            past_kv = None  # discard KV-cache — switching models

            total_json_chars += len(text)

            print(f"  {agent.name}: {latent_steps} latent steps, "
                  f"then generated text ({len(text)} chars)")
            print(f"  JSONMessage: {len(json_wire)} bytes on wire")
            if verbose:
                preview = text[:200] + ("..." if len(text) > 200 else "")
                print(f"  Content: {preview}")

            hop_traces.append({
                "hop": hop_num,
                "source": agent.name,
                "target": next_agent.name,
                "mode": "JSON",
                "latent_steps": latent_steps,
                "text_chars": len(text),
                "json_wire_bytes": len(json_wire),
            })

        else:
            # --- LATENT mode: generate KV-cache, AVP codec roundtrip ---
            hop_num = idx + 1
            hop_header = f"Hop {hop_num}: {agent.name} -> {next_agent.name}"
            print(f"\n{'':─<64}")
            print(f"  {hop_header}")
            print(f"  Handshake: {identity.model_family} "
                  f"({identity.hidden_dim}d, {identity.num_layers}L) <-> "
                  f"{next_identity.model_family} "
                  f"({next_identity.hidden_dim}d, {next_identity.num_layers}L) "
                  f"=> LATENT")

            past_kv = connector.generate_latent_steps(
                input_ids, latent_steps=latent_steps,
                attention_mask=attention_mask, past_key_values=past_kv,
            )

            # AVP codec roundtrip
            codec_t0 = time.perf_counter()

            kv_bytes, kv_header = serialize_kv_cache(past_kv)
            metadata = AVPMetadata(
                session_id=session_id,
                source_agent_id=agent.name,
                target_agent_id=next_agent.name,
                model_id=m["name"],
                hidden_dim=identity.hidden_dim,
                num_layers=identity.num_layers,
                payload_type=PayloadType.KV_CACHE,
                dtype=DataType.FLOAT32,
                mode=CommunicationMode.LATENT,
            )
            wire_bytes = encode_kv_cache(kv_bytes, metadata)
            wire_size = len(wire_bytes)

            avp_msg = avp_decode(wire_bytes)
            legacy_kv, _ = deserialize_kv_cache(avp_msg.payload, device=device_str)
            past_kv = legacy_to_dynamic_cache(legacy_kv)

            codec_ms = (time.perf_counter() - codec_t0) * 1000
            total_codec_ms += codec_ms
            total_wire_bytes += wire_size
            json_content = None  # clear any prior JSON

            kv_seq_len = get_past_length(past_kv)

            print(f"  {agent.name}: {latent_steps} latent steps, "
                  f"KV seq_len={kv_seq_len}")
            print(f"  AVP wire: {wire_size:,} bytes | Codec: {codec_ms:.1f}ms")

            hop_traces.append({
                "hop": hop_num,
                "source": agent.name,
                "target": next_agent.name,
                "mode": "LATENT",
                "latent_steps": latent_steps,
                "kv_seq_len": kv_seq_len,
                "wire_bytes": wire_size,
                "codec_ms": codec_ms,
            })

    wall_time = time.perf_counter() - t0
    latent_hops = sum(1 for h in hop_traces if h["mode"] == "LATENT")
    json_hops = sum(1 for h in hop_traces if h["mode"] == "JSON")

    return {
        "question": question,
        "final_answer": final_answer,
        "wall_time": wall_time,
        "hops": hop_traces,
        "latent_hops": latent_hops,
        "json_hops": json_hops,
        "total_codec_ms": total_codec_ms,
        "total_wire_bytes": total_wire_bytes,
        "total_json_chars": total_json_chars,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AVP Mixed-Model Demo: automatic handshake negotiation"
    )
    parser.add_argument(
        "--model_a", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="First model (agents 1-2) (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--model_b", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Second model (agents 3-4) (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu/mps/cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--latent_steps", type=int, default=20,
        help="Latent steps per non-judger agent (default: 20)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens for text generation (default: 512)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Override default question",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full agent outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = auto_device(args.device)
    question = args.question or DEFAULT_QUESTION

    print("=" * 64)
    print("AVP Mixed-Model Demo")
    print("=" * 64)
    print()

    # Load both models
    print("Loading models...")
    models = {}
    models["model_a"] = load_model(args.model_a, device)
    models["model_b"] = load_model(args.model_b, device)
    print()

    id_a = models["model_a"]["identity"]
    id_b = models["model_b"]["identity"]
    print(f"Model A: {args.model_a} ({id_a.model_family}, "
          f"hidden={id_a.hidden_dim}, layers={id_a.num_layers})")
    print(f"Model B: {args.model_b} ({id_b.model_family}, "
          f"hidden={id_b.hidden_dim}, layers={id_b.num_layers})")
    print()

    pipeline_str = " -> ".join(
        f"{a.name} [{a.model_key[-1].upper()}]" for a in AGENTS
    )
    print(f"Pipeline: {pipeline_str}")
    print()

    q_preview = question[:100] + ("..." if len(question) > 100 else "")
    print(f"Question: {q_preview}")

    # Run pipeline
    result = run_mixed_pipeline(
        models=models,
        question=question,
        latent_steps=args.latent_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=args.verbose,
    )

    # Summary
    print()
    print("=" * 64)
    print(f"Summary: {len(result['hops'])} hops "
          f"({result['latent_hops']} LATENT, {result['json_hops']} JSON) | "
          f"Total: {result['wall_time']:.1f}s")
    if result["total_wire_bytes"] > 0:
        print(f"  LATENT wire: {result['total_wire_bytes']:,} bytes | "
              f"Codec overhead: {result['total_codec_ms']:.1f}ms")
    if result["total_json_chars"] > 0:
        print(f"  JSON content: {result['total_json_chars']:,} chars")
    print("=" * 64)


if __name__ == "__main__":
    main()
