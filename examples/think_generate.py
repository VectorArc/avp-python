#!/usr/bin/env python3
"""avp.think() / avp.generate() — the easiest way to use AVP.

Requires: pip install "avp[hf]" (torch + transformers + GPU)

Run:
    python examples/think_generate.py
"""

import avp


def easy_api_same_model():
    """Same-model latent transfer via avp.think() and avp.generate()."""
    print("--- Easy API: Same Model ---\n")

    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # noqa: N806
    prompt = "Analyze this math problem: 24 * 17 + 3"

    # Agent A: latent thinking (builds KV-cache, no text output)
    context = avp.think(prompt, model=MODEL, steps=20)
    print(f"  Thinking done: {context.num_steps} steps, seq_len={context.seq_len}")

    # Agent B: generate using A's context
    answer = avp.generate(prompt, model=MODEL, context=context, steps=0)
    print(f"  Answer: {answer[:200]}")

    # Or do both in one call
    answer = avp.generate(prompt, model=MODEL, steps=20)
    print(f"  One-liner: {answer[:200]}")


def easy_api_cross_model():
    """Cross-model transfer via source_model parameter."""
    print("\n--- Easy API: Cross Model ---\n")

    prompt = "Solve step by step: 24 * 17 + 3"

    # Think on larger model, generate on smaller model
    answer = avp.generate(
        prompt,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        source_model="Qwen/Qwen2.5-7B-Instruct",
        steps=20,
    )
    print(f"  Cross-model answer: {answer[:200]}")


def connector_api():
    """Connector API for full control."""
    print("\n--- Connector API ---\n")

    connector = avp.HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    prompt = "Analyze this math problem: 24 * 17 + 3"

    # Agent A: think
    context = connector.think(prompt, steps=20)
    print(f"  Context: {context.num_steps} steps, seq_len={context.seq_len}")
    print(f"  Model hash: {context.model_hash[:16]}...")

    # Agent B: generate
    answer = connector.generate(prompt, context=context)
    print(f"  Answer: {answer[:200]}")

    # Cross-process serialization
    wire_bytes = context.to_bytes()
    print(f"  Wire size: {len(wire_bytes) / 1024 / 1024:.1f} MB")

    restored = avp.AVPContext.from_bytes(wire_bytes)
    print(f"  Restored: {restored.num_steps} steps, seq_len={restored.seq_len}")


def context_store():
    """Multi-turn context passing with ContextStore."""
    print("\n--- Context Store ---\n")

    store = avp.ContextStore(default_ttl=300)
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # noqa: N806

    # Agent A: researcher
    avp.generate(
        "Research: What is the capital of France?",
        model=MODEL, store=store, store_key="researcher",
    )
    print("  Researcher: stored context")

    # Agent B: writer (receives researcher's context)
    answer = avp.generate(
        "Write a summary about the capital of France",
        model=MODEL, store=store, store_key="writer", prior_key="researcher",
    )
    print(f"  Writer: {answer[:200]}")

    print(f"  Active contexts: {store.active_count}")


if __name__ == "__main__":
    print("=== AVP Easy API Demo ===\n")
    print("Note: Requires GPU and model downloads.\n")

    try:
        easy_api_same_model()
    except Exception as e:
        print(f"  Skipped (requires GPU + model): {e}")

    try:
        easy_api_cross_model()
    except Exception as e:
        print(f"  Skipped (requires GPU + 2 models): {e}")

    try:
        connector_api()
    except Exception as e:
        print(f"  Skipped (requires GPU + model): {e}")

    try:
        context_store()
    except Exception as e:
        print(f"  Skipped (requires GPU + model): {e}")
