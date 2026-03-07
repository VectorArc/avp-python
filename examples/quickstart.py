#!/usr/bin/env python3
"""AVP Quickstart: latent communication between agents.

Requires: pip install "avp[hf]"
(torch + transformers for HuggingFace connector)

Uses a tiny GPT-2 model with random weights for demo — output is garbage.
For real results, use a real model (e.g. Qwen/Qwen2.5-1.5B-Instruct on GPU).
"""

import avp


def main():
    print("=== AVP Quickstart ===\n")

    # --- High-Level API: think() / generate() ---
    print("--- High-Level API ---\n")

    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError:
        print("transformers not installed. Install with: pip install 'avp[hf]'")
        return

    # Create a tiny model (random weights, no download, CPU-only)
    config = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128)
    model = GPT2LMHeadModel(config).eval()

    # Minimal tokenizer for demo purposes
    class SimpleTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kw):
            import torch
            ids = [ord(c) % 254 + 2 for c in text]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

        def decode(self, ids, **kw):
            return "".join(chr(int(i) % 128) for i in ids if int(i) >= 2).strip()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
            return "\n".join(parts) + "\n<|assistant|>"

    tokenizer = SimpleTokenizer()
    tokenizer.chat_template = True  # signal that apply_chat_template exists

    connector = avp.HuggingFaceConnector(model=model, tokenizer=tokenizer, device="cpu")

    # Agent A: latent reasoning (no text output, builds KV-cache)
    context = connector.think("What is 2 + 2?", steps=5)
    print(f"Agent A: think() produced context with {context.num_steps} steps, "
          f"seq_len={context.seq_len}")

    # Agent B: generate with Agent A's context
    # (random weights produce empty/garbage output — real models give real answers)
    answer = connector.generate("Give the answer.", context=context, max_new_tokens=20)
    print(f"Agent B: generate() -> {answer!r}")

    # Capability discovery
    print(f"\ncan_think: {connector.can_think}")
    print(f"model_hash: {context.model_hash[:16]}...")

    # --- Cross-Process Transfer ---
    print("\n--- Cross-Process Serialization ---\n")

    wire = context.to_bytes(session_id="demo", source_agent_id="agent-a")
    print(f"Serialized:  {len(wire):,} bytes")

    restored = avp.AVPContext.from_bytes(wire, device="cpu")
    print(f"Restored:    steps={restored.num_steps}, "
          f"hash={restored.model_hash[:16]}...")
    print(f"Hash match:  {restored.model_hash == context.model_hash}")

    # --- Low-Level Codec ---
    print("\n--- Low-Level Codec (no model needed) ---\n")

    import numpy as np
    from avp.utils import embedding_to_json

    embedding = np.random.randn(384).astype(np.float32)

    # Encode as AVP binary
    metadata = avp.AVPMetadata(
        model_id="all-MiniLM-L6-v2",
        source_agent_id="quickstart-agent",
        hidden_dim=384,
        payload_type=avp.PayloadType.HIDDEN_STATE,
        dtype=avp.DataType.FLOAT32,
        tensor_shape=embedding.shape,
    )
    from avp.utils import embedding_to_bytes
    avp_data = avp.encode(embedding_to_bytes(embedding), metadata)
    print(f"AVP binary:    {len(avp_data):,} bytes")

    # Encode with compression
    avp_zstd = avp.encode(
        embedding_to_bytes(embedding), metadata,
        compression=avp.CompressionLevel.BALANCED,
    )
    print(f"AVP+zstd:      {len(avp_zstd):,} bytes")

    # JSON baseline
    json_data = embedding_to_json(embedding, {"model_id": "all-MiniLM-L6-v2"})
    print(f"JSON baseline: {len(json_data):,} bytes")
    print(f"AVP is {len(json_data) / len(avp_data):.1f}x smaller than JSON")

    # Decode and verify roundtrip
    msg = avp.decode(avp_data)
    decoded = np.frombuffer(msg.payload, dtype=np.float32)
    print(f"Roundtrip exact: {np.array_equal(embedding, decoded)}")


if __name__ == "__main__":
    main()
