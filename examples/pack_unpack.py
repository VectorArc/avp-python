#!/usr/bin/env python3
"""pack() / unpack() — the easiest way to use AVP.

Layer 0: pip install avp           (no torch, no GPU)
Layer 1: pip install "avp[latent]" (adds model identity)
Layer 2: pip install "avp[latent]" (adds latent reasoning — needs GPU)

Run:
    python examples/pack_unpack.py
"""

import avp


def layer0_json_messaging():
    """Layer 0: JSON messaging. Zero optional deps."""
    print("--- Layer 0: JSON Messaging ---\n")

    # Agent A packs a message
    msg = avp.pack("The derivative of x^2 is 2x.")
    wire = msg.to_bytes()
    print(f"  Wire format: {wire.decode()}")
    print(f"  Wire size:   {len(wire)} bytes")

    # Agent B unpacks it
    text = avp.unpack(wire)
    print(f"  Unpacked:    {text}")

    # str() gives the content directly
    print(f"  str(msg):    {msg}")

    # Works with plain text too (no AVP envelope)
    print(f"  Plain text:  {avp.unpack('just a string')}")
    print()


def layer1_model_identity():
    """Layer 1: Model identity. Downloads config only (~1 KB), not weights."""
    print("--- Layer 1: Model Identity ---\n")

    try:
        msg = avp.pack(
            "The derivative of x^2 is 2x.",
            model="Qwen/Qwen2.5-1.5B-Instruct",
        )
    except Exception as e:
        print(f"  Skipped (install avp[latent] for this layer): {e}\n")
        return

    print(f"  Identity: {msg.identity}")
    print("  Wire format includes model fingerprint for compatibility checking")

    wire = msg.to_bytes()
    import json
    d = json.loads(wire)
    print(f"  Wire JSON keys: {list(d.keys())}")
    print()


def layer2_latent_transfer():
    """Layer 2: Latent reasoning. Requires GPU + torch + transformers."""
    print("--- Layer 2: Latent Transfer ---\n")

    try:
        import torch
    except ImportError:
        print("  Skipped (install avp[latent] for this layer)\n")
        return

    # Use a tiny random model for demo (no download, works on CPU)
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=2, n_positions=128)
    model = GPT2LMHeadModel(config).eval()

    class SimpleTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, **kw):
            ids = [ord(c) % 254 + 2 for c in text]
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }

        def decode(self, ids, **kw):
            return "".join(chr(int(i) % 128) for i in ids if int(i) >= 2).strip()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
            return "\n".join(parts) + "\n<|assistant|>"

    tokenizer = SimpleTokenizer()
    tokenizer.chat_template = True

    # Register the model so pack()/unpack() can find it by name.
    # In real code, pack() creates connectors automatically from HuggingFace model names.
    connector = avp.HuggingFaceConnector(model=model, tokenizer=tokenizer, device="cpu")
    avp.easy._connector_cache["demo-model"] = connector
    avp.easy._identity_cache["demo-model"] = {
        "model_id": "demo-model",
        "hidden_dim": config.n_embd,
        "num_layers": config.n_layer,
        "model_family": "gpt2",
    }

    # Agent A: pack with latent reasoning
    msg = avp.pack("What is 2 + 2?", model="demo-model", think_steps=5)
    print(f"  Context: {msg.context.num_steps} steps, seq_len={msg.context.seq_len}")
    print(f"  Wire:    {len(msg.to_bytes()):,} bytes (AVP binary, not JSON)")

    # Agent B: unpack with generation
    answer = avp.unpack(msg, model="demo-model", max_new_tokens=10)
    print(f"  Answer:  {answer!r} (random weights = garbage, real model = real answer)")
    print()


def main():
    print("=== avp.pack() / avp.unpack() ===\n")
    layer0_json_messaging()
    layer1_model_identity()
    layer2_latent_transfer()
    print("Done. See avp-python/README.md for full documentation.")


if __name__ == "__main__":
    main()
