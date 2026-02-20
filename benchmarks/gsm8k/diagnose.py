#!/usr/bin/env python3
"""Diagnostic script to isolate 0% correctness issues in GSM8K benchmark.

Runs a series of targeted tests to identify whether the problem is:
1. Model capability (too small for math)
2. Answer extraction bugs
3. Latent mode communication failure
4. Prompt/format issues

Usage:
    python benchmarks/gsm8k/diagnose.py [--model_name MODEL] [--device DEVICE]
"""

import argparse
import os
import sys
import time

# Fix Windows console encoding for model output containing Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Hard-coded easy test problems (avoid dataset download for quick diagnosis)
# ---------------------------------------------------------------------------
EASY_PROBLEMS = [
    {
        "question": "Tom has 5 apples. He buys 3 more apples. How many apples does Tom have now?",
        "answer": "Tom has 5 + 3 = 8 apples. #### 8",
        "gold": "8",
    },
    {
        "question": "A store sells pencils for $2 each. Sarah buys 4 pencils. How much does she pay?",
        "answer": "Sarah pays 4 * 2 = $8. #### 8",
        "gold": "8",
    },
    {
        "question": "There are 10 birds on a tree. 3 fly away. How many birds are left on the tree?",
        "answer": "10 - 3 = 7 birds. #### 7",
        "gold": "7",
    },
]

GSM8K_SAMPLE = {
    "question": (
        "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' market?"
    ),
    "answer": (
        "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. "
        "She makes 9 * 2 = <<9*2=18>>$18 every day. #### 18"
    ),
    "gold": "18",
}


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K diagnostic tests")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def auto_device(device):
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} on {device}...")
    t0 = time.perf_counter()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def generate(model, tokenizer, messages, device, max_new_tokens=512,
             temperature=0.0, do_sample=False):
    """Generate text from chat messages. Greedy by default for reproducibility."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=0.95, do_sample=True)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated_ids = outputs[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ===========================================================================
# Test 1: Answer extraction sanity check (no model needed)
# ===========================================================================
def test_answer_extraction():
    from benchmarks.gsm8k.evaluate import (
        check_correct,
        extract_gold,
        extract_gsm8k_answer,
    )

    print("\n" + "=" * 70)
    print("TEST 1: Answer Extraction Sanity Check")
    print("=" * 70)

    cases = [
        # (model_output, expected_extraction, gold_solution, expected_correct)
        ("The answer is \\boxed{18}.", "18", "She makes $18. #### 18", True),
        ("\\boxed{42}", "42", "#### 42", True),
        ("Step 1: 5+3=8. Step 2: 8*2=16. The answer is 16.", "16", "#### 16", True),
        ("The total is $2304 dollars.", "2304", "#### 18", False),
        ("I think the answer is 7 birds.", "7", "#### 7", True),
        ("No numbers here.", None, "#### 5", False),
        ("\\boxed{3,500}", "3500", "#### 3500", True),
        ("The answer is \\boxed{18.0}.", "18.0", "#### 18", True),
        # Edge: intermediate numbers before boxed answer
        ("5+3=8, then 8*2=16, so \\boxed{16}", "16", "#### 16", True),
        # Model outputs #### format (GSM8K convention)
        ("So Janet makes $18 per day. #### 18", "18", "#### 18", True),
        # "The final answer is X" pattern
        ("Step 1: 16-3-4=9. Step 2: 9*2=18. The final answer is 18.", "18", "#### 18", True),
    ]

    passed = 0
    for i, (output, expected_pred, gold_sol, expected_correct) in enumerate(cases):
        pred = extract_gsm8k_answer(output)
        gold = extract_gold(gold_sol) if gold_sol else None
        correct = check_correct(pred, gold)

        ok_pred = (pred == expected_pred)
        ok_correct = (correct == expected_correct)
        status = "PASS" if (ok_pred and ok_correct) else "FAIL"
        if status == "PASS":
            passed += 1

        if status == "FAIL":
            print(f"  [{status}] Case {i}: pred={pred!r} (expected {expected_pred!r}), "
                  f"correct={correct} (expected {expected_correct})")
            print(f"         output: {output[:80]}")
        else:
            print(f"  [{status}] Case {i}: pred={pred!r}, correct={correct}")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ===========================================================================
# Test 2: Direct single-agent solve (baseline model capability)
# ===========================================================================
def test_direct_solve(model, tokenizer, device, max_new_tokens):
    print("\n" + "=" * 70)
    print("TEST 2: Direct Single-Agent Solve (Model Capability Baseline)")
    print("=" * 70)

    from benchmarks.gsm8k.evaluate import check_correct, extract_gsm8k_answer

    problems = EASY_PROBLEMS + [GSM8K_SAMPLE]
    correct_count = 0

    for p in problems:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and give the final numeric answer inside \\boxed{}."},
            {"role": "user", "content": p["question"]},
        ]

        output = generate(model, tokenizer, messages, device,
                          max_new_tokens=max_new_tokens, do_sample=False)

        pred = extract_gsm8k_answer(output)
        correct = check_correct(pred, p["gold"])
        if correct:
            correct_count += 1

        status = "CORRECT" if correct else "WRONG"
        print(f"\n  [{status}] Q: {p['question'][:70]}...")
        print(f"    Gold: {p['gold']}, Pred: {pred}")
        print(f"    Output: {output[:300]}")

    print(f"\n  Result: {correct_count}/{len(problems)} correct")
    return correct_count, len(problems)


# ===========================================================================
# Test 3: Text mode 4-agent chain on easy problem
# ===========================================================================
def test_text_chain(model, tokenizer, device, max_new_tokens):
    print("\n" + "=" * 70)
    print("TEST 3: Text Mode 4-Agent Chain")
    print("=" * 70)

    from benchmarks.gsm8k.evaluate import check_correct, extract_gsm8k_answer
    from benchmarks.gsm8k.pipeline_text import run_text_pipeline

    problems = [EASY_PROBLEMS[0], GSM8K_SAMPLE]
    correct_count = 0

    for p in problems:
        result = run_text_pipeline(
            model, tokenizer, device,
            question=p["question"],
            gold_solution=p["answer"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            verbose=False,
        )

        if result["correct"]:
            correct_count += 1

        status = "CORRECT" if result["correct"] else "WRONG"
        print(f"\n  [{status}] Q: {p['question'][:70]}...")
        print(f"    Gold: {result['gold']}, Pred: {result['prediction']}")

        for agent_trace in result["agents"]:
            out = agent_trace["output"][:200] if agent_trace["output"] else "(latent)"
            print(f"    [{agent_trace['name']}] {out}")

    print(f"\n  Result: {correct_count}/{len(problems)} correct")
    return correct_count, len(problems)


# ===========================================================================
# Test 4: Latent coherence — does model produce intelligible text after
#          receiving KV-cache from latent steps?
# ===========================================================================
def test_latent_coherence(model, tokenizer, device, max_new_tokens):
    print("\n" + "=" * 70)
    print("TEST 4: Latent Coherence Test")
    print("=" * 70)

    import torch
    from avp.connectors.huggingface import HuggingFaceConnector

    connector = HuggingFaceConnector(model=model, tokenizer=tokenizer, device=device)

    # Simple test: feed a prompt via latent steps, then ask model to continue
    test_prompt = "The capital of France is"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Run latent steps to build KV-cache
    past_kv = connector.generate_latent_steps(
        input_ids, latent_steps=10, attention_mask=attention_mask
    )

    # Now generate text using the latent KV-cache as context
    # Use a continuation prompt
    continue_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Based on the information above, what is the answer?"},
    ]
    continue_text = tokenizer.apply_chat_template(
        continue_messages, tokenize=False, add_generation_prompt=True
    )
    cont_encoded = tokenizer(continue_text, return_tensors="pt", add_special_tokens=False)
    cont_ids = cont_encoded["input_ids"].to(device)
    cont_mask = cont_encoded["attention_mask"].to(device)

    from benchmarks.gsm8k.agents import generate_text as gen_text
    output, _ = gen_text(
        model, tokenizer, cont_ids, cont_mask, device,
        past_key_values=past_kv, max_new_tokens=100,
        temperature=0.7, top_p=0.95,
    )

    print(f"\n  Context (via latent KV): \"{test_prompt}\"")
    print(f"  Latent steps: 10")
    print(f"  Output: {output[:300]}")

    has_paris = "paris" in output.lower()
    coherent = len(output) > 5 and not output.startswith("####")
    print(f"\n  Contains 'Paris': {has_paris}")
    print(f"  Coherent output: {coherent}")

    # Test 2: Math problem via latent
    print("\n  --- Math problem via latent ---")
    math_messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": "What is 5 + 3?"},
    ]
    math_text = tokenizer.apply_chat_template(
        math_messages, tokenize=False, add_generation_prompt=True
    )
    math_encoded = tokenizer(math_text, return_tensors="pt", add_special_tokens=False)
    math_ids = math_encoded["input_ids"].to(device)
    math_mask = math_encoded["attention_mask"].to(device)

    math_kv = connector.generate_latent_steps(
        math_ids, latent_steps=10, attention_mask=math_mask
    )

    # Judger prompt for the math question
    judger_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            "You were given a math problem. Solve it and give the final answer "
            "inside \\boxed{}."
        )},
    ]
    judger_text = tokenizer.apply_chat_template(
        judger_messages, tokenize=False, add_generation_prompt=True
    )
    j_encoded = tokenizer(judger_text, return_tensors="pt", add_special_tokens=False)
    j_ids = j_encoded["input_ids"].to(device)
    j_mask = j_encoded["attention_mask"].to(device)

    math_output, _ = gen_text(
        model, tokenizer, j_ids, j_mask, device,
        past_key_values=math_kv, max_new_tokens=100,
        temperature=0.7, top_p=0.95,
    )

    print(f"  Context (via latent KV): \"What is 5 + 3?\"")
    print(f"  Output: {math_output[:300]}")

    from benchmarks.gsm8k.evaluate import extract_gsm8k_answer
    pred = extract_gsm8k_answer(math_output)
    print(f"  Extracted answer: {pred}")
    has_8 = pred == "8" if pred else False
    print(f"  Correct (expected 8): {has_8}")

    return has_paris, has_8


# ===========================================================================
# Test 5: Latent mode 4-agent chain on easy problem
# ===========================================================================
def test_latent_chain(model, tokenizer, device, max_new_tokens):
    print("\n" + "=" * 70)
    print("TEST 5: Latent Mode 4-Agent Chain")
    print("=" * 70)

    from avp.connectors.huggingface import HuggingFaceConnector
    from benchmarks.gsm8k.evaluate import check_correct, extract_gsm8k_answer
    from benchmarks.gsm8k.pipeline_latent import run_latent_pipeline

    connector = HuggingFaceConnector(model=model, tokenizer=tokenizer, device=device)
    identity = connector.get_model_identity()

    problems = [EASY_PROBLEMS[0], GSM8K_SAMPLE]
    correct_count = 0

    for p in problems:
        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity,
            question=p["question"],
            gold_solution=p["answer"],
            model_name="diagnostic",
            latent_steps=10,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            verbose=False,
        )

        if result["correct"]:
            correct_count += 1

        status = "CORRECT" if result["correct"] else "WRONG"
        print(f"\n  [{status}] Q: {p['question'][:70]}...")
        print(f"    Gold: {result['gold']}, Pred: {result['prediction']}")
        print(f"    KV seq_len at judger: {result['kv_seq_len_judger']}")
        print(f"    Judger output: {result['raw_output'][:300]}")

    print(f"\n  Result: {correct_count}/{len(problems)} correct")
    return correct_count, len(problems)


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    device = auto_device(args.device)

    print("=" * 70)
    print("GSM8K DIAGNOSTIC TESTS")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # Test 1: No model needed
    extraction_ok = test_answer_extraction()

    # Load model for remaining tests
    model, tokenizer = load_model(args.model_name, device)

    # Test 2: Direct solve (baseline capability)
    direct_correct, direct_total = test_direct_solve(
        model, tokenizer, device, args.max_new_tokens
    )

    # Test 3: Text mode chain
    text_correct, text_total = test_text_chain(
        model, tokenizer, device, args.max_new_tokens
    )

    # Test 4: Latent coherence
    latent_paris, latent_math = test_latent_coherence(
        model, tokenizer, device, args.max_new_tokens
    )

    # Test 5: Latent chain
    latent_correct, latent_total = test_latent_chain(
        model, tokenizer, device, args.max_new_tokens
    )

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  1. Answer extraction:      {'PASS' if extraction_ok else 'FAIL'}")
    print(f"  2. Direct solve:           {direct_correct}/{direct_total} correct")
    print(f"  3. Text 4-agent chain:     {text_correct}/{text_total} correct")
    print(f"  4. Latent coherence:")
    print(f"     - 'Paris' from KV:      {'YES' if latent_paris else 'NO'}")
    print(f"     - '5+3=8' from KV:      {'YES' if latent_math else 'NO'}")
    print(f"  5. Latent 4-agent chain:   {latent_correct}/{latent_total} correct")

    print("\n  Interpretation:")
    if direct_correct == 0:
        print("  -> Model too small/weak for math. 0% is a MODEL issue, not a code bug.")
        print("  -> Try a larger model: Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-Math-1.5B-Instruct")
    elif text_correct > 0 and latent_correct == 0:
        print("  -> Text mode works but latent fails. Latent communication may be broken.")
    elif text_correct == 0 and direct_correct > 0:
        print("  -> Single agent works but 4-agent chain fails. Chain prompts need fixing.")
    elif latent_paris:
        print("  -> Latent KV-cache transfer works (model understands context).")
        print("  -> Math failures are model capability, not protocol issues.")
    print("=" * 70)


if __name__ == "__main__":
    main()
