# AVP Benchmark Results: Latent vs Text Multi-Agent Communication

> 51-78% token savings | 1.5-5x faster | 7 benchmarks | 5 models | same-model + cross-model

## TL;DR

AVP transfers KV-cache between agents instead of regenerating text. Same multi-agent pipeline, fewer tokens, less compute.

**Same-model lead result:** On GSM8K with Qwen2.5-7B, latent communication matches direct single-agent accuracy (**85%**) while beating text chains by 10 percentage points, using **51% fewer tokens**. Token savings scale with agent count: 51-57% for 2-agent handoffs, 73-78% for 4-agent chains.

**Cross-model lead result:** Zero-training vocabulary projection enables latent transfer across model families. Qwen2.5-7B to Llama-3.2-3B achieves **72% accuracy** on GSM8K with a **6 KB wire payload** — the researcher's reasoning compressed into a single embedding vector.

**Where it works and where it doesn't:** Latent transfer excels on structured tasks (math, reasoning). On reading comprehension (HotpotQA), single-embedding cross-model projection drops to 8% — the information bottleneck is too severe for tasks requiring distributed context across many passages.

---

## The Problem: Multi-Agent Token Waste

In a text-based multi-agent chain, each agent receives the full conversation history as text. Every agent re-tokenizes and re-processes everything prior agents already computed.

```
+-----------+   text    +-----------+   text    +-----------+   text    +-----------+
|  Planner  | ---------> |  Critic   | ---------> |  Refiner  | ---------> |  Judger   |
|           |           |           |           |           |           |           |
| prompt:   |           | prompt:   |           | prompt:   |           | prompt:   |
|  186 tok  |           |  545 tok  |           | 1073 tok  |           | 1397 tok  |
|           |           | (+334     |           | (+850     |           | (+1186    |
|           |           |  wasted)  |           |  wasted)  |           |  wasted)  |
+-----------+           +-----------+           +-----------+           +-----------+
              Prompt grows at every hop. Context tokens are redundantly re-processed.
```

**The "communication tax":** Across three models tested, **47-53% of all tokens in text mode are wasted re-processing** — context that was already computed by a prior agent but must be re-tokenized, re-embedded, and re-attended at every subsequent hop.

No multi-agent framework today transfers computed representations between agents. Every framework — LangChain, CrewAI, AutoGen, OpenAI Swarm — copies text.

---

## What AVP Does

AVP (Agent Vector Protocol) transfers KV-cache states between agents instead of text. When Agent A finishes reasoning, its attention key-value cache — the computed representation of everything it processed — is serialized and injected into Agent B's context. Agent B reads the prior reasoning directly from attention states, skipping re-tokenization and re-computation.

```
Same-model:     Agent A's KV-cache --> inject into Agent B --> reads directly via attention
Cross-model:    Agent A's hidden state --> project to B's embedding space --> inject into Agent B
Text baseline:  Agent A generates text --> Agent B re-tokenizes and re-processes ALL of it
```

For same-model agents, AVP transfers the full KV-cache (all layers, all positions). For cross-model agents, AVP projects the final hidden state through a vocabulary-mediated bottleneck into the target model's embedding space — zero training required.

AVP defines a binary format, handshake, and codec — not the transport. It works alongside any agent protocol (A2A, MCP, gRPC, WebSockets). See the [full specification](https://github.com/VectorArc/avp-spec).

---

## Benchmark Suite

### Tasks

| Benchmark | Agents | Topology | Dataset | What It Tests |
|-----------|--------|----------|---------|---------------|
| **GSM8K 2-Agent** | 2 | Handoff | [GSM8K](https://arxiv.org/abs/2110.14168) (math) | Delegation pattern — most common real-world scenario |
| **HotpotQA** | 2 | Handoff | [HotpotQA](https://hotpotqa.github.io/) (QA) | Comprehension transfer — can KV-cache carry understanding? |
| **Fan-Out** | 3 | Fan-out/fan-in | GSM8K (math) | Parallel specialists — non-sequential topology |
| **GSM8K 4-Agent** | 4 | Chain | GSM8K (math) | Long sequential chain — maximum token savings |

### Models

| Model | Params | Family | Tied Weights |
|-------|--------|--------|--------------|
| Qwen/Qwen2.5-7B-Instruct | 7B | Qwen2 | No (untied) |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Qwen2 | Yes |
| meta-llama/Llama-3.2-3B-Instruct | 3.2B | Llama | Yes |
| meta-llama/Llama-3.2-1B-Instruct | 1B | Llama | Yes |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Qwen2 | No (untied) |

### Hardware

- **Cloud (primary):** NVIDIA A100 80GB ([Modal](https://modal.com/))
- **Local (supplementary):** NVIDIA RTX 3070 Ti Laptop (8GB VRAM), WSL2 Ubuntu, CUDA 12.8

### Communication Modes

| Mode | Description |
|------|-------------|
| **Direct** | Single agent, no chain (baseline for model capability) |
| **Latent** (AVP) | Multi-agent, KV-cache transfer between agents |
| **Text** | Multi-agent, each agent generates text, concatenated into next prompt |
| **Rosetta** | Cross-model, hidden state projection between different models |

**Parameters:** `latent_steps=20`, `max_new_tokens=512`, `temperature=0.7`, `top_p=0.95`, `seed=42`

---

## Same-Model Results

### GSM8K 2-Agent Handoff

The most common real-world pattern: one agent researches, another solves. Single KV-cache transfer hop.

**A100 cloud, n=50 (Llama) / n=20 (Qwen):**

| Model | Direct | Latent (AVP) | Text | Token Savings vs Text |
|-------|--------|-------------|------|-----------------------|
| **Qwen2.5-7B** | **85%** (17/20) | **85%** (17/20) | 75% (15/20) | **51%** |
| **Llama-3.2-3B** | **76%** (38/50) | **76%** (38/50) | 74% (37/50) | — |
| Qwen2.5-1.5B | 55% (11/20) | 60% (12/20) | 65% (13/20) | **53%** |

**Key result:** On Qwen 7B (untied weights), latent matches direct at 85% and beats text by 10 percentage points. The KV-cache carries the researcher's full reasoning with zero accuracy cost.

On Llama-3.2-3B (n=50), all three modes converge — direct, latent, and text all within 2pp. Latent's advantage here is purely efficiency: same accuracy, fewer tokens.

### HotpotQA Multi-Hop QA

Reading comprehension with 10 supporting paragraphs per question. The Finder agent reads paragraphs and identifies relevant facts; the Answerer synthesizes a final answer. In latent mode, the Answerer receives the Finder's KV-cache — it never sees the source paragraphs in its prompt.

**A100 cloud, n=20:**

| Model | Direct (EM / F1) | Latent (EM / F1) | Text (EM / F1) | Token Savings |
|-------|-------------------|-------------------|-----------------|---------------|
| **Qwen2.5-7B** | 50% / 0.63 | 40% / 0.63 | **50% / 0.72** | **22%** |
| Qwen2.5-1.5B | 45% / 0.60 | 25% / 0.46 | 35% / 0.52 | **19%** |

Latent F1 matches direct F1 on 7B (0.63 vs 0.63) despite lower exact match — the KV-cache transfers partial understanding even when exact answers differ. Text achieves highest F1 (0.72) because the Answerer can re-read the source paragraphs directly.

Token savings are lower here (19-22%) because context paragraphs are large relative to agent output — the "re-processing tax" is proportionally smaller in 2-agent QA than in multi-agent reasoning chains.

### Fan-Out Aggregation

Two specialists approach a math problem differently (algebraic and arithmetic), then an aggregator combines results. Tests non-sequential topology with sequential KV injection.

**A100 cloud, n=20:**

| Model | Direct | Latent (AVP) | Text | Token Savings |
|-------|--------|-------------|------|---------------|
| **Qwen2.5-7B** | 85% (17/20) | 65% (13/20) | **90%** (18/20) | **56%** |
| Qwen2.5-1.5B | 55% (11/20) | 55% (11/20) | 60% (12/20) | **60%** |

Fan-out latent drops at 7B (-25pp vs text). Sequential KV injection from two specialists — each approaching the problem differently — overwhelms the aggregator. The aggregator benefits from reading both specialists' explicit reasoning in text. Token savings remain strong (56-60%).

### GSM8K 4-Agent Chain (Original Benchmark)

4-agent chain (Planner, Critic, Refiner, Judger) with 3 KV-cache hops. The original benchmark — highest token savings due to cumulative context savings across 4 agents.

**RTX 3070 Ti local, n=20:**

| Model | Direct | Latent (AVP) | Text | Token Savings | Speedup |
|-------|--------|-------------|------|---------------|---------|
| **Llama-3.2-3B** | **75%** (15/20) | **70%** (14/20) | 65% (13/20) | **74%** | **2.1x** |
| Qwen2.5-1.5B | **50%** (10/20) | 35% (7/20) | 40% (8/20) | **73%** | **3.8x** |
| DeepSeek-R1-1.5B | 15% (3/20) | 45% (9/20) | **70%** (14/20) | **78%** | **3.6x** |

On Llama-3.2-3B: latent (70%) beats text (65%) by 5pp while being 2.1x faster with 74% token savings.

DeepSeek-R1-1.5B is an outlier — verbose `<think>` chain-of-thought exhausts the 512-token budget in direct mode (15%). Multi-agent text chains help dramatically (70%) by giving each agent its own token budget. Latent captures intermediate benefit (45%).

### Token Savings Scale with Agent Count

Token savings are a structural property of the protocol — eliminating redundant context re-processing — not dependent on model capability.

| Agent Count | Token Savings | Example |
|-------------|---------------|---------|
| 2 agents | 51-57% | GSM8K 2-agent handoff |
| 3 agents | 56-62% | Fan-out aggregation |
| 4 agents | 73-78% | GSM8K 4-agent chain |

Text prompts grow **O(n^2)** — each new agent re-processes all prior output. Latent prompts stay **O(n)** — prior context arrives as pre-computed KV-cache. The more agents in the chain, the bigger the advantage.

### AVP Overhead (A100)

| Benchmark | Model | Codec Overhead | Wire Size | Peak VRAM (Latent) |
|-----------|-------|---------------|-----------|-------------------|
| GSM8K 2-Agent | 7B | 37ms | 8.5 MB | 295 MB |
| GSM8K 2-Agent | 1.5B | 12ms | 4.2 MB | 943 MB |
| HotpotQA | 7B | 237ms | 76 MB | 944 MB |
| Fan-out | 7B | 248ms | 24.7 MB | 300 MB |

Codec overhead is **under 1% of wall time** for all configurations. Wire size scales with model hidden dimensions and sequence length. At datacenter bandwidth (1-10 Gbps), wire transfer adds under 5% to wall time.

### Cost Implications

Cost savings are proportional to time savings — same GPU, less compute time. In the 4-agent chain (highest savings): latent is 2-4x faster than text, which translates directly to 50-75% cost reduction per query. At scale (100K queries/day on A100 at $1.39/hr), this is **$25,000-48,000/month** in compute savings depending on model.

For 2-agent handoffs (most common pattern): ~50% cost reduction. The savings compound as agent count increases.

---

## Cross-Model Results (Rosetta Stone)

### How It Works

When agents run different models, the KV-cache can't transfer directly — different architectures have different dimensions and layer structures. Instead, AVP projects the source model's final hidden state into the target model's embedding space:

```
Source hidden state  -->  logits over vocabulary  (h @ W_src.T)
                     -->  softmax probabilities
                     -->  weighted average of target embeddings  (probs @ W_tgt)
                     -->  inject into target model
```

**Same-family** (e.g., Qwen 7B to Qwen 1.5B): Models share a tokenizer, so the full vocabulary is used for projection.

**Cross-family** (e.g., Qwen 7B to Llama 3B): Models have different tokenizers, but BPE tokenizers share many tokens (ASCII characters, common English words, punctuation). AVP identifies overlapping tokens (~85% for Qwen/Llama), renormalizes the probability distribution over just the shared tokens, and projects through the overlapping portion.

Both methods require **zero training** — the projection uses only the models' existing embedding weight matrices.

### GSM8K Cross-Model Results (A100, n=50)

| Source --> Target | Projection | Accuracy | 95% CI | Wire Size |
|-------------------|------------|----------|--------|-----------|
| Qwen 7B --> Llama 3B | vocab overlap | **72%** (36/50) | [58%-83%] | 6 KB |
| Llama 3B --> Qwen 7B | vocab overlap | **88%** (44/50) | [76%-94%] | 7 KB |
| Qwen 7B --> Qwen 1.5B | vocab mediated | **62%** (31/50) | [48%-74%] | 6 KB |
| Llama 3B --> Qwen 1.5B | vocab overlap | **60%** (30/50) | [46%-72%] | 3 KB |
| Qwen 7B --> Llama 1B | vocab overlap | **40%** (20/50) | [28%-54%] | 4 KB |

### Solver Capability Is the Dominant Variable

Rosetta accuracy tracks the target model's own capability ceiling, not the source model's:

| Target Model | Own Direct Accuracy | Best Rosetta | Penalty |
|--------------|---------------------|--------------|---------|
| Qwen 7B | 85% | 88% | none |
| Llama 3B | 76% | 72% | -4pp |
| Qwen 1.5B | 55% | 62% | none |
| Llama 1B | ~50% | 40% | -10pp |

The projection is not the bottleneck — the target model's inherent capability is. A 7B researcher doesn't make a 1B solver smarter; but it doesn't degrade the solver either (except on the weakest model).

**Researcher quality barely matters.** Same target, different source: Qwen 7B to Qwen 1.5B = 62%, Llama 3B to Qwen 1.5B = 60%. The 7B's extra reasoning capability doesn't meaningfully help the 1.5B solver — the bottleneck is on the receiving end.

### Task Sensitivity: Where Cross-Model Works and Fails

The single-embedding projection was tested across three task types (A100, n=50, Qwen 7B to Llama 3B):

| Benchmark | Rosetta Accuracy | Same-Model 7B Latent | Gap |
|-----------|-----------------|---------------------|-----|
| **GSM8K 2-Agent** | **72%** | 85% | -13pp |
| **Fan-Out** | **56%** | 65% | -9pp |
| **HotpotQA** | **8% EM** / 0.21 F1 | 40% EM / 0.63 F1 | catastrophic |

Task complexity determines projection viability:

- **Structured math** (GSM8K): The question is short and structured. A single embedding encodes "approach this as a rate problem" effectively. **72%.**
- **Multi-specialist math** (fan-out): Two specialists' reasoning compressed to one vector. Partial signal loss, but math structure is forgiving. **56%.**
- **Reading comprehension** (HotpotQA): The Finder reads 10 paragraphs (1000-2000 tokens) and must transfer that comprehension through a single 3072-dimensional vector. The information bottleneck is too severe. **8%.**

The HotpotQA result demonstrates a fundamental limitation of single-embedding projection: distributed comprehension across many token positions cannot be compressed into one vector.

**Quality gate:** The SDK includes a per-transfer advisory gate (`avp.assess_transfer(prompt_tokens=N)`) that recommends latent vs JSON fallback based on prompt length. Short structured prompts (<=512 tokens, e.g. GSM8K) pass; long comprehension prompts (>512 tokens, e.g. HotpotQA with 10 paragraphs) are flagged for JSON fallback. The gate is advisory — callers decide how to act.

### Wire Size Comparison

| Transfer Mode | Wire Size | Use Case |
|--------------|-----------|----------|
| Same-model KV-cache | 4-76 MB | Full-fidelity, same model/architecture |
| Cross-model Rosetta | 3-7 KB | Cross-model/cross-family, structured tasks |
| Text (JSON) | 1-5 KB | Universal fallback, any model combination |

Cross-model Rosetta achieves near-solver-ceiling accuracy on math tasks with wire size comparable to text — but carries latent reasoning instead of surface tokens.

---

## Limitations

1. **Sample sizes.** Same-model cloud results use n=20-50. Cross-model results use n=50. Accuracy differences of 5-10pp are within sampling noise at these sizes. Token savings percentages (structural property) are robust; accuracy comparisons need n=200+ to be conclusive.

2. **Small models (1B-7B).** Latent transfer may behave differently on 13B+ models where attention heads have more capacity and can better utilize injected KV states.

3. **Single-embedding bottleneck.** Cross-model projection currently transfers one hidden state vector per hop. This works for structured tasks (math: 72%) but fails for comprehension (HotpotQA: 8%). The SDK provides an advisory quality gate (`avp.assess_transfer()`) that recommends JSON fallback for long prompts where single-embedding transfer is unlikely to work.

4. **Fan-out accuracy gap.** Sequential KV injection from multiple specialists loses signal at 7B scale (-25pp vs text). The aggregator benefits from reading each specialist's explicit reasoning rather than merged KV states.

5. **Self-hosted only.** AVP requires direct KV-cache access. Cloud APIs (OpenAI, Anthropic, Google) don't expose KV-cache internals. AVP is for self-hosted deployments — vLLM, HuggingFace Transformers, or custom inference.

6. **VRAM footprint.** Same-model latent uses up to ~1 GB peak for 1.5B-3B models. Cross-model rosetta requires both models in memory simultaneously (~10 GB for 7B + 3B).

7. **Bandwidth requirements.** Same-model KV-cache transfer: 4-76 MB per sample. Practical only at datacenter bandwidth (>=1 Gbps). Cross-model projection: 3-7 KB — works anywhere.

8. **Temperature sampling variance.** With temperature=0.7, individual sample results vary between runs. Aggregate metrics (token savings, speed ratios) are stable; per-sample accuracy is noisy.

---

## Reproduce These Results

**Requirements:** NVIDIA GPU with >= 8GB VRAM, CUDA 12+, Python 3.10+

```bash
# Clone and install
git clone https://github.com/VectorArc/avp-python.git
cd avp-python
pip install -e ".[latent]"
pip install datasets
```

### Same-Model Benchmarks

```bash
# GSM8K 2-Agent — most realistic pattern, strongest latent result
python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --mode all --max_samples 20 --latent_steps 20 --seed 42

# HotpotQA — reading comprehension, latent transfers understanding
python benchmarks/hotpotqa/run_hotpotqa.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --mode all --max_samples 20 --latent_steps 20 --seed 42

# Fan-out aggregation — non-sequential topology
python benchmarks/fanout/run_fanout.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --mode all --max_samples 20 --latent_steps 20 --seed 42

# Original 4-agent chain — highest token savings
python benchmarks/gsm8k/run_gsm8k.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --mode all --max_samples 20 --latent_steps 20 --seed 42
```

### Cross-Model Benchmarks

```bash
# Qwen 7B --> Llama 3B (cross-family, requires both models in memory)
python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_b meta-llama/Llama-3.2-3B-Instruct \
    --mode rosetta --max_samples 20 --latent_steps 20 --seed 42

# Same-family cross-size (Qwen 7B --> Qwen 1.5B)
python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --model_b Qwen/Qwen2.5-1.5B-Instruct \
    --mode rosetta --max_samples 20 --latent_steps 20 --seed 42
```

**Notes:**
- First run downloads models (~3-6 GB each) from HuggingFace
- Llama models are gated — request access at [meta-llama](https://huggingface.co/meta-llama) and set `HF_TOKEN`
- Cross-model rosetta requires VRAM for both models simultaneously (~10 GB for 7B + 3B)
- Each `--mode all` run takes 10-50 minutes depending on model size and GPU
- Run `--mode latent` or `--mode text` individually for faster iteration
- Results are saved to `benchmarks/results/` as JSON

---

## Upcoming

- **Multi-embedding transfer** — Send N hidden states instead of one, addressing the single-embedding bottleneck for comprehension tasks
- **Compact hidden state mode** — Same-model transfer using hidden states instead of full KV-cache (~60x smaller wire)
- **Larger sample sizes** — N=200+ for statistically significant accuracy comparisons
- **vLLM serving engine benchmarks** — AVP through vLLM's KVConnectorBase_V1 plugin
- **KV-cache compression** — CacheGen-style compression targeting 3-4x wire size reduction
- **Larger models** — 13B+ to test scaling behavior of both same-model and cross-model transfer
