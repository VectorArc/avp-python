# Engine & Framework Integration Guide

AVP works **alongside** your agent framework, not instead of it. Your framework handles routing, state, and agent lifecycle. AVP handles the LLM call.

## Engines

### HuggingFace – `pip install avp[hf]`

The reference implementation. Full latent pipeline with think/generate and cross-model rosetta.

```python
from avp import HuggingFaceConnector

connector = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Agent A thinks (builds KV-cache, no text output)
context = connector.think("Analyze this math problem: 24 * 17 + 3", steps=20)

# Agent B generates using Agent A's KV-cache
answer = connector.generate("Solve step by step: 24 * 17 + 3", context=context)
```

Cross-model:

```python
researcher = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
solver = HuggingFaceConnector.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

context = researcher.think("Analyze this problem", steps=20)
answer = solver.generate("Solve it", context=context, source=researcher, cross_model=True)
```

Calibration is automatic and one-time per model pair (~0.5–2s), cached to `~/.avp/maps/`.

### Ollama – `pip install avp[ollama]`

Uses Ollama's downloaded GGUF files. Auto-unloads the model from the Ollama server to free VRAM, then loads it via llama.cpp for latent communication.

```python
from avp.connectors.ollama import OllamaConnector

connector = OllamaConnector.from_ollama("qwen2.5:7b")
context = connector.think("Analyze this problem", steps=10)
answer = connector.generate("Solve step by step", context=context)
```

Cross-model:

```python
researcher = OllamaConnector.from_ollama("qwen2.5:7b")
solver = OllamaConnector.from_ollama("llama3.2:3b")
context = researcher.think("Analyze this", steps=10)
answer = solver.generate("Solve it", context=context, source=researcher, cross_model=True)
```

Any model you've pulled with `ollama pull` works. AVP resolves the model name to the GGUF blob on disk. No torch required.

### llama.cpp – `pip install avp[llamacpp]`

Direct GGUF file loading. Runs on CPU or GPU. Uses llama.cpp's embeddings API for hidden state extraction. No forks or custom builds required.

```python
from avp.connectors.llamacpp import LlamaCppConnector

connector = LlamaCppConnector.from_pretrained("Qwen2.5-7B-Instruct-Q4_K_M.gguf")
context = connector.think("Analyze this problem", steps=10)
answer = connector.generate("Solve step by step", context=context)
```

Cross-model:

```python
researcher = LlamaCppConnector.from_pretrained("qwen2-7b.gguf")
solver = LlamaCppConnector.from_pretrained("llama3-3b.gguf")
context = researcher.think("Analyze this", steps=10)
answer = solver.generate("Solve it", context=context, source=researcher, cross_model=True)
```

No torch required. Projection math uses numpy only.

### vLLM – `pip install avp[vllm]`

vLLM integration uses two engine plugins: a KV connector for cache transfer and a model plugin for latent thinking steps during prefill. Supports Qwen2, Llama, Mistral, and Gemma architectures. See [vLLM Integration](VLLM_INTEGRATION.md) for full details.

```python
from vllm import LLM, SamplingParams

engine = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
    kv_connector="avp.connectors.vllm_kv_connector.AVPKVConnectorV1Dynamic",
    kv_role="kv_both",
    enable_prompt_embeds=True,
)

# Agent A: think (latent steps build KV-cache, saved to file store)
engine.generate("Analyze this problem: 24 * 17 + 3", SamplingParams(max_tokens=1))

# Agent B: generate from Agent A's cached computation
output = engine.generate("Solve step by step: 24 * 17 + 3", SamplingParams(max_tokens=256))
```

---

## Frameworks

AVP works as a sidecar. Your framework sees text in, text out. The KV-cache lives in a `ContextStore` on the GPU side. The framework's state carries only string reference keys.

```
┌─────────────────────────────────────────────────┐
│  Your Framework (LangGraph / CrewAI / any)       │
│                                                  │
│  Agent A node              Agent B node          │
│    │                         │                   │
│    │  "Research X"           │  "Solve X"        │
│    ▼                         ▼                   │
│  ┌──────────────────────────────────────┐        │
│  │  avp.generate()                      │        │
│  │  ContextStore (GPU-side, in-memory)  │        │
│  │  KV-cache lives here, not in state   │        │
│  └──────────────────────────────────────┘        │
│    │                         │                   │
│    ▼                         ▼                   │
│  text result               text result           │
│  (framework stores this)   (framework stores)    │
└─────────────────────────────────────────────────┘
```

### LangChain – `pip install avp[langchain]`

`ChatAVP` is a LangChain `BaseChatModel` that uses AVP latent thinking under the hood.

```python
from avp.integrations.langchain import ChatAVP
import avp

store = avp.ContextStore(default_ttl=300)

# Researcher thinks, solver generates (linked via store key)
researcher = ChatAVP(model="Qwen/Qwen2.5-7B-Instruct", role="think",
                     store=store, store_key="task-1")
solver = ChatAVP(model="Qwen/Qwen2.5-7B-Instruct", role="generate",
                 store=store, store_key="task-1")

# In a LangGraph workflow:
researcher.invoke("Analyze this math problem: 24 * 17 + 3")
answer = solver.invoke("Solve step by step: 24 * 17 + 3")
```

### CrewAI – `pip install avp[crewai]`

`AVPLLM` is a CrewAI `BaseLLM` that uses AVP latent thinking.

```python
from avp.integrations.crewai import AVPLLM
from crewai import Agent, Task, Crew
import avp

store = avp.ContextStore(default_ttl=300)

researcher = Agent(
    role="Researcher",
    goal="Analyze math problems",
    llm=AVPLLM(model="Qwen/Qwen2.5-7B-Instruct", role="think",
               store=store, store_key="task-1"),
)
solver = Agent(
    role="Solver",
    goal="Solve math problems step by step",
    llm=AVPLLM(model="Qwen/Qwen2.5-7B-Instruct", role="generate",
               store=store, store_key="task-1"),
)
```

### AutoGen – `pip install avp[autogen]`

`AVPChatCompletionClient` is an AutoGen `ChatCompletionClient` that uses AVP latent thinking.

```python
from avp.integrations.autogen import AVPChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import avp

store = avp.ContextStore(default_ttl=300)

researcher = AssistantAgent(
    "researcher",
    model_client=AVPChatCompletionClient(
        model="Qwen/Qwen2.5-7B-Instruct", role="think",
        store=store, store_key="task-1",
    ),
)
solver = AssistantAgent(
    "solver",
    model_client=AVPChatCompletionClient(
        model="Qwen/Qwen2.5-7B-Instruct", role="generate",
        store=store, store_key="task-1",
    ),
)
```

### LangGraph (easy API pattern)

If you don't need the framework-specific integrations, AVP's easy API works with any framework:

```python
from langgraph.graph import StateGraph
from typing import TypedDict
import avp

MODEL = "Qwen/Qwen2.5-7B-Instruct"
store = avp.ContextStore(default_ttl=300)

class State(TypedDict):
    query: str
    research: str
    answer: str

def researcher(state: State) -> dict:
    text = avp.generate(
        f"Research this problem step by step: {state['query']}",
        model=MODEL, store=store, store_key="researcher",
    )
    return {"research": text}

def solver(state: State) -> dict:
    text = avp.generate(
        f"Using this research, solve: {state['query']}\n\nResearch: {state['research']}",
        model=MODEL, store=store, prior_key="researcher",
    )
    return {"answer": text}

graph = StateGraph(State)
graph.add_node("researcher", researcher)
graph.add_node("solver", solver)
graph.add_edge("researcher", "solver")
graph.set_entry_point("researcher")
graph.set_finish_point("solver")

app = graph.compile()
result = app.invoke({"query": "What is 24 * 17 + 3?"})
```

---

## ContextStore

`ContextStore` is a thread-safe, TTL-aware dictionary of `AVPContext` objects. It holds KV-cache tensors in GPU memory so they can be passed between agents without serialization.

```python
store = avp.ContextStore(default_ttl=300)  # 5 min TTL

# Store after thinking
ctx = avp.think("Research this", model=MODEL)
store.store("agent-a", ctx)

# Retrieve in another agent
ctx = store.get("agent-a")  # None after TTL expires

# Housekeeping
store.active_count    # number of live entries
store.cleanup()       # remove expired entries (also happens automatically)
```

When used with `avp.generate(store=, store_key=, prior_key=)`, storing and retrieving happens automatically.

## Requirements

- **Self-hosted models with GPU access.** AVP needs KV-cache internals that cloud APIs (OpenAI, Anthropic, Google) don't expose.
- **Same machine or datacenter.** KV-cache is 28–130 MB per transfer. This is for co-located agents, not cross-internet.
- **Supported engines:** HuggingFace Transformers, llama.cpp, Ollama, or vLLM.
