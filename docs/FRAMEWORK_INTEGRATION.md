# Framework Integration Guide

AVP works **alongside** your agent framework, not instead of it. Your framework handles routing, state, and agent lifecycle. AVP handles the LLM call — replacing `llm.invoke()` with `avp.generate()`.

## The Pattern: Sidecar

Your framework sees text in, text out. It never touches tensors. The KV-cache lives in a `ContextStore` on the GPU side. The framework's state carries only string reference keys.

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

The framework never knows AVP exists. It gets text back, stores text in its state, checkpoints text to its database. The latent context is a side-channel that accelerates the LLM call.

## Before and After

**Before (text chain):**
```python
def agent_a(state):
    result = llm.invoke("Research: " + state["query"])
    return {"research": result.content}

def agent_b(state):
    result = llm.invoke("Solve using this research: " + state["research"])
    return {"answer": result.content}
```

**After (AVP latent chain):**
```python
import avp

MODEL = "Qwen/Qwen2.5-7B-Instruct"
store = avp.ContextStore(default_ttl=300)

def agent_a(state):
    result = avp.generate(
        "Research: " + state["query"],
        model=MODEL, store=store, store_key="agent-a",
    )
    return {"research": result}

def agent_b(state):
    result = avp.generate(
        "Solve using this research: " + state["research"],
        model=MODEL, store=store, prior_key="agent-a",
    )
    return {"answer": result}
```

Agent B automatically receives Agent A's KV-cache via the store. The framework sees the same text interface. Token savings: 51-78%.

## LangGraph

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

LangGraph checkpoints `State` to its database — all strings, no tensors. The `ContextStore` holds KV-cache in GPU memory with TTL expiry.

## CrewAI

```python
from crewai import Agent, Task, Crew
import avp

MODEL = "Qwen/Qwen2.5-7B-Instruct"
store = avp.ContextStore(default_ttl=300)

def avp_llm_call(prompt: str, store_key: str, prior_key: str = None) -> str:
    return avp.generate(
        prompt, model=MODEL, store=store,
        store_key=store_key, prior_key=prior_key,
    )

researcher = Agent(
    role="Researcher",
    goal="Analyze math problems",
    llm=lambda prompt: avp_llm_call(prompt, store_key="researcher"),
)

solver = Agent(
    role="Solver",
    goal="Solve math problems",
    llm=lambda prompt: avp_llm_call(prompt, store_key="solver", prior_key="researcher"),
)
```

CrewAI serializes agent output as text through its JSON message bus. AVP's latent context bypasses this entirely via the side-channel store.

## Cross-Model

The same pattern works across models. Agent A thinks on a larger model, Agent B generates on a smaller one:

```python
import avp

store = avp.ContextStore(default_ttl=300)

def researcher(state):
    text = avp.generate(
        "Research: " + state["query"],
        model="Qwen/Qwen2.5-7B-Instruct",
        store=store, store_key="researcher",
    )
    return {"research": text}

def solver(state):
    text = avp.generate(
        "Solve: " + state["query"] + "\nResearch: " + state["research"],
        model="meta-llama/Llama-3.2-3B-Instruct",
        source_model="Qwen/Qwen2.5-7B-Instruct",
        store=store, prior_key="researcher",
    )
    return {"answer": text}
```

Cross-model projection is automatic — calibration happens once per model pair (~0.5-2s), cached to `~/.avp/maps/`.

## What ContextStore Does

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
- **Same machine or datacenter.** KV-cache is 28-130 MB per transfer. This is for co-located agents, not cross-internet.
- **HuggingFace Transformers or vLLM.** These are the supported inference backends.
