#!/usr/bin/env python3
"""Two-agent demo: Researcher sends embeddings to Analyst over AVP.

Agent A ("Researcher"):
  - Takes a text query
  - Generates a real embedding using sentence-transformers (all-MiniLM-L6-v2)
  - Encodes as AVP binary
  - Sends to Agent B via AVPClient

Agent B ("Analyst"):
  - Runs a FastAPI server on localhost
  - Receives AVP binary, decodes embedding
  - Computes cosine similarity against a local corpus
  - Returns best match

Usage:
  pip install -e ".[demo]"
  python examples/agent_demo.py
"""

import asyncio
import threading
import time

import numpy as np
import uvicorn

import avp

# --- Corpus for Agent B ---

CORPUS = [
    "Machine learning models can process large datasets efficiently.",
    "The weather forecast predicts rain for the weekend.",
    "Neural networks are inspired by biological brain structures.",
    "The stock market experienced significant volatility today.",
    "Transfer learning allows models to leverage pre-trained knowledge.",
    "The new restaurant downtown serves excellent Italian food.",
    "Attention mechanisms revolutionized natural language processing.",
    "The hiking trail offers beautiful mountain views.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("=" * 60)
    print("AVP Agent Demo: Researcher → Analyst")
    print("=" * 60)

    # --- Load embedding model ---
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\nsentence-transformers not installed. Using random embeddings.")
        print("Install with: pip install -e '.[demo]'\n")
        run_with_random_embeddings()
        return

    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded: {dim}-dimensional embeddings")

    # Pre-encode corpus for Agent B
    print("Encoding corpus...")
    corpus_embeddings = model.encode(CORPUS, convert_to_numpy=True).astype(np.float32)

    # --- Start Agent B (Analyst) ---
    async def analyst_handler(msg: avp.AVPMessage) -> dict:
        """Process incoming embedding: find most similar corpus entry."""
        query_emb = msg.embedding
        similarities = [cosine_similarity(query_emb, ce) for ce in corpus_embeddings]
        best_idx = int(np.argmax(similarities))

        # Encode the best match embedding as AVP for the response payload
        response_avp = avp.encode(
            corpus_embeddings[best_idx],
            model_id="all-MiniLM-L6-v2",
            agent_id="analyst",
        )

        return {
            "best_match": CORPUS[best_idx],
            "similarity": round(similarities[best_idx], 4),
            "match_index": best_idx,
            "response_avp_size": len(response_avp),
            "received_from": msg.metadata.agent_id or "unknown",
        }

    app = avp.create_app(analyst_handler)
    config = uvicorn.Config(app, host="127.0.0.1", port=9200, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server
    import httpx

    for _ in range(50):
        try:
            httpx.get("http://127.0.0.1:9200/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)
    else:
        print("ERROR: Server did not start")
        return

    print("Agent B (Analyst) running on http://127.0.0.1:9200\n")

    # --- Agent A (Researcher) sends queries ---
    queries = [
        "How do transformers work in AI?",
        "What's a good place to eat?",
        "Tell me about deep learning architectures.",
    ]

    with avp.AVPClient("http://127.0.0.1:9200", agent_id="researcher") as client:
        for query in queries:
            print(f"Query: \"{query}\"")

            # Generate embedding
            t0 = time.perf_counter()
            query_emb = model.encode(query, convert_to_numpy=True).astype(np.float32)
            embed_time = time.perf_counter() - t0

            # Encode as AVP
            t0 = time.perf_counter()
            avp_data = avp.encode(
                query_emb,
                model_id="all-MiniLM-L6-v2",
                agent_id="researcher",
                compression=avp.CompressionLevel.BALANCED,
            )
            encode_time = time.perf_counter() - t0

            # Transmit
            t0 = time.perf_counter()
            resp = client.transmit(
                query_emb,
                model_id="all-MiniLM-L6-v2",
                compression=avp.CompressionLevel.BALANCED,
            )
            transmit_time = time.perf_counter() - t0

            result = resp.json()
            from avp.utils import embedding_to_json

            json_size = len(embedding_to_json(query_emb, {"model_id": "all-MiniLM-L6-v2"}))

            print(f"  Best match: \"{result['best_match']}\"")
            print(f"  Similarity: {result['similarity']}")
            print(f"  AVP size: {len(avp_data)} bytes | JSON would be: {json_size} bytes "
                  f"({json_size / len(avp_data):.1f}x larger)")
            print(f"  Timing: embed={embed_time*1000:.1f}ms, "
                  f"encode={encode_time*1000:.2f}ms, "
                  f"roundtrip={transmit_time*1000:.1f}ms")
            print()

    print("Demo complete.")
    server.should_exit = True


def run_with_random_embeddings():
    """Fallback demo using random embeddings (no torch required)."""
    dim = 384
    corpus_embeddings = np.random.randn(len(CORPUS), dim).astype(np.float32)

    async def handler(msg: avp.AVPMessage) -> dict:
        query_emb = msg.embedding
        sims = [cosine_similarity(query_emb, ce) for ce in corpus_embeddings]
        best_idx = int(np.argmax(sims))
        return {
            "best_match": CORPUS[best_idx],
            "similarity": round(sims[best_idx], 4),
        }

    app = avp.create_app(handler)
    config = uvicorn.Config(app, host="127.0.0.1", port=9200, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import httpx

    for _ in range(50):
        try:
            httpx.get("http://127.0.0.1:9200/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.1)

    print("\nAgent B (Analyst) running with random embeddings\n")

    queries = ["How do transformers work?", "Good restaurant?", "Deep learning?"]
    with avp.AVPClient("http://127.0.0.1:9200", agent_id="researcher") as client:
        for query in queries:
            emb = np.random.randn(dim).astype(np.float32)
            avp_data = avp.encode(emb, model_id="random", agent_id="researcher")

            t0 = time.perf_counter()
            resp = client.transmit(emb, model_id="random")
            elapsed = time.perf_counter() - t0

            result = resp.json()
            from avp.utils import embedding_to_json

            json_size = len(embedding_to_json(emb))

            print(f"Query: \"{query}\"")
            print(f"  Match: \"{result['best_match']}\"")
            print(f"  AVP: {len(avp_data)} bytes | JSON: {json_size} bytes "
                  f"({json_size / len(avp_data):.1f}x larger)")
            print(f"  Roundtrip: {elapsed*1000:.1f}ms\n")

    print("Demo complete (random embeddings).")
    server.should_exit = True


if __name__ == "__main__":
    main()
