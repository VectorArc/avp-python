#!/usr/bin/env python3
"""Cross-process AVP demo: two separate processes communicate via HTTP.

Demonstrates AVP latent transfer between independent processes, each running
their own model instance. The client (Researcher agent) produces latent
context via think(), serializes it to AVP binary format, and sends it over
HTTP. The server (Solver agent) deserializes the context and generates an
answer using its own model instance.

Usage:
    # Terminal 1 — start the solver server
    python examples/cross_process_demo.py server --model Qwen/Qwen2.5-1.5B-Instruct

    # Terminal 2 — run the researcher client
    python examples/cross_process_demo.py client --model Qwen/Qwen2.5-1.5B-Instruct \
        --question "What is 15% of 240?"
"""

import argparse
import sys
import os

# Ensure avp-python root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_server(model_name: str, device: str, port: int, latent_steps: int):
    """Start a FastAPI server that receives AVP context and generates answers."""
    import json
    import torch
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from avp import HuggingFaceConnector
    from avp.context import AVPContext

    print(f"Loading model: {model_name} on {device}...")
    connector = HuggingFaceConnector.from_pretrained(model_name, device=device)
    identity = connector.get_model_identity()
    print(f"Model loaded. Identity: {identity.model_family}, "
          f"hidden_dim={identity.hidden_dim}, layers={identity.num_layers}")
    print(f"Model hash: {connector._model_hash[:16]}...")

    app = FastAPI(title="AVP Cross-Process Demo — Solver Agent")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name, "model_hash": connector._model_hash}

    @app.post("/avp/solve")
    async def solve(request: Request):
        """Receive AVP context bytes, generate answer, return it."""
        body = await request.body()
        question = request.headers.get("X-Question", "")

        print(f"\n--- Received AVP context ({len(body):,} bytes) ---")
        print(f"Question: {question}")

        # Deserialize AVP context
        context = AVPContext.from_bytes(body, device=device)
        print(f"Context: seq_len={context.seq_len}, steps={context.num_steps}, "
              f"model_family={context.model_family}")

        # Generate answer using the received context
        prompt = (
            f"You received analysis from a Researcher agent. "
            f"Using their work, solve this problem.\n\n"
            f"Question: {question}\n\n"
            f"Compute the final answer and put it inside \\boxed{{}}."
        )
        answer = connector.generate(prompt, context=context)
        print(f"Generated answer ({len(answer)} chars): {answer[:200]}...")

        return JSONResponse({
            "answer": answer,
            "context_bytes": len(body),
            "context_seq_len": context.seq_len,
        })

    print(f"\nStarting server on port {port}...")
    print(f"Waiting for AVP context from client...\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def run_client(model_name: str, device: str, port: int, question: str, latent_steps: int):
    """Run think(), serialize context, POST to server, print answer."""
    import time
    import httpx

    from avp import HuggingFaceConnector

    print(f"Loading model: {model_name} on {device}...")
    connector = HuggingFaceConnector.from_pretrained(model_name, device=device)
    identity = connector.get_model_identity()
    print(f"Model loaded. Hash: {connector._model_hash[:16]}...")

    # Step 1: Think (produce latent context)
    print(f"\n--- Researcher Agent: thinking ({latent_steps} steps) ---")
    print(f"Question: {question}")
    t0 = time.perf_counter()

    research_prompt = (
        f"You are a Researcher Agent. Analyze this math problem. "
        f"Identify the key quantities, relationships, and approach.\n\n"
        f"Question: {question}"
    )
    context = connector.think(research_prompt, steps=latent_steps)
    think_time = time.perf_counter() - t0
    print(f"Think complete: seq_len={context.seq_len}, time={think_time:.2f}s")

    # Step 2: Serialize to AVP binary
    t0 = time.perf_counter()
    avp_bytes = context.to_bytes(
        session_id="cross-process-demo",
        source_agent_id="researcher",
        target_agent_id="solver",
        model_id=model_name,
    )
    serialize_time = time.perf_counter() - t0
    print(f"Serialized: {len(avp_bytes):,} bytes in {serialize_time*1000:.1f}ms")

    # Step 3: POST to server
    server_url = f"http://localhost:{port}/avp/solve"
    print(f"\n--- Sending to server at {server_url} ---")
    t0 = time.perf_counter()

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            server_url,
            content=avp_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Question": question,
            },
        )

    roundtrip_time = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}")
        sys.exit(1)

    result = resp.json()

    # Step 4: Print results
    print(f"\n{'=' * 60}")
    print("CROSS-PROCESS AVP DEMO RESULTS")
    print(f"{'=' * 60}")
    print(f"Question: {question}")
    print(f"Answer:   {result['answer']}")
    print(f"\nMetrics:")
    print(f"  Think time:        {think_time:.2f}s ({latent_steps} latent steps)")
    print(f"  Serialize time:    {serialize_time*1000:.1f}ms")
    print(f"  Wire size:         {result['context_bytes']:,} bytes")
    print(f"  Context seq_len:   {result['context_seq_len']}")
    print(f"  Server roundtrip:  {roundtrip_time:.2f}s (includes generation)")
    print(f"\nProcess isolation:")
    print(f"  - Client PID: {os.getpid()}")
    print(f"  - Separate model instances in client and server")
    print(f"  - Transport: HTTP + AVP binary format")
    print(f"  - Context serialized/deserialized across process boundary")
    print(f"{'=' * 60}")


def auto_device() -> str:
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Cross-process AVP demo: server + client"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Server subcommand
    server_parser = subparsers.add_parser("server", help="Start the solver server")
    server_parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model to load (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    server_parser.add_argument("--device", type=str, default=None)
    server_parser.add_argument("--port", type=int, default=9200)
    server_parser.add_argument("--latent_steps", type=int, default=20)

    # Client subcommand
    client_parser = subparsers.add_parser("client", help="Run the researcher client")
    client_parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model to load (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    client_parser.add_argument("--device", type=str, default=None)
    client_parser.add_argument("--port", type=int, default=9200)
    client_parser.add_argument(
        "--question", type=str, default="What is 15% of 240?",
        help="Math question to solve",
    )
    client_parser.add_argument("--latent_steps", type=int, default=20)

    args = parser.parse_args()
    device = args.device or auto_device()

    if args.command == "server":
        run_server(args.model, device, args.port, args.latent_steps)
    elif args.command == "client":
        run_client(args.model, device, args.port, args.question, args.latent_steps)


if __name__ == "__main__":
    main()
