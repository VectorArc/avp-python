# Contributing to AVP Python SDK

## Dev Environment Setup

```bash
git clone https://github.com/VectorArc/avp-python.git
cd avp-python
pip install -e ".[dev]"
```

This installs torch, transformers, transport, server, and dev tools (pytest, ruff).

## Running Tests

```bash
# All tests (requires torch + transformers)
pytest tests/

# Specific test file
pytest tests/test_codec.py

# vLLM integration tests (requires CUDA GPU + vLLM installed)
pytest tests/test_vllm_integration.py -m requires_vllm
```

vLLM integration tests are excluded by default. Run them explicitly with `-m requires_vllm` on a machine with a CUDA GPU.

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Configuration is in `pyproject.toml`: line length 99, target Python 3.10+.

## Bug Reports

Open an [Issue](https://github.com/VectorArc/avp-python/issues) with:

- What you expected vs what happened
- Steps to reproduce
- Python version, torch version, GPU (if applicable)

## Feature Proposals

Open a [GitHub Issue](https://github.com/VectorArc/avp-python/issues) or submit a Pull Request. For protocol-level changes (binary format, handshake, modes), use the [spec repo's RFC process](https://github.com/VectorArc/avp-spec/blob/main/CONTRIBUTING.md).

## Code of Conduct

- Be respectful
- Focus on ideas
- Welcome newcomers
- Assume good faith
