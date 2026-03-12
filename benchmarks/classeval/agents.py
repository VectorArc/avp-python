"""Agent definitions and prompt builders for the ClassEval benchmark.

ClassEval tests incremental class generation where methods depend on each other.
Three generation strategies:
1. Direct -- generate the entire class in one shot
2. Text chain -- generate methods one at a time, accumulating text context
3. Latent chain -- generate methods one at a time, accumulating KV-cache context
"""

from typing import Dict, List


SYSTEM_MESSAGE = (
    "You are an expert Python class implementer. "
    "You write clean, correct, production-quality Python code."
)


def build_direct_prompt(
    skeleton: str,
    class_description: str,
    import_statement: str,
) -> List[Dict[str, str]]:
    """Build prompt for direct (single-shot) class generation.

    The model generates the entire class implementation at once.
    """
    user_content = (
        f"Implement the following Python class completely. "
        f"Fill in ALL method bodies. Only output the class code "
        f"-- no explanation, no tests, no markdown fences.\n\n"
        f"## Required imports:\n{import_statement}\n\n"
        f"## Class description:\n{class_description}\n\n"
        f"## Class skeleton:\n{skeleton}"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_text_prompt(
    skeleton: str,
    class_description: str,
    method_info: Dict,
    prior_methods_text: str,
    import_statement: str,
) -> List[Dict[str, str]]:
    """Build prompt for text-chain incremental generation.

    Prior methods are included as text context in the prompt.
    """
    method_name = method_info["method_name"]
    method_desc = method_info.get("method_description", "")

    prior_section = ""
    if prior_methods_text.strip():
        prior_section = (
            f"\n\n## Already implemented methods:\n"
            f"```python\n{prior_methods_text}\n```\n"
        )

    user_content = (
        f"You are implementing the class below one method at a time. "
        f"Now implement the method `{method_name}`. "
        f"Only output the method body (indented with 4 spaces, as it would appear "
        f"inside the class). No explanation, no tests, no markdown fences, "
        f"no class definition, no decorator -- just the method def and its body.\n\n"
        f"## Required imports:\n{import_statement}\n\n"
        f"## Class skeleton:\n{skeleton}\n\n"
        f"## Method to implement:\n{method_name}: {method_desc}"
        f"{prior_section}"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_latent_prompt(
    skeleton: str,
    class_description: str,
    method_info: Dict,
    import_statement: str,
) -> List[Dict[str, str]]:
    """Build prompt for latent-chain incremental generation.

    Prior context is carried via KV-cache, so the prompt only contains
    the skeleton and current method description.
    """
    method_name = method_info["method_name"]
    method_desc = method_info.get("method_description", "")

    user_content = (
        f"You are implementing the class below one method at a time. "
        f"Now implement the method `{method_name}`. "
        f"Only output the method body (indented with 4 spaces, as it would appear "
        f"inside the class). No explanation, no tests, no markdown fences, "
        f"no class definition, no decorator -- just the method def and its body.\n\n"
        f"## Required imports:\n{import_statement}\n\n"
        f"## Class skeleton:\n{skeleton}\n\n"
        f"## Method to implement:\n{method_name}: {method_desc}"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
