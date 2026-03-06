"""HumanEval code evaluation: extract code, execute tests, compute pass@1."""

import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    """Extract generated code from model output.

    Handles:
    1. Markdown code fences (```python ... ```)
    2. Raw function body after signature
    3. Complete function definition

    Returns the complete function (signature + body) ready for execution.
    """
    # Strip markdown fences if present
    fenced = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        code = fenced[-1].strip()
    else:
        code = text.strip()

    # If the output contains a complete function definition, use it
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        return code

    # Otherwise, assume the model output is just the body — prepend the prompt
    # The prompt already contains the function signature and docstring
    return prompt + code


def execute_code(code: str, test_code: str, timeout: int = 10) -> Dict:
    """Execute generated code against test cases in a subprocess.

    Args:
        code: The complete function code to test
        test_code: The test assertions from HumanEval
        timeout: Maximum execution time in seconds

    Returns:
        Dict with 'passed' (bool), 'error' (str or None)
    """
    # Combine function code + test code
    full_code = code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=True, encoding="utf-8"
    ) as f:
        f.write(full_code)
        f.flush()

        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return {"passed": True, "error": None}
            else:
                error = result.stderr.strip()
                # Truncate very long errors
                if len(error) > 500:
                    error = error[:500] + "..."
                return {"passed": False, "error": error}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"passed": False, "error": str(e)}


def build_test_code(test: str, entry_point: str) -> str:
    """Build executable test code from HumanEval test field.

    The HumanEval test field contains a check() function definition.
    We need to call it after defining it.
    """
    return test + f"\ncheck({entry_point})\n"


def check_correct(
    generated_code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: int = 10,
) -> Dict:
    """Full evaluation pipeline: extract code, build tests, execute.

    Returns:
        Dict with 'passed', 'error', 'code' (the extracted code)
    """
    code = extract_code(generated_code, prompt, entry_point)
    test_code = build_test_code(test, entry_point)
    result = execute_code(code, test_code, timeout=timeout)
    result["code"] = code
    return result


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute pass@1 accuracy."""
    total = len(results)
    if total == 0:
        return {"pass_at_1": 0.0, "passed": 0, "total": 0}

    passed = sum(1 for r in results if r.get("correct", False))
    return {
        "pass_at_1": passed / total,
        "passed": passed,
        "total": total,
    }
