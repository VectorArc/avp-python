"""DebugBench code evaluation: extract code, build tests from examples, execute.

DebugBench uses LeetCode-style problems with a Solution class. Unlike HumanEval,
there are no pre-written test cases -- we build them from the examples field.

Evaluation strategy:
1. Extract the fixed code from model output (markdown fences or raw)
2. Compare against the reference solution by running both with example inputs
3. If the generated code produces the same outputs as the reference solution
   on all examples, it passes. This avoids fragile input/output parsing.
"""

import ast
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional


def extract_code(text: str) -> str:
    """Extract Python code from model output.

    Handles:
    1. Markdown code fences (```python ... ```)
    2. Raw code (class definition or function)

    Returns the extracted code string.
    """
    # Try markdown fences first
    fenced = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        # Use the last fenced block (models sometimes output multiple)
        return fenced[-1].strip()

    # If the output contains a class or def, extract from there
    # Find the first 'class ' or 'def ' and take everything from there
    match = re.search(r"^(class\s|def\s)", text, re.MULTILINE)
    if match:
        return text[match.start():].strip()

    # Fall back to entire text
    return text.strip()


def _extract_method_name(solution_code: str) -> Optional[str]:
    """Extract the main method name from a Solution class.

    Parses the code with AST to find methods in the Solution class,
    excluding __init__ and helper methods (those starting with _).
    """
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = item.name
                    if name != "__init__" and not name.startswith("_"):
                        methods.append(name)
            if methods:
                return methods[0]
    return None


def _build_comparison_script(
    generated_code: str,
    solution_code: str,
    examples: List[str],
) -> Optional[str]:
    """Build a script that runs both generated and reference code on examples.

    Instead of parsing example I/O (fragile), we run both solutions and compare
    their outputs. If the generated code produces the same results as the
    reference solution on all examples, it passes.

    Returns the test script string, or None if we cannot build one.
    """
    method_name = _extract_method_name(solution_code)
    if method_name is None:
        # Try extracting from generated code as fallback
        method_name = _extract_method_name(generated_code)
    if method_name is None:
        return None

    # Parse example inputs from the examples field
    parsed_inputs = _parse_example_inputs(examples, method_name)
    if not parsed_inputs:
        return None

    # Verify both codes contain a Solution class (LeetCode pattern)
    if "class Solution" not in solution_code:
        return None
    if "class Solution" not in generated_code:
        return None

    # Build the comparison script
    # We import both solutions under different names, call the method,
    # and compare outputs.
    lines = [
        "import sys",
        "from typing import List, Optional, Tuple, Dict, Set",
        "from collections import defaultdict, deque, Counter",
        "from itertools import combinations, permutations, accumulate",
        "from functools import lru_cache",
        "from math import inf, gcd, sqrt, ceil, floor, log2",
        "from heapq import heappush, heappop, heapify",
        "from bisect import bisect_left, bisect_right",
        "import string",
        "",
        "# --- Reference Solution ---",
    ]

    # Rename Solution class in reference code
    ref_code = solution_code.replace("class Solution", "class ReferenceSolution", 1)
    lines.append(ref_code)

    lines.append("")
    lines.append("# --- Generated Solution ---")

    # Rename Solution class in generated code
    gen_code = generated_code.replace("class Solution", "class GeneratedSolution", 1)
    lines.append(gen_code)

    lines.append("")
    lines.append("# --- Test ---")
    lines.append("ref = ReferenceSolution()")
    lines.append("gen = GeneratedSolution()")
    lines.append("errors = []")

    for i, args_str in enumerate(parsed_inputs):
        lines.append(f"try:")
        lines.append(f"    ref_result = ref.{method_name}({args_str})")
        lines.append(f"    gen_result = gen.{method_name}({args_str})")
        lines.append(f"    if ref_result != gen_result:")
        lines.append(f"        errors.append('Example {i+1}: expected ' + repr(ref_result) + ' got ' + repr(gen_result))")
        lines.append(f"except Exception as e:")
        lines.append(f"    errors.append('Example {i+1}: ' + str(e))")

    lines.append("if errors:")
    lines.append("    print('FAIL: ' + '; '.join(errors), file=sys.stderr)")
    lines.append("    sys.exit(1)")
    lines.append("else:")
    lines.append("    print('PASS')")

    return "\n".join(lines)


def _parse_example_inputs(
    examples: List[str], method_name: str
) -> List[str]:
    """Parse example inputs from the DebugBench examples field.

    Each example is a string like:
        "Input: nums = [2,7,11,15], target = 9\\nOutput: [0,1]"

    Returns a list of argument strings suitable for calling the method, e.g.:
        ["[2,7,11,15], 9"]

    We extract the Input line, parse variable assignments, and reconstruct
    the argument list.
    """
    parsed = []
    for example in examples:
        # Extract the Input line
        input_match = re.search(r"Input:\s*(.*?)(?:\n|$)", example)
        if not input_match:
            continue

        input_str = input_match.group(1).strip()
        if not input_str:
            continue

        # Parse "var1 = val1, var2 = val2" pattern
        # Split on ", " that is followed by a variable name and "="
        # This handles cases like: nums = [1, 2, 3], target = 9
        args = _parse_input_assignments(input_str)
        if args:
            parsed.append(", ".join(args))

    return parsed


def _parse_input_assignments(input_str: str) -> List[str]:
    """Parse 'var = value, var2 = value2' into a list of value strings.

    Handles nested structures (lists, strings with commas) correctly by
    tracking bracket/quote depth.
    """
    # Split on top-level ", " followed by identifier "="
    # We need to find the boundaries between assignments
    assignments = []
    depth = 0
    in_string = False
    string_char = None
    current = []
    i = 0

    while i < len(input_str):
        c = input_str[i]

        if in_string:
            current.append(c)
            if c == '\\' and i + 1 < len(input_str):
                current.append(input_str[i + 1])
                i += 2
                continue
            if c == string_char:
                in_string = False
            i += 1
            continue

        if c in ('"', "'"):
            in_string = True
            string_char = c
            current.append(c)
            i += 1
            continue

        if c in ('(', '[', '{'):
            depth += 1
            current.append(c)
            i += 1
            continue

        if c in (')', ']', '}'):
            depth -= 1
            current.append(c)
            i += 1
            continue

        # At top level, check for ", identifier ="
        if depth == 0 and c == ',':
            # Look ahead: skip whitespace, then check for identifier followed by =
            rest = input_str[i + 1:].lstrip()
            if re.match(r'[a-zA-Z_]\w*\s*=\s', rest):
                assignments.append("".join(current).strip())
                current = []
                i += 1
                continue

        current.append(c)
        i += 1

    if current:
        assignments.append("".join(current).strip())

    # Extract values from "var = value" assignments
    values = []
    for assignment in assignments:
        eq_match = re.match(r'[a-zA-Z_]\w*\s*=\s*(.*)', assignment, re.DOTALL)
        if eq_match:
            values.append(eq_match.group(1).strip())
        else:
            # No variable name, treat entire thing as a value
            values.append(assignment.strip())

    return values


def _build_solution_match_script(
    generated_code: str,
    solution_code: str,
) -> str:
    """Fallback: simple structural comparison when examples cannot be parsed.

    Checks if the generated code is syntactically valid Python that defines
    the same Solution class with the same method.
    """
    lines = [
        "import ast",
        "import sys",
        "",
        "generated = " + repr(generated_code),
        "",
        "try:",
        "    tree = ast.parse(generated)",
        "except SyntaxError as e:",
        "    print(f'FAIL: SyntaxError: {e}', file=sys.stderr)",
        "    sys.exit(1)",
        "",
        "# Check that a Solution class with a method exists",
        "found_class = False",
        "found_method = False",
        "for node in ast.walk(tree):",
        "    if isinstance(node, ast.ClassDef) and node.name == 'Solution':",
        "        found_class = True",
        "        for item in node.body:",
        "            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):",
        "                if item.name != '__init__' and not item.name.startswith('_'):",
        "                    found_method = True",
        "",
        "if not found_class:",
        "    print('FAIL: No Solution class found', file=sys.stderr)",
        "    sys.exit(1)",
        "if not found_method:",
        "    print('FAIL: No method found in Solution class', file=sys.stderr)",
        "    sys.exit(1)",
        "",
        "print('PASS')",
    ]
    return "\n".join(lines)


def execute_script(script: str, timeout: int = 10) -> Dict:
    """Execute a test script in a subprocess.

    Args:
        script: Complete Python script to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dict with 'passed' (bool), 'error' (str or None)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=True, encoding="utf-8"
    ) as f:
        f.write(script)
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
                if len(error) > 500:
                    error = error[:500] + "..."
                return {"passed": False, "error": error}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"passed": False, "error": str(e)}


def check_correct(
    generated_text: str,
    solution: str,
    question: str,
    examples: List[str],
    timeout: int = 10,
) -> Dict:
    """Full evaluation pipeline: extract code, build tests, execute.

    Returns:
        Dict with 'passed', 'error', 'code' (the extracted code),
        'eval_method' (how we evaluated: 'comparison' or 'syntax')
    """
    code = extract_code(generated_text)

    # Try comparison-based evaluation first
    script = _build_comparison_script(code, solution, examples)
    if script is not None:
        result = execute_script(script, timeout=timeout)
        result["code"] = code
        result["eval_method"] = "comparison"
        return result

    # Fall back to syntax-only validation
    script = _build_solution_match_script(code, solution)
    result = execute_script(script, timeout=timeout)
    result["code"] = code
    result["eval_method"] = "syntax"
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
