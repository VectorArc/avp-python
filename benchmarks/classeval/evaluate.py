"""ClassEval code evaluation: extract methods, assemble classes, execute tests."""

import re
import subprocess
import tempfile
import textwrap
from typing import Dict, List, Optional


def extract_method_code(text: str, method_name: str) -> str:
    """Extract a single method definition from generated text.

    Handles:
    1. Markdown code fences (```python ... ```)
    2. Raw method definition with def keyword
    3. Bare method body (no def line)

    Returns the method code (def line + body) indented at class level (4 spaces).
    """
    # Strip markdown fences if present
    fenced = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        code = fenced[-1].strip()
    else:
        code = text.strip()

    # If the output contains the method def, extract it
    # Match 'def method_name(' possibly with leading whitespace
    pattern = rf"^(\s*def\s+{re.escape(method_name)}\s*\(.*)"
    match = re.search(pattern, code, re.MULTILINE | re.DOTALL)
    if match:
        # Extract from the def line to the end, or until the next top-level def
        method_block = match.group(0)
        # Find where the next def at the same or lower indentation starts
        lines = method_block.split("\n")
        if lines:
            # Determine indentation of the def line
            def_indent = len(lines[0]) - len(lines[0].lstrip())
            result_lines = [lines[0]]
            for line in lines[1:]:
                stripped = line.lstrip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    current_indent = len(line) - len(stripped)
                    if current_indent <= def_indent and stripped:
                        break
                result_lines.append(line)
            code = "\n".join(result_lines)

    # Normalize indentation to 4 spaces (class-level method)
    code = textwrap.dedent(code)
    # Re-indent to 4 spaces
    lines = code.split("\n")
    indented_lines = []
    for line in lines:
        if line.strip():
            indented_lines.append("    " + line)
        else:
            indented_lines.append("")
    return "\n".join(indented_lines).rstrip()


def extract_class_code(text: str, class_name: str) -> str:
    """Extract a complete class definition from generated text.

    Used by the direct pipeline where the model generates the entire class.
    """
    # Strip markdown fences if present
    fenced = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        code = fenced[-1].strip()
    else:
        code = text.strip()

    # Try to find the class definition
    pattern = rf"^(\s*class\s+{re.escape(class_name)}\s*[\(:].*)"
    match = re.search(pattern, code, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(0).strip()

    # If no class def found, return the whole code block
    return code


def assemble_class(
    class_name: str,
    skeleton: str,
    generated_methods: Dict[str, str],
    import_statement: str,
    methods_info: Optional[List[Dict]] = None,
) -> str:
    """Assemble generated methods into a complete class.

    Uses the skeleton's structure (class header + constructor + field init) as
    the base, then appends each generated method. This avoids fragile
    regex-based stub replacement since ClassEval skeletons have highly variable
    formatting (decorators, docstrings, static methods, empty bodies, etc.).

    Args:
        class_name: Name of the class
        skeleton: Original class skeleton (may include imports at top)
        generated_methods: {method_name: method_code} from generation.
            Each value should be the complete method with proper class-level
            indentation (4 spaces for def, 8 for body).
        import_statement: Import statements to prepend
        methods_info: Optional list of method dicts (with 'method_name' and
            'dependencies'). Used to determine which methods are @staticmethod.

    Returns:
        Complete executable class code with imports.
    """
    # Extract the class header and constructor from the skeleton.
    # Everything up to (but not including) the first non-__init__ method def.
    lines = skeleton.split("\n")
    header_lines = []
    i = 0

    # Collect imports and class preamble (comments, decorators, class def)
    # until we reach the first method definition
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this is a method def (not __init__)
        if re.match(r"    def\s+(?!__init__)\w+\s*\(", line):
            break
        # Check if this is a decorator before a non-init method
        if stripped.startswith("@") and i + 1 < len(lines):
            next_line = lines[i + 1]
            if re.match(r"    def\s+(?!__init__)\w+\s*\(", next_line):
                break

        header_lines.append(line)
        i += 1

    header = "\n".join(header_lines).rstrip()

    # Build the method name -> decorator mapping from the skeleton
    decorators = {}
    for j, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("@") and j + 1 < len(lines):
            next_stripped = lines[j + 1].strip()
            m = re.match(r"def\s+(\w+)\s*\(", next_stripped)
            if m:
                decorators[m.group(1)] = line  # preserve original indentation

    # Append generated methods
    method_sections = []
    for method_name, method_code in generated_methods.items():
        if not method_code or not method_code.strip():
            continue
        section = ""
        # Add decorator if the skeleton had one for this method
        if method_name in decorators:
            decorator_line = decorators[method_name]
            # Only add if the generated code doesn't already include it
            if "@" not in method_code.split("\n")[0]:
                section = decorator_line + "\n"
        section += method_code
        method_sections.append(section)

    class_code = header
    if method_sections:
        class_code += "\n\n" + "\n\n".join(method_sections)
    class_code = class_code.rstrip() + "\n"

    # Ensure imports are present
    if import_statement.strip():
        for line in reversed(import_statement.strip().split("\n")):
            line = line.strip()
            if line and line not in class_code:
                class_code = line + "\n" + class_code

    return class_code


def execute_tests(
    class_code: str,
    test_code: str,
    import_statement: str,
    timeout: int = 30,
) -> Dict:
    """Execute generated class code against test cases in a subprocess.

    Args:
        class_code: The complete class code (with imports already included)
        test_code: The test code from ClassEval
        import_statement: Import statements (prepended to test code if needed)
        timeout: Maximum execution time in seconds

    Returns:
        Dict with 'passed' (bool), 'error' (str or None),
        'tests_passed' (int), 'tests_total' (int)
    """
    # The class_code already has imports prepended by assemble_class.
    # Test code may need the same imports.
    full_code = class_code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=True, encoding="utf-8"
    ) as f:
        f.write(full_code)
        f.flush()

        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return {
                    "passed": True,
                    "error": None,
                    "stdout": result.stdout.strip()[:500] if result.stdout else "",
                }
            else:
                error = result.stderr.strip()
                if len(error) > 500:
                    error = error[:500] + "..."
                return {"passed": False, "error": error, "stdout": ""}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"Timeout after {timeout}s", "stdout": ""}
        except Exception as e:
            return {"passed": False, "error": str(e), "stdout": ""}


def build_test_code(test_code: str, class_name: str) -> str:
    """Build executable test code from ClassEval test field.

    The test field already contains `import unittest` and TestCase classes.
    We append a unittest.main() call if not present.
    """
    code = test_code.strip()
    if "unittest.main()" not in code:
        code += "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    return code


def check_correct(
    class_code: str,
    test_code: str,
    import_statement: str,
    timeout: int = 30,
) -> Dict:
    """Full evaluation pipeline: execute class code against tests.

    Returns:
        Dict with 'passed', 'error', 'code' (the class code)
    """
    result = execute_tests(class_code, test_code, import_statement, timeout=timeout)
    result["code"] = class_code
    return result


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute class-level pass rate."""
    total = len(results)
    if total == 0:
        return {"pass_at_1": 0.0, "passed": 0, "total": 0}

    passed = sum(1 for r in results if r.get("correct", False))
    return {
        "pass_at_1": passed / total,
        "passed": passed,
        "total": total,
    }


def compute_method_accuracy(results: List[Dict]) -> Dict:
    """Compute method-level pass rate across all classes."""
    total_methods = 0
    passed_methods = 0
    for r in results:
        method_results = r.get("method_results", {})
        for method_name, method_passed in method_results.items():
            total_methods += 1
            if method_passed:
                passed_methods += 1
    if total_methods == 0:
        return {"method_pass_rate": 0.0, "methods_passed": 0, "methods_total": 0}
    return {
        "method_pass_rate": passed_methods / total_methods,
        "methods_passed": passed_methods,
        "methods_total": total_methods,
    }
