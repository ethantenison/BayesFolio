from __future__ import annotations

import argparse
import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Violation:
    """Docstring policy violation for a newly added public symbol.

    Attributes:
        file_path: Python file where violation occurred.
        line_number: 1-based source line of symbol definition.
        symbol_name: Symbol name requiring a docstring.
        symbol_kind: Symbol category: class, function, or method.
    """

    file_path: Path
    line_number: int
    symbol_name: str
    symbol_kind: str


def _run_git_diff(base_sha: str, head_sha: str, file_path: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            base_sha,
            head_sha,
            "--",
            str(file_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _added_lines_from_diff(diff_text: str) -> set[int]:
    added_lines: set[int] = set()
    for raw_line in diff_text.splitlines():
        match = HUNK_PATTERN.match(raw_line)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1
        added_lines.update(range(start, start + count))
    return added_lines


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _is_public_method_name(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return False
    return _is_public_name(name)


def _check_nodes(tree: ast.Module, file_path: Path, added_lines: set[int] | None) -> list[Violation]:
    violations: list[Violation] = []

    def in_scope(line_number: int) -> bool:
        if added_lines is None:
            return True
        return line_number in added_lines

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_public_name(node.name) and in_scope(node.lineno) and ast.get_docstring(node) is None:
                violations.append(Violation(file_path, node.lineno, node.name, "function"))
            continue

        if not isinstance(node, ast.ClassDef):
            continue

        if _is_public_name(node.name) and in_scope(node.lineno) and ast.get_docstring(node) is None:
            violations.append(Violation(file_path, node.lineno, node.name, "class"))

        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_public_method_name(child.name) and in_scope(child.lineno) and ast.get_docstring(child) is None:
                qualified_name = f"{node.name}.{child.name}"
                violations.append(Violation(file_path, child.lineno, qualified_name, "method"))

    return violations


def check_files(
    file_paths: list[Path],
    base_sha: str | None,
    head_sha: str | None,
) -> list[Violation]:
    """Check Python files for missing docstrings on newly added public symbols.

    Args:
        file_paths: Candidate Python files to inspect.
        base_sha: Optional base git SHA for PR diff scoping.
        head_sha: Optional head git SHA for PR diff scoping.

    Returns:
        List of violations for public symbols missing docstrings.
    """

    violations: list[Violation] = []
    for file_path in file_paths:
        if file_path.suffix != ".py":
            continue
        if not file_path.exists():
            continue
        if any(part in {"tests", "experiments", "scratch"} for part in file_path.parts):
            continue

        content = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        added_lines: set[int] | None = None
        if base_sha and head_sha:
            diff_text = _run_git_diff(base_sha=base_sha, head_sha=head_sha, file_path=file_path)
            added_lines = _added_lines_from_diff(diff_text)
            if not added_lines:
                continue

        violations.extend(_check_nodes(tree=tree, file_path=file_path, added_lines=added_lines))

    return violations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when newly added public symbols in changed files lack docstrings.",
    )
    parser.add_argument("--base-sha", type=str, default=None)
    parser.add_argument("--head-sha", type=str, default=None)
    parser.add_argument("--files", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    """Run docstring gate and return process exit code.

    Returns:
        Zero when no violations are found; one otherwise.
    """

    args = _parse_args()
    file_paths = [Path(path) for path in args.files]
    violations = check_files(file_paths=file_paths, base_sha=args.base_sha, head_sha=args.head_sha)

    if not violations:
        print("Docstring gate passed: no missing docstrings for newly added public symbols.")
        return 0

    print("Docstring gate failed. Missing docstrings found:")
    for violation in violations:
        print(f"- {violation.file_path}:{violation.line_number} {violation.symbol_kind} {violation.symbol_name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
