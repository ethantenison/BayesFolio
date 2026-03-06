"""Audit module and public-symbol docstring coverage.

Usage:
    poetry run python scripts/audit_doc_coverage.py
    poetry run python scripts/audit_doc_coverage.py --include-tests
    poetry run python scripts/audit_doc_coverage.py --output docs/refactor/docstring_inventory.md

This script scans Python files in the repository and reports missing:
- top-level module docstrings
- docstrings on top-level public classes/functions

By default, it audits `bayesfolio/` only. Use `--include-tests` to also audit `tests/`.
"""

from __future__ import annotations

import argparse
import ast
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MissingSymbolDoc:
    """Record for a top-level public symbol missing a docstring.

    Attributes:
        file_path: Repository-relative file path.
        line: 1-based line number where symbol is defined.
        symbol_type: Symbol category (`class`, `function`, `async_function`).
        symbol_name: Public symbol name.
    """

    file_path: str
    line: int
    symbol_type: str
    symbol_name: str


@dataclass(frozen=True)
class AuditResult:
    """Docstring audit summary and detailed findings.

    Attributes:
        scanned_files: Number of Python files parsed.
        missing_module_doc_files: Relative paths missing module docstrings.
        missing_symbol_docs: Missing top-level public symbol docstring records.
    """

    scanned_files: int
    missing_module_doc_files: list[str]
    missing_symbol_docs: list[MissingSymbolDoc]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for docstring coverage audit.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="Audit module/public docstring coverage.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root path (default: current directory).",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include tests/ in coverage audit.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown output path for full report.",
    )
    return parser.parse_args()


def discover_python_files(repo_root: Path, include_tests: bool) -> list[Path]:
    """Discover Python files to audit.

    Args:
        repo_root: Repository root directory.
        include_tests: Whether to include `tests/` tree.

    Returns:
        Sorted list of candidate Python file paths.
    """

    roots = [repo_root / "bayesfolio"]
    if include_tests:
        roots.append(repo_root / "tests")

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)

    return sorted(files)


def relative_to_repo(path: Path, repo_root: Path) -> str:
    """Convert absolute path to repository-relative POSIX path.

    Args:
        path: File path to convert.
        repo_root: Repository root directory.

    Returns:
        POSIX-style relative path.
    """

    return path.relative_to(repo_root).as_posix()


def audit_file(path: Path, repo_root: Path) -> tuple[bool, list[MissingSymbolDoc]]:
    """Audit one file for module and public-symbol docstring presence.

    Args:
        path: Python file path.
        repo_root: Repository root directory.

    Returns:
        Tuple containing:
            - boolean indicating whether module docstring exists
            - list of missing top-level public symbol doc records
    """

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    module_has_doc = ast.get_docstring(tree) is not None
    missing_symbol_docs: list[MissingSymbolDoc] = []

    for node in tree.body:
        symbol_name = getattr(node, "name", None)
        if symbol_name is None or symbol_name.startswith("_"):
            continue

        symbol_type: str | None = None
        if isinstance(node, ast.ClassDef):
            symbol_type = "class"
        elif isinstance(node, ast.FunctionDef):
            symbol_type = "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            symbol_type = "async_function"

        if symbol_type is None:
            continue

        if ast.get_docstring(node) is None:
            missing_symbol_docs.append(
                MissingSymbolDoc(
                    file_path=relative_to_repo(path, repo_root),
                    line=int(node.lineno),
                    symbol_type=symbol_type,
                    symbol_name=str(symbol_name),
                )
            )

    return module_has_doc, missing_symbol_docs


def run_audit(repo_root: Path, include_tests: bool) -> AuditResult:
    """Run docstring coverage audit across selected repository trees.

    Args:
        repo_root: Repository root directory.
        include_tests: Whether to include tests tree.

    Returns:
        Aggregated audit result.
    """

    files = discover_python_files(repo_root=repo_root, include_tests=include_tests)
    missing_module_doc_files: list[str] = []
    missing_symbol_docs: list[MissingSymbolDoc] = []

    for file_path in files:
        has_module_doc, missing_symbols = audit_file(file_path, repo_root)
        if not has_module_doc:
            missing_module_doc_files.append(relative_to_repo(file_path, repo_root))
        missing_symbol_docs.extend(missing_symbols)

    return AuditResult(
        scanned_files=len(files),
        missing_module_doc_files=sorted(missing_module_doc_files),
        missing_symbol_docs=sorted(
            missing_symbol_docs,
            key=lambda item: (item.file_path, item.line, item.symbol_name),
        ),
    )


def format_markdown(result: AuditResult, include_tests: bool) -> str:
    """Render markdown report for docstring audit result.

    Args:
        result: Aggregated audit output.
        include_tests: Whether tests were included.

    Returns:
        Markdown document content.
    """

    lines: list[str] = []
    lines.append("# Docstring Coverage Audit")
    lines.append("")
    lines.append(f"- Scanned files: **{result.scanned_files}**")
    lines.append(f"- Included tests: **{include_tests}**")
    lines.append(f"- Files missing module docstrings: **{len(result.missing_module_doc_files)}**")
    lines.append(f"- Public symbols missing docstrings: **{len(result.missing_symbol_docs)}**")
    lines.append("")

    lines.append("## Missing Module Docstrings")
    lines.append("")
    if not result.missing_module_doc_files:
        lines.append("- None")
    else:
        for path in result.missing_module_doc_files:
            lines.append(f"- {path}")
    lines.append("")

    lines.append("## Missing Public Symbol Docstrings")
    lines.append("")
    if not result.missing_symbol_docs:
        lines.append("- None")
    else:
        for item in result.missing_symbol_docs:
            lines.append(f"- {item.file_path}:{item.line} — {item.symbol_type} `{item.symbol_name}`")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Execute docstring coverage audit and print summary/report output."""

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    result = run_audit(repo_root=repo_root, include_tests=bool(args.include_tests))

    print(f"Scanned files: {result.scanned_files}")
    print(f"Missing module docstrings: {len(result.missing_module_doc_files)}")
    print(f"Missing public symbol docstrings: {len(result.missing_symbol_docs)}")

    if args.output:
        output_path = (repo_root / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            format_markdown(result=result, include_tests=bool(args.include_tests)),
            encoding="utf-8",
        )
        print(f"Report written to: {output_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
