"""Enforce the currently clean BayesFolio package import boundaries.

These architecture tests inspect Python source without importing application
modules. They protect the contracts, IO, and core dependency directions and
keep the refactored monthly orchestrator free of direct runtime side effects,
while broader legacy engine violations remain outside this enforcement slice.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "bayesfolio"


def _is_package_or_child(module_name: str, package_name: str) -> bool:
    return module_name == package_name or module_name.startswith(f"{package_name}.")


def _source_package_parts(source_path: Path) -> tuple[str, ...]:
    relative = source_path.relative_to(REPOSITORY_ROOT).with_suffix("")
    return relative.parts[:-1]


def _resolve_from_import(node: ast.ImportFrom, source_path: Path) -> str:
    if node.level == 0:
        return node.module or ""

    package_parts = _source_package_parts(source_path)
    parent_levels = node.level - 1
    if parent_levels > len(package_parts):
        return node.module or ""

    base_parts = package_parts[: len(package_parts) - parent_levels]
    if node.module:
        base_parts = (*base_parts, *node.module.split("."))
    return ".".join(base_parts)


def _iter_import_targets(source_path: Path) -> Iterator[tuple[int, str]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        base_module = _resolve_from_import(node, source_path)
        for alias in node.names:
            if alias.name == "*":
                yield node.lineno, base_module
            elif base_module:
                yield node.lineno, f"{base_module}.{alias.name}"
            else:
                yield node.lineno, alias.name


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _boundary_violation(source_layer: str, imported_module: str) -> str | None:
    if source_layer == "contracts":
        for forbidden in ("bayesfolio.engine", "bayesfolio.io"):
            if _is_package_or_child(imported_module, forbidden):
                return f"contracts cannot import {forbidden}"

    if source_layer == "io" and _is_package_or_child(imported_module, "bayesfolio.engine"):
        return "io cannot import bayesfolio.engine"

    if source_layer == "core" and _is_package_or_child(imported_module, "bayesfolio"):
        if not _is_package_or_child(imported_module, "bayesfolio.core"):
            return "core cannot import other bayesfolio packages"

    return None


def test_clean_package_import_boundaries() -> None:
    """Reject imports that invert the clean contracts, IO, or core layers."""

    violations: list[str] = []
    for source_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_path = source_path.relative_to(PACKAGE_ROOT)
        source_layer = relative_path.parts[0]
        if source_layer not in {"contracts", "core", "io"}:
            continue

        for line_number, imported_module in _iter_import_targets(source_path):
            reason = _boundary_violation(source_layer, imported_module)
            if reason is not None:
                violations.append(
                    f"{source_path.relative_to(REPOSITORY_ROOT)}:{line_number}: {reason} ({imported_module})"
                )

    assert not violations, "Import boundary violations:\n" + "\n".join(sorted(violations))


def test_monthly_orchestrator_has_no_direct_runtime_io() -> None:
    """Keep monthly engine sequencing behind its injected runtime port."""

    source_path = PACKAGE_ROOT / "engine" / "automation" / "monthly_portfolio.py"
    forbidden_imports = {"json", "mlflow", "pathlib", "subprocess"}
    forbidden_calls = {"check_output", "mkdir", "read_bytes", "read_parquet", "to_csv", "write_text"}
    violations: list[str] = []

    for line_number, imported_module in _iter_import_targets(source_path):
        imported_root = imported_module.split(".", maxsplit=1)[0]
        if imported_root in forbidden_imports or _is_package_or_child(imported_module, "bayesfolio.io"):
            violations.append(f"{source_path.relative_to(REPOSITORY_ROOT)}:{line_number}: {imported_module}")

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (name := _call_name(node)) in forbidden_calls:
            violations.append(f"{source_path.relative_to(REPOSITORY_ROOT)}:{node.lineno}: {name}()")

    assert not violations, "Direct monthly runtime I/O:\n" + "\n".join(sorted(violations))
