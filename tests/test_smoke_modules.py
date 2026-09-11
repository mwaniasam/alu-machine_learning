"""Smoke tests for selected educational utility modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str) -> ModuleType:
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summation_i_squared_smoke() -> None:
    module = _load_module("math/calculus/9-sum_total.py")
    assert module.summation_i_squared(5) == 55
    assert module.summation_i_squared(0) is None


def test_matrix_shape_smoke() -> None:
    module = _load_module("math/linear_algebra/2-size_me_please.py")
    assert module.matrix_shape([[1, 2], [3, 4]]) == [2, 2]


def test_determinant_smoke() -> None:
    module = _load_module("math/advanced_linear_algebra/0-determinant.py")
    assert module.determinant([[1, 2], [3, 4]]) == -2
