"""pytest conftest: mock sklearn/scipy before collection.

The repair_model_training package imports sklearn at module level, but the
installed sklearn was compiled against NumPy 1.x while the system has NumPy 2.x.
Pre-populating sys.modules with fake packages prevents the native extension
from loading. The solver only uses sklearn at *runtime* (model.predict_proba),
never at import time, so these mocks are sufficient for unit tests.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


class _AutoMockModule(types.ModuleType):
    """A module that returns a MagicMock for any attribute that isn't set,
    and acts as a package (has __path__) so sub-imports don't crash."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []       # marks it as a package
        self.__package__ = name
        self.__spec__ = None
        self._mocks: dict = {}

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name not in self._mocks:
            self._mocks[name] = MagicMock(name=f"{self.__name__}.{name}")
        return self._mocks[name]


def _register(name: str) -> None:
    """Register a fake package and all its parent packages in sys.modules."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        full = ".".join(parts[:i])
        if full not in sys.modules:
            sys.modules[full] = _AutoMockModule(full)


_FAKE_PACKAGES = [
    "sklearn",
    "sklearn.ensemble",
    "sklearn.preprocessing",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.utils",
    "sklearn.utils.class_weight",
    "sklearn.pipeline",
    "sklearn.base",
    "scipy",
    "scipy.sparse",
    "joblib",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.figure",
    "matplotlib.axes",
]

for _pkg in _FAKE_PACKAGES:
    _register(_pkg)
