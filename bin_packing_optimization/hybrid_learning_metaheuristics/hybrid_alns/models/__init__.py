from __future__ import annotations

import pickle
from importlib.resources import files
from typing import Any

_PACKAGE = __name__


def load_repair_model(filename: str) -> dict[str, Any]:
    resource = files(_PACKAGE) / filename

    try:
        with resource.open("rb") as fh:
            bundle: dict[str, Any] = pickle.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model artefact not found: {filename!r}. "
            "Run the repair_model_training notebook to generate it."
        ) from None

    return bundle
