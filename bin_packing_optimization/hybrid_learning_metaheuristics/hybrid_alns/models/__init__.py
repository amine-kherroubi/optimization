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


def load_bandit_state(filename: str) -> dict[str, Any]:
    """Load a pre-trained LinUCB bandit state from a pickle bundled with this
    package. Pair with ``BinPackingSolver`` by passing the returned dict as
    ``method_args["bandit_state"]``."""
    resource = files(_PACKAGE) / filename

    try:
        with resource.open("rb") as fh:
            state: dict[str, Any] = pickle.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Bandit state not found: {filename!r}. "
            "Run bandit_training.py to generate it."
        ) from None

    return state


def save_bandit_state(state: dict[str, Any], filepath: str) -> None:
    """Persist a bandit state dict to disk (e.g. the output of
    ``solver.get_bandit_state()``)."""
    import os

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "wb") as fh:
        pickle.dump(state, fh)
