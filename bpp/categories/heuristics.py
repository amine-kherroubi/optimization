from .base import CategoryAdapter

HEURISTICS_STRATEGY = CategoryAdapter(
    "specific_heuristics",
    (
        "next fit",
        "first fit",
        "best fit",
        "worst fit",
        "next fit decreasing",
        "first fit decreasing",
        "best fit decreasing",
        "worst fit decreasing",
        "relocation",
        "swap",
    ),
    "2_specific_heuristics/solver.py",
    {"nf": "next fit", "ff": "first fit", "bf": "best fit", "wf": "worst fit", "nfd": "next fit decreasing", "ffd": "first fit decreasing", "bfd": "best fit decreasing", "wfd": "worst fit decreasing"},
)
