"""Reusable package for the PE static-analysis coursework.

Package layout:
- ``core``: shared metrics and small cross-cutting utilities.
- ``features``: EMBER-compatible PE feature extraction and feature analysis.
- ``prediction``: single-file static PE scoring.
- ``training``: SGD and LightGBM training jobs.
- ``gold``: external gold-set verification, deduplication, and inference.
- ``sorel``: SOREL-20M manifests, hash export, and LMDB feature access.
- ``gui``: Tk desktop checker.
"""

__all__ = [
    "core",
    "features",
    "gold",
    "gui",
    "prediction",
    "sorel",
    "training",
]
