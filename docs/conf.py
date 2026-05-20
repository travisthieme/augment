"""Sphinx configuration for augment docs."""
from __future__ import annotations

from importlib import metadata

project = "augment"
author = "Travis J. Thieme"

try:
    release = metadata.version("augment")
except metadata.PackageNotFoundError:
    release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

myst_enable_extensions = ["colon_fence", "deflist"]

# Keep type hints readable in the rendered docs.
autodoc_typehints = "description"
