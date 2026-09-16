"""Sphinx configuration for the Symmetrix-XL documentation."""

from datetime import date

project = "Symmetrix-XL"
copyright = f"{date.today().year}, Symmetrix-XL contributors"
author = "Symmetrix-XL contributors"
release = "0.1.0"

extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = [
    "_build",
    "automatic_build_and_packaging_plan.md",
]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "Symmetrix-XL"
html_static_path = []
