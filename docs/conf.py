"""Sphinx configuration for the Symmetrix documentation."""

from datetime import date

project = "Symmetrix"
copyright = f"{date.today().year}, Symmetrix contributors"
author = "Symmetrix contributors"
release = "0.1.0"

extensions = ["myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = [
    "_build",
    "automatic_build_and_packaging_plan.md",
    "mh1_pair_spline_implementation_plan.md",
    "mh1_rtc_portability_implementation_plan.md",
    "mh1_spline_gpu_direct_implementation_plan.md",
    "mh1_staged_direct_implementation_plan.md",
]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "Symmetrix"
html_static_path = []
