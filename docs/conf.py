import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "deduplicate_lib"
copyright = "2026, Julian Holland"
author = "Julian Holland"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]

html_theme = "furo"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "description"
