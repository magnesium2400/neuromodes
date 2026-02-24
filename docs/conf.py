# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "neuromodes"
copyright = "2025, Monash University"
author = "Neural Systems and Behaviour Lab"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx.ext.mathjax"
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "tags": "ams",
        "useLabelIds": True,
    }
}

nb_execution_mode = 'off'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_logo = '_static/logo.png'
html_theme_options = {'logo_only': True}
