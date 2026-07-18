"""Sphinx configuration for the AsteroScale documentation."""

from importlib.metadata import version as package_version


project = "AsteroScale"
author = "Martin Nielsen"
release = package_version("AsteroScale")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_mock_imports = ["dynesty", "jax"]
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]

# Tutorial notebooks can contain nested-sampling runs. Keep hosted builds
# fast and deterministic by rendering committed outputs instead of executing
# notebooks during every documentation build.
nb_execution_mode = "off"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "AsteroScale"
html_theme_options = {
    "github_url": "https://github.com/nielsenmb/AsteroScale",
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
