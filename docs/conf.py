import simppler

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "simppler"
copyright = "2025, Thomas Vandal"
author = "Thomas Vandal"
release = simppler.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for myst-nb -----------------------------------------------------
# Execute manually to render \r properly
nb_execution_mode = "off"
nb_execution_timeout = -1


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "simppler"
html_theme_options = {
    "repository_url": "https://github.com/vandalt/simppler",
    "use_repository_button": True,
}
html_static_path = ["_static"]

# -- Options for autodoc -----------------------------------------------------
autodoc_typehints = "description"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "numpy.typing.ArrayLike": "ArrayLike",
}
