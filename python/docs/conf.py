# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Point Sphinx to the source code directory
sys.path.insert(0, os.path.abspath('../src'))

project = 'smooth'
copyright = '2026, Filotas Theodosiou, Ivan Svetunkov, Leonidas Tsaprounis, Claude AI'
author = 'Filotas Theodosiou, Ivan Svetunkov, Leonidas Tsaprounis, Claude AI'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # understands NumPy / Google style
    "sphinx_autodoc_typehints",
]
html_theme = "sphinx_rtd_theme"
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Autodoc configuration
autodoc_default_options = {
    'members': False,
    'member-order': 'bysource',
    'undoc-members': False,
    'exclude-members': '__weakref__, __dict__, __module__, __init__, __str__, __repr__',
}

# Only include class docstring, not __init__
autoclass_content = 'class'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
