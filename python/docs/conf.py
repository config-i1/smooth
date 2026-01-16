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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
