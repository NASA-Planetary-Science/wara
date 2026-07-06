project = "wara"
copyright = "2025, Mauricio Ayllon Unzueta"
author = "Mauricio Ayllon Unzueta"

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
]

autoapi_dirs = ["../wara"]
autoapi_ignore = ["*/version.py", "*/gui_legacy/*", "*/mplwidget*"]
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../figs/wara-logo.png"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
