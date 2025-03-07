#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# GemPy documentation build configuration file, created by
# sphinx-quickstart on Tue Jul  4 10:28:52 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys
import warnings

import pyvista
from sphinx_gallery.sorting import FileNameSortKey

import gempy

# External examples:
sys.path.insert(0, os.path.abspath('.'))
import make_external_gallery

make_external_gallery.make_example_gallery()

# region PyVista Configuration
pyvista.set_error_output_file('errors.txt')
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy. Ensure that offscreen rendering is used for docs generation
pyvista.set_plot_theme('document')  # Preferred plotting style for documentation
pyvista.FIGURE_PATH = os.path.join(os.path.abspath('_images/'), 'auto-generated/')  # Save figures in specified directory
pyvista.BUILDING_GALLERY = True
if not os.path.exists(pyvista.FIGURE_PATH):
    os.makedirs(pyvista.FIGURE_PATH)

sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.doctest',
        'sphinx.ext.intersphinx',
        'sphinx.ext.todo',
        'sphinx.ext.coverage',
        'sphinx.ext.mathjax',
        'sphinx.ext.viewcode',
        'sphinx.ext.githubpages',
        'sphinx.ext.napoleon',
        'sphinx_gallery.gen_gallery',
        "pyvista.ext.plot_directive",
        "sphinx_design",
        'sphinx_automodapi.automodapi',
        'sphinx_automodapi.smart_resolver',
]

if run_intersphinx := False:  # Example configuration for intersphinx: refer to the Python standard library.
    intersphinx_mapping = {
            'numpy'     : ('https://numpy.org/doc/stable/', None),
            'python'    : ('https://docs.python.org/{.major}'.format(sys.version_info), None),
            'matplotlib': ('https://matplotlib.org/stable/', None),
            'mayavi'    : ('http://docs.enthought.com/mayavi/mayavi', None),
            'sklearn'   : ('https://scikit-learn.org/stable', None),
            'skimage'   : ('https://scikit-image.org/docs/dev/', None),
            'pyvista'   : ('https://docs.pyvista.org/', None),
            'sphinx'    : ('https://www.sphinx-doc.org/en/master/', None),
            'pandas'    : ('https://pandas.pydata.org/pandas-docs/stable/', None),
    }

napoleon_google_docstring = True

autodoc_default_options = {
        'autodoc_default_flags': ['members'],
        'members'              : None,
        'member-order'         : 'bysource',
        'special-members'      : '__init__',
        'undoc-members'        : True,
        'exclude-members'      : '__weakref__'
}
autosummary_generate = True
autosummary_imported_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#

source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# region General information about the project.
project = 'GemPy'
year = datetime.date.today().year
copyright = u'2017-{}, Gempy Developers'.format(year)

with open(os.path.join(os.path.dirname(__file__), '../../AUTHORS.rst'), 'r') as f:
    author = f.read()

version = gempy.__version__
release = gempy.__version__  # The full version, including alpha/beta/rc tags.
today_fmt = '%d %B %Y'
language = 'en'

# endregion

# region building the documentation configuration
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store', 'errors.txt', '../test']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
highlight_language = 'python3'

todo_include_todos = True  # If true, `todo` and `todoList` produce output, else they produce nothing.

linkcheck_retries = 3
linkcheck_timeout = 500

# * -- Sphinx Gallery Options
sphinx_gallery_conf = {
        # path to your examples scripts
        "examples_dirs"          : [
                "../../examples/tutorials",
                "../../examples/examples",
                # "../../examples/integrations",
        ],
        # path where to save gallery generated examples
        "gallery_dirs"           : [
                'tutorials',
                "examples",
                # 'integrations'
        ],
        "filename_pattern"       : r"\.py",
 #       "ignore_pattern": r"(__init__\.py|ch1_7_3d_visualization\.py|mik\.py)$",  # Exclude specific files: uncomment for faster build as long as examples are broken
        "download_all_examples"  : False,  # Remove the "Download all examples" button from the top level gallery
        "within_subsection_order": FileNameSortKey,  # Sort gallery example by file name instead of number of lines (default)
        "backreferences_dir"     : 'gen_modules/backreferences',  # directory where function granular galleries are stored
        "doc_module"             : ("gempy", "gempy_viewer", 'numpy', 'pandas', 'matplotlib'),  # Modules for which function level galleries are created.  In
        "image_scrapers"         : ('pyvista', 'matplotlib'),
        'first_notebook_cell'    : ("%matplotlib inline\n"
                                    "from pyvista import set_plot_theme\n"
                                    "set_plot_theme('document')"),
        'reference_url'          : {
                'gempy': None,  # The module you locally document uses None
                'numpy': 'https://numpy.org/doc/stable/'

        },
}

# endregion

# region -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'alabaster'
html_theme_options = {
        'github_user'     : 'gempy-project',
        'github_repo'     : 'gempy',
        'github_type'     : 'star',
        'logo'            : 'logos/gempy.png',
        'logo_name'       : False,
        'travis_button'   : False,
        'page_width'      : '1200px',
        'fixed_sidebar'   : False,
        'show_related'    : True,
        'sidebar_collapse': True,
}

# Custom sidebar templates, maps document names to template names.
# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}

html_sidebars = {'**': ['about.html', 'navigation.html',
                        'relations.html',
                        'searchbox.html',
                        'donate.html', ]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = '_static/logos/favicon.ico'

# endregion

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'GemPydoc'

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'gempy', 'GemPy Documentation', [author], 1)]

# region other options
# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')
# endregion
