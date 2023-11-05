import sphinx_gallery
import sphinx_gallery.gen_rst
import sphinx_gallery.utils
import os
import re
import codecs
import stat


def save_rst_example(example_rst, example_file, time_elapsed, memory_used,
                     gallery_conf):
    example_fname = os.path.relpath(example_file, gallery_conf["src_dir"])
    ipynb_fname = re.sub("\.py$", "", example_fname) + ".ipynb"
    ref_fname = example_fname.replace(os.path.sep, "_")
    example_rst = (EXAMPLE_HEADER.format(example_fname, ref_fname) +
                   example_rst)
    fname = os.path.basename(example_file)
    example_rst += CODE_DOWNLOAD.format(
        fname, sphinx_gallery.utils.replace_py_ipynb(fname), ref_fname,
        ipynb_fname)
    write_file_new = re.sub(r"\.py$", ".rst.new", example_file)
    with codecs.open(write_file_new, "w", encoding="utf-8") as f:
        f.write(example_rst)
    sphinx_gallery.utils._replace_md5(write_file_new, mode="t")


extensions = [
    "sphinx_gallery.gen_gallery",
]
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'show_signature': False,
}
html_theme = 'theme'
html_theme_path = ['.']
pygments_style = 'sphinx'
EXAMPLE_HEADER = """
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "{0}"
.. LINE NUMBERS ARE GIVEN BELOW.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_{1}:

"""

CODE_DOWNLOAD = """
.. _sphx_glr_download_{2}:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: {0} <{0}>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: {1} <{1}>`

.. raw:: html

   <a href="https://colab.research.google.com/github/cselab/odil/blob/gh-pages/doc/{3}">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
   </a>
"""

sphinx_gallery.gen_rst.save_rst_example = save_rst_example
