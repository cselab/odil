import sphinx_gallery

extensions = [
    "sphinx_gallery.gen_gallery",
]
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'show_signature': False,
}
