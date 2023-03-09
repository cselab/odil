#!/usr/bin/env python3

import os
import sys


def get_path(ext, suff=''):
    return os.path.splitext(os.path.basename(
        sys.argv[0]))[0] + suff + '.' + ext


def savefig(fig, suff='', ext='svg', path=None, **kwargs):
    if path is None:
        path = get_path(ext, suff)
    elif ext is not None:
        path = os.path.splitext(path)[0] + '.' + ext
    print(path)
    metadata = {
        'Date': None
    } if ext == 'svg' else {
        'DateModified': None
    } if ext == 'pdf' else {}
    fig.savefig(path, metadata=metadata, **kwargs)
