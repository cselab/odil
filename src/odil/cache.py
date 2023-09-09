import os
import functools


def cache_to_file(targetbase, update=False, name_with_arg0=False):
    """
    Factory for a decorator that caches the result of
    function and stores it to a target file.

    targetbase: base path to cache file
    update: force cache update
    name_with_arg0: append cache name by first argument converted to string

    Example: Creates file "_cache_7.pickle" with `int(7)`.

    @cache_to_file("_cache.pickle", name_with_arg0=False)
    def F(a):
        return a
    F(7)
    """

    ext = os.path.splitext(targetbase)[1]
    if ext == '.pickle':
        import pickle

        def load(path):
            with open(path, 'rb') as f:
                print("Loading cache '{}'".format(path))
                return pickle.load(f)

        def save(content, path):
            with open(path, 'wb') as f:
                print("Saving cache '{}'".format(path))
                pickle.dump(content, f, protocol=4)
    elif ext == '.json':
        import json

        def load(path):
            with open(path, 'r') as f:
                print("Loading cache '{}'".format(path))
                return json.load(f)

        def save(content, path):
            with open(path, 'w') as f:
                print("Saving cache '{}'".format(path))
                json.dump(content, f)

    elif ext == '.npy':
        import numpy

        def load(path):
            print("Loading cache '{}'".format(path))
            return numpy.load(path, allow_pickle=True).item()

        def save(content, path):
            print("Saving cache '{}'".format(path))
            numpy.save(path, content, allow_pickle=True)
    else:
        raise ValueError(
            "Unrecognized extension '{}', expected .pickle.".format(ext))

    def decorator(func):

        def inner(*args, **kwargs):
            name, ext = os.path.splitext(targetbase)
            if len(args) and name_with_arg0:
                name += '_{:}'.format(args[0])
            target = name + ext
            if not update and os.path.isfile(target):
                return load(target)
            result = func(*args, **kwargs)
            d = os.path.dirname(target)
            if d: os.makedirs(d, exist_ok=True)
            save(result, target)
            return result

        return functools.wraps(func)(inner)

    return decorator
