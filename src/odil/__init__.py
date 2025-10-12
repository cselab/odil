# ruff: noqa: F401

from . import (
    core,
    core_min,
    linsolver,
)
from .backend import (
    ModBase,
    ModNumpy,
    ModTensorflow,
)
from .core import (
    Array,
    Domain,
    Field,
    MultigridField,
    NeuralNet,
    Problem,
    State,
    restrict_to_coarser,
)
from .history import (
    History,
)
from .io import (
    parse_raw_xmf,
    read_raw,
    read_raw_with_xmf,
    write_raw_with_xmf,
    write_raw_xmf,
    write_vtk_poly,
)
from .optimizer import (
    EarlyStopError,
)
from .util import (
    make_callback,
    optimize,
    printlog,
    set_log_file,
    setup_outdir,
)


def lazy_import(name):
    import importlib.util
    import os
    import sys

    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


runtime = lazy_import("odil.runtime")
plot = lazy_import("odil.plot")
