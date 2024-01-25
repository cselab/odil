import os, sys
import importlib.util


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


runtime = lazy_import("odil.runtime")
plot = lazy_import("odil.plot")

from .io import (
    parse_raw_xmf,
    read_raw,
    read_raw_with_xmf,
    write_raw_xmf,
    write_raw_with_xmf,
    write_vtk_poly,
)
from .backend import ModBase, ModNumpy, ModTensorflow
from .history import History
from .util import (
    setup_outdir,
    optimize,
    make_callback,
    printlog,
    set_log_file,
)
from .core import (
    Domain,
    State,
    Problem,
    Field,
    MultigridField,
    NeuralNet,
    Array,
    restrict_to_coarser,
)
from . import linsolver
from . import core_min
from . import core
from .optimizer import EarlyStopError

del os, sys, importlib
