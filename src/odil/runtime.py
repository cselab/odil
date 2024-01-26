import os
import sys
import numpy as np

from .backend import ModNumpy, ModTensorflow

if not int(os.environ.get("ODIL_MT", 0)):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=1")
    os.environ["TENSORFLOW_INTER_OP_PARALLELISM"] = "1"
    os.environ["TENSORFLOW_INTRA_OP_PARALLELISM"] = "1"

enable_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ["", "-1"]
if not enable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if not int(os.environ.get("ODIL_WARN", 0)):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

enable_jit = bool(int(os.environ.get("ODIL_JIT", 0)))

backend_name = os.environ.get("ODIL_BACKEND", '')

if not backend_name:
    try:
        import tensorflow as tf
        backend_name = 'tf'
    except:
        pass

if not backend_name:
    try:
        import jax
        backend_name = 'jax'
    except:
        sys.stderr.write(
            f"Cannot select a default backend. Tried: tensorflow, jax\n")
        exit(1)

if backend_name == 'tf':
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    _gpus = tf.config.list_physical_devices('GPU')
    if _gpus:
        try:
            for gpu in _gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)

    jax = None
    mod = ModTensorflow(tf)
elif backend_name == 'jax':
    tf = None
    import jax
    if not enable_gpu:
        jax.config.update('jax_platform_name', 'cpu')
    # Enable support for float64 operations.
    jax.config.update("jax_enable_x64", True)

    mod = ModNumpy(jax.numpy, jax=jax)
else:
    sys.stderr.write(
        f"Unknown ODIL_BACKEND='{backend_name}', options are: tf, jax\n")
    exit(1)

# Default data type.
dtype_name = os.environ.get("ODIL_DTYPE", 'float32')
if dtype_name in ['float32', 'float64']:
    dtype = np.dtype(dtype_name)
else:
    sys.stderr.write(
        f"Expected ODIL_DTYPE=float32 or float64, got '{dtype}' \n")
    exit(1)

del os
del np
del sys
