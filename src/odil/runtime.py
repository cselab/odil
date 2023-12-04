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

usegpu = os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ["", "-1"]
if not usegpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if not int(os.environ.get("ODIL_WARN", 0)):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

backend = os.environ.get("ODIL_BACKEND", 'tf')
jit = bool(int(os.environ.get("ODIL_JIT", 0)))
if backend == 'tf':
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)

    jax = None
    mod = ModTensorflow(tf)
elif backend == 'jax':
    tf = None
    import jax
    if not usegpu:
        jax.config.update('jax_platform_name', 'cpu')
    # Enable support for float64 operations.
    jax.config.update("jax_enable_x64", True)

    mod = ModNumpy(jax.numpy, jax=jax)
else:
    sys.stderr.write(
        f"Unknown ODIL_BACKEND='{backend}', options are: tf, jax\n")
    exit(1)

# Default data type.
dtype_name = os.environ.get("ODIL_DTYPE", 'float32')
if dtype_name in ['float32', 'float64']:
    dtype = np.dtype(dtype_name)
else:
    sys.stderr.write(
        f"Expected ODIL_DTYPE=float32 or float64, got '{dtype}' \n")
    exit(1)
