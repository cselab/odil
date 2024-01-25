import sys
import os
import json
import argparse
import psutil
import time
import numpy as np

from .optimizer import make_optimizer, Optimizer
from .history import History
g_log_file = sys.stderr  # File used by printlog()
g_log_echo = False  # True if printlog() should print to stderr.


def assert_equal(first, second, msg=None):
    if not (first == second):
        raise ValueError("Expected equal '{:}' and '{:}'{}".format(
            first, second, msg))


def set_log_file(f=None, echo=None):
    global g_log_file, g_log_echo
    if f is not None:
        g_log_file = f
    if echo is not None:
        g_log_echo = echo


def printlog(*msg):
    m = ' '.join(map(str, msg)) + '\n'
    if g_log_echo and g_log_file != sys.stderr:
        sys.stderr.write(m)
        sys.stderr.flush()
    g_log_file.write(m)
    g_log_file.flush()


class Timer():

    def __init__(self):
        self._starts = []  # Stack of pairs (key, time).
        self.counters = dict()  # Key to time.

    def push(self, key=None):
        self._starts.append((key, time.time()))

    def pop(self, key=None):
        start = self._starts.pop()
        assert start[0] is None or key is None or start[0] == key, \
                "Inconsistent keys passed to push() and pop(): "\
                "{:} and {:}".format(start[0], key)
        if key is None:
            key = start[0]
        dt = time.time() - start[1]
        self.counters[key] = self.counters.get(key, 0.) + dt

    def append(self, timer):
        for k in timer.counters:
            self.counters[k] = self.counters.get(k, 0.) + timer.counters[k]


def get_error(u, v):
    e1 = np.mean(abs(u - v))
    e2 = np.mean((u - v)**2)**0.5
    einf = np.max(abs(u - v))
    return e1, e2, einf


def add_arguments(parser):
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        help="Maximum epochs, "
                        "defaults to product of plot_every and frames")
    parser.add_argument('--every_factor',
                        type=float,
                        default=1,
                        help="Multiplier for all *_every options")
    parser.add_argument('--plot_every',
                        type=int,
                        default=5,
                        help="Epochs between plots")
    parser.add_argument('--report_every',
                        type=int,
                        default=10,
                        help="Epochs between reports to stdout")
    parser.add_argument('--history_every',
                        type=int,
                        default=1,
                        help="Epochs between entries of training history")
    parser.add_argument('--checkpoint_every',
                        type=int,
                        default=0,
                        help="Epochs between checkpoints")
    parser.add_argument('--frames',
                        type=int,
                        default=10,
                        help="Frames to plot. Zero disables first frame.")
    parser.add_argument('--outdir',
                        type=str,
                        default='.',
                        help='Output directory')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adamn',
                        help="Optimizer")
    parser.add_argument('--seed',
                        default=1000,
                        type=int,
                        help="Seed for numpy.random and tensorflow.random")
    parser.add_argument('--plot_title',
                        type=int,
                        default=0,
                        help="Enable title in plots")
    parser.add_argument('--plotext',
                        type=str,
                        default='pdf',
                        help="Extension of plots")
    parser.add_argument('--history_full',
                        type=int,
                        default=0,
                        help="Number of epochs to write "
                        "history at every point")
    parser.add_argument('--montage',
                        type=int,
                        default=1,
                        help="Run montage after plotting")
    parser.add_argument('--double',
                        type=int,
                        default=None,
                        help="Double precision. Defaults to runtime.dtype")
    parser.add_argument('--echo',
                        type=int,
                        default=0,
                        help="Echo log to stderr")
    parser.add_argument('--epoch_start',
                        type=int,
                        default=0,
                        help="Initial value of epoch")
    parser.add_argument('--frame_start',
                        type=int,
                        default=0,
                        help="Initial value of frame")
    parser.add_argument('--checkpoint',
                        type=str,
                        help="Continue from checkpoint in state_*.pickle")
    parser.add_argument('--checkpoint_train',
                        type=str,
                        help="Continue from history in state_*_train.pickle"
                        ". By default, infers the name from --checkpoint"
                        ". Set to '' to disable default behavior")
    parser.add_argument('--callback_update_state',
                        type=int,
                        default=0,
                        help="Update state after callback")
    parser.add_argument('--bfgs_m',
                        type=int,
                        default=50,
                        help="History size for L-BFGS")
    parser.add_argument('--bfgs_maxls',
                        type=int,
                        default=50,
                        help="Max evaluations in line search")
    parser.add_argument('--bfgs_pgtol',
                        type=float,
                        default=None,
                        help="Convergence tolerance for L-BFGS-B")
    parser.add_argument('--adam_epsilon',
                        type=float,
                        help="Parameter epsilon in Adam")
    parser.add_argument('--adam_beta_1',
                        type=float,
                        help="Parameter beta_1 in Adam")
    parser.add_argument('--adam_beta_2',
                        type=float,
                        help="Parameter beta_2 in Adam")
    parser.add_argument('--multigrid',
                        type=int,
                        default=0,
                        help="Use multigrid decomposition")
    parser.add_argument('--mg_interp',
                        type=str,
                        default='stack',
                        choices=[
                            'conv',
                            'stack',
                        ],
                        help="Multigrid interpolation method")
    parser.add_argument('--dump_data',
                        type=int,
                        default=1,
                        help="Dump data_*.pickle with every plot")
    parser.add_argument('--jac_nsmp0',
                        type=int,
                        default=50,
                        help="Number of samples "
                        "for initialization of Jacobi optimizer")
    parser.add_argument('--jac_nsmp1',
                        type=int,
                        default=1,
                        help="Number of samples "
                        "for each step of Jacobi optimizer")
    parser.add_argument('--jac_factor',
                        type=float,
                        default=1,
                        help="Factor for the diagonal update"
                        "for each step of Jacobi optimizer. "
                        "Increase above 1 for more weight to recent values")
    parser.add_argument('--jac_epsilon',
                        type=float,
                        default=1e-8,
                        help="Parameter epsilon in Jacobi optimizer. "
                        "Added to the diagonal to avoid division by zero")
    parser.add_argument('--nn_initializer',
                        type=str,
                        default='legacy',
                        choices=['legacy', 'glorot', 'lecun', 'he'],
                        help="Initializer for weights of neural networks")


def optimize_newton(args, problem, state, callback=None, **kwargs):
    domain = problem.domain
    mod = domain.mod

    def eval_pinfo(state):
        loss, _, terms, names, norms = problem.eval_loss_grad(state)
        pinfo = {'terms': terms, 'names': names, 'norms': norms, 'loss': loss}
        return pinfo

    from .linsolver import solve
    opt = Optimizer(name='newton', displayname='Newton')
    printlog("Running {} optimizer".format(opt.displayname))

    # Compute loss and residuals with initial state, to be used by callback.
    pinfo = eval_pinfo(state)
    if callback:
        callback(state, args.epoch_start, pinfo)

    for epoch in range(args.epoch_start, args.epochs):
        vector, matrix = problem.linearize(state)
        opt.evals += 1
        linstatus = dict()
        delta = solve(matrix, -vector, args, linstatus, args.linsolver)
        if args.linsolver_verbose:
            printlog(linstatus)
        packed = domain.pack_state(state)
        domain.unpack_state(packed + delta, state)
        if callback:
            pinfo = eval_pinfo(state)
            pinfo['linsolver'] = linstatus
            callback(state, epoch + 1, pinfo)
    arrays = domain.arrays_from_state(state)
    optinfo = argparse.Namespace()
    optinfo.epochs = args.epochs
    optinfo.evals = args.epochs
    return arrays, optinfo


def optimize_grad(args, optname, problem, state, callback=None, **kwargs):
    domain = problem.domain
    mod = domain.mod

    def loss_grad(arrays):
        domain.arrays_to_state(arrays, state)
        loss, grads, terms, names, norms = problem.eval_loss_grad(state)
        pinfo = {'terms': terms, 'names': names, 'norms': norms, 'loss': loss}
        return loss, grads, pinfo

    def callback_wrap(arrays, epoch, pinfo):
        domain.arrays_to_state(arrays, state)
        callback(state, epoch, pinfo)
        if args.callback_update_state:
            new = domain.arrays_from_state(state)
            for i in range(len(new)):
                arrays[i] = new[i]

    # Custom parameters.
    if args.bfgs_m is not None:
        kwargs['m'] = args.bfgs_m
    if args.bfgs_pgtol is not None:
        kwargs['pgtol'] = args.bfgs_pgtol
    if args.bfgs_maxls is not None:
        kwargs['maxls'] = args.bfgs_maxls
    if args.adam_epsilon is not None:
        kwargs['epsilon'] = args.adam_epsilon
    if args.adam_beta_1 is not None:
        kwargs['beta_1'] = args.adam_beta_1
    if args.adam_beta_2 is not None:
        kwargs['beta_2'] = args.adam_beta_2

    opt = make_optimizer(optname, dtype=domain.dtype, mod=mod, **kwargs)
    printlog("Running {} optimizer".format(opt.displayname))

    # Compute loss and residuals with initial state, to be used by callback.
    arrays = domain.arrays_from_state(state)
    _, _, pinfo = loss_grad(arrays)
    if callback:
        callback(state, args.epoch_start, pinfo)

    arrays, optinfo = opt.run(arrays,
                              loss_grad=loss_grad,
                              epochs=args.epochs - args.epoch_start,
                              callback=callback_wrap if callback else None,
                              epoch_start=args.epoch_start,
                              lr=args.lr,
                              **kwargs)
    return arrays, optinfo


def optimize(args, optname, problem, state, callback, **kwargs):
    if optname == 'newton':
        return optimize_newton(args, problem, state, callback, **kwargs)
    return optimize_grad(args, optname, problem, state, callback, **kwargs)


def get_memory_usage_kb():
    '''
    Returns current memory usage in KiB.
    '''
    process = psutil.Process()
    return process.memory_info().rss // 1024


def get_env_config():
    keys = [
        'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'ODIL_WARN', 'ODIL_BACKEND',
        'ODIL_JIT', 'ODIL_MT', 'ODIL_DTYPE'
    ]
    return {k: os.environ.get(k, '') for k in keys}


def setup_outdir(args, relpath_args=None):
    '''
    Creates the output directory, configuration `args.json`, and log file `train.log`.
    Updates the arguments with new relative paths and number of epochs.
    Sets random seeds.

    args: `argparse.Namespace`
        Arguments produced by `argparse.ArgumentParser.parse_args()`.
    relpath_args: `list` of `str`
        List of names of attributes of `args` to be treated as relative paths.
        They are updated to paths relative to the output directory.
    '''
    from . import runtime
    mod = runtime.mod
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'args.json'), 'w') as f:
        d = dict(
            vars(args),
            **get_env_config(),
            runtime_backend=runtime.backend,
            runtime_dtype=runtime.dtype_name,
            runtime_jit=runtime.jit,
        )
        json.dump(d, f, sort_keys=True, indent=4)

    # Switch to output directory.
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    set_log_file(open("train.log", 'w'), echo=args.echo)

    # Update relative paths.
    if relpath_args is None:
        relpath_args = []
    for k in relpath_args:
        if getattr(args, k):
            setattr(args, k, os.path.relpath(getattr(args, k), start=outdir))

    # Update arguments.
    mulint = lambda v, k: None if v is None else max(1, round(v * k))
    args.plot_every = mulint(args.plot_every, args.every_factor)
    args.history_every = mulint(args.history_every, args.every_factor)
    args.report_every = mulint(args.report_every, args.every_factor)
    if args.epochs is None:
        args.epochs = args.frames * args.plot_every

    if args.seed is not None:
        np.random.seed(args.seed)
        mod.random.set_seed(args.seed)
    printlog(' '.join(sys.argv))


def make_callback(problem,
                  args=None,
                  epoch_func=None,
                  report_func=None,
                  history_func=None,
                  checkpoint_func=None,
                  plot_func=None):
    # Persistent state of the callback.
    cbinfo = argparse.Namespace()
    cbinfo.walltime = 0  # Walltime of the last call with task_report=True.
    cbinfo.epoch = 0  # Epoch of the last call with task_report=True.
    cbinfo.time_callback = 0  # Total time spent in callback.
    cbinfo.time_start = time.time()
    cbinfo.problem = problem
    cbinfo.args = args
    cbinfo.frame = 0

    if args.history_every:
        cbinfo.history = History(csvpath='train.csv', warmup=1)
    else:
        cbinfo.history = None

    def callback(state, epoch, pinfo):
        problem = cbinfo.problem
        domain = problem.domain
        args = cbinfo.args
        history = cbinfo.history
        time_prev = time.time()

        cbinfo.task_report = (args.report_every
                              and epoch % args.report_every == 0)
        cbinfo.task_history = (history is not None
                               and (epoch % args.history_every == 0
                                    or epoch < args.history_full))
        cbinfo.task_plot = (epoch % args.plot_every == 0
                            and (epoch or args.frames))
        cbinfo.task_checkpoint = (args.checkpoint_every
                                  and epoch % args.checkpoint_every == 0)

        cbinfo.pinfo = pinfo

        if isinstance(problem.tracers, dict):
            problem.tracers['epoch'] = epoch
        if epoch_func is not None:
            epoch_func(problem, state, epoch, cbinfo)

        # Subtract first part of callback time.
        curtime = time.time()
        cbinfo.time_callback += curtime - time_prev
        time_prev = curtime
        walltime = curtime - cbinfo.time_start - cbinfo.time_callback

        if cbinfo.task_report:
            memusage = get_memory_usage_kb()
            printlog("\nepoch={:05d}".format(epoch))
            if pinfo and 'norms' in pinfo:
                norms, names = pinfo['norms'], pinfo['names']
                printlog('residual: ' + ', '.join(
                    '{}:{:.5g}'.format(name or str(i), np.array(norm))
                    for i, (norm, name) in enumerate(zip(norms, names))))
            if report_func is not None:
                report_func(problem, state, epoch, cbinfo)
            printlog("memory: {:} MiB".format(memusage // 1024))
            printlog("walltime: {:.3f} s".format(walltime))
            printlog("walltime+callback: {:.3f} s".format(  #
                walltime + cbinfo.time_callback))
            if epoch > cbinfo.epoch:
                wte = (walltime - cbinfo.walltime) / (epoch - cbinfo.epoch)
                printlog("walltime/epoch: {:.3f} ms".format(wte * 1000))
                printlog("throughput: {:.3f}M cells/s".format(
                    np.prod(domain.cshape) / wte / 1e6))
                cbinfo.walltime = walltime
                cbinfo.epoch = epoch

        if cbinfo.task_history:
            memusage = get_memory_usage_kb()
            history.append('epoch', epoch)
            history.append('frame', cbinfo.frame)
            if pinfo and 'norms' in pinfo:
                norms, names = pinfo['norms'], pinfo['names']
                for i, (norm, name) in enumerate(zip(norms, names)):
                    history.append('norm_{:}'.format(name or str(i)),
                                   np.array(norm))
            if pinfo and 'loss' in pinfo:
                history.append('loss', pinfo['loss'])
            if args.linsolver_history and 'linsolver' in pinfo:
                for key, val in pinfo['linsolver'].items():
                    if isinstance(val, (int, float, str, np.floating)):
                        history.append('lin_' + key, val)
            history.append('walltime', walltime)
            history.append('memory', memusage / 1024)
            if history_func is not None:
                history_func(problem, state, epoch, history, cbinfo)
            history.write()

        if cbinfo.task_plot:
            if plot_func is not None:
                plot_func(problem, state, epoch, cbinfo.frame, cbinfo)
            cbinfo.frame += 1

        if cbinfo.task_checkpoint:
            if checkpoint_func is not None:
                checkpoint_func(problem, state, epoch, cbinfo)
            else:
                from .core import checkpoint_save
                path = "checkpoint_{:06d}.pickle".format(epoch)
                printlog(path)
                checkpoint_save(domain, state, path)

        # Subtract second part of callback time.
        curtime = time.time()
        cbinfo.time_callback += time.time() - time_prev
        time_prev = curtime

    callback.cbinfo = cbinfo
    return callback
