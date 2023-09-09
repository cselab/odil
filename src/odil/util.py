import sys
import os
import json
import argparse
import psutil
import time
import numpy as np
from collections import defaultdict
from functools import partial

from .optimizer import make_optimizer, Optimizer
from .history import History
from .backend import ModNumpy

g_log_file = sys.stderr  # File used by printlog()
g_log_echo = False  # True if printlog() should print to stderr.

cupy = None
cupyx = None


def assert_equal(first, second, msg=None):
    if not (first == second):
        raise ValueError("Expected equal '{:}' and '{:}'{}".format(
            first, second, msg))


def import_cupy(args):
    global cupy, cupyx
    import cupy
    import cupyx
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg
    from .backend import ModCupy
    printlog("Using CuPy with memory limit {:.3f} GiB".format(
        cupy.get_default_memory_pool().get_limit() / (1 << 30)))
    mod = ModCupy(cupy, cupyx.scipy.sparse)
    return mod


def set_log_file(f, echo=False):
    global g_log_file, g_log_echo
    g_log_file = f
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
                        default=5,
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
    parser.add_argument('--bfgs_m',
                        type=int,
                        default=50,
                        help="History size for L-BFGS")
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
    from .linsolver import solve
    opt = Optimizer(name='newton', displayname='Newton')
    printlog("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])
    for epoch in range(args.epoch_start, args.epochs + 1):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        const, m = problem.linearize(state, epoch=epoch)

        # Compute loss and residuals with initial state, to be used by callback.
        opt.pinfo = [np.mean(c**2)**0.5 for c in const]
        if callback:
            callback(packed, epoch, opt)
        if epoch == args.epochs:
            break

        opt.evals += 1
        const = np.hstack([eq.flatten() for eq in const])
        problem.timer_total.push('tt_linsolver')
        timer = Timer()
        timer.push('linsolver')
        dpacked = solve(m, -const, args, dhistory, args.linsolver)
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last.append(timer)
        packed += dpacked


def optimize_grad(args, optname, problem, state, callback=None, **kwargs):
    domain = problem.domain
    mod = domain.mod

    def loss_grad(arrays, epoch):
        domain.arrays_to_state(arrays, state)
        loss, grads, terms, names, norms = problem.eval_loss_grad(state)
        pinfo = {'terms': terms, 'names': names, 'norms': norms, 'loss': loss}
        return loss, grads, pinfo

    def callback_wrap(arrays, epoch, pinfo):
        domain.arrays_to_state(arrays, state)
        return callback(state, epoch, pinfo)

    # Custom parameters.
    if args.bfgs_m is not None:
        kwargs['m'] = args.bfgs_m
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
    loss, _, pinfo = loss_grad(arrays, args.epoch_start)
    if callback:
        callback(state, args.epoch_start, pinfo)

    arrays, optinfo = opt.run(arrays,
                              loss_grad=loss_grad,
                              epochs=args.epochs - args.epoch_start,
                              callback=callback_wrap if callback else None,
                              epoch_start=args.epoch_start,
                              lr=args.lr,
                              **kwargs)
    printlog(optinfo)


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


def optimize_multigrid_base(opt,
                            args,
                            problem,
                            state,
                            callback=None,
                            mod=None,
                            datalevels=None):

    verbose = args.linsolver_verbose

    def printv(*m):
        if verbose:
            printlog(*m)

    printv("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])
    printv("Constructing Multigrid...")
    from .multigrid import Multigrid
    mg = Multigrid(problem.domain.cshape,
                   restriction=args.restriction,
                   nvar=len(state.fields),
                   mod=mod,
                   dtype=problem.domain.dtype,
                   nlevels=args.nlvl)
    printv('levels: {}'.format(', '.join(map(str, mg.nnw))))
    for epoch in range(args.epoch_start, args.epochs + 1):
        s = problem.unpack_state(packed)
        problem.domain.assign_active_state(state, s)
        if callback:
            callback(packed, epoch, opt)
        if epoch == args.epochs:
            break

        timer = Timer()
        timer.push('linsolver')
        if datalevels is not None:
            AA = [None] * mg.nlevels
            for level in range(mg.nlevels):
                printv("level=", level)
                lproblem = datalevels[level].problem
                lstate = datalevels[level].state

                def coarsen(u):
                    R = mg.RRsingle[level - 1]
                    u = mod.reshape(u, [-1])
                    return mod.reshape(R @ u, lproblem.domain.cshape)

                if level > 0:
                    for k in problem.domain.fieldnames:
                        lstate.fields[k].assign(
                            coarsen(datalevels[level - 1].state.fields[k]))

                const, m = lproblem.linearize(lstate,
                                              epoch=epoch,
                                              mod=mod.mod,
                                              modsp=mod.modsp)
                opt.last_residual = [np.mean(c**2)**0.5 for c in const]
                const = mod.stack([c for c in const]).flatten()
                if level == 0:
                    rhs = -m.T @ const
                AA[level] = m.T @ m
                if level == 0:
                    matr = AA[0]
            opt.evals += 1
            mg.update_A(AA)
        else:
            printv("Evaluating gradients...")
            const, m = problem.linearize(state,
                                         epoch=epoch,
                                         mod=mod.mod,
                                         modsp=mod.modsp)
            opt.last_residual = [mod.numpy(mod.norm(c)) for c in const]
            opt.evals += 1
            const = mod.stack([c for c in const]).flatten()
            printv("Computing rhs...")
            rhs = -m.T @ const
            printv("Computing matr...")
            matr = m.T @ m

            printv("Calling update_A()...")
            mg.update_A(matr)

        sol = np.zeros_like(rhs)
        for it in range(args.linsolver_maxiter):
            sol = mg.step(sol,
                          rhs,
                          ndirect=args.ndirect,
                          pre=args.smooth_pre,
                          post=args.smooth_post,
                          smoother=partial(mg.smoother_jacobi,
                                           omega=args.omega,
                                           full=False))
            if verbose > 1:
                printv('it={:d} r={:.5g}'.format(
                    it,
                    np.mean((rhs - matr @ sol)**2)**0.5))
        dpacked = mod.numpy(sol)
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last = timer
        packed += dpacked


def optimize_multigrid(args, problem, state, callback, datalevels=None):
    import scipy.sparse
    mod = ModNumpy(np, scipy.sparse)
    opt = Optimizer(name='multigrid', displayname='Multigrid')
    return optimize_multigrid_base(opt,
                                   args,
                                   problem,
                                   state,
                                   callback,
                                   mod=mod,
                                   datalevels=datalevels)


def optimize_multigridcp(args, problem, state, callback):
    mod = import_cupy(args)
    opt = Optimizer(name='multigridcp', displayname='MultigridCupy')
    return optimize_multigrid_base(opt,
                                   args,
                                   problem,
                                   state,
                                   callback,
                                   mod=mod)


def optimize_multigridop_base(opt,
                              args,
                              problem,
                              state,
                              callback=None,
                              mod=None):
    mod = problem.mod
    verbose = args.linsolver_verbose
    domain = problem.domain

    def printv(m):
        if verbose:
            printlog(m)

    printv("Running {} optimizer".format(opt.displayname))
    packed = problem.pack_state(state)
    dhistory = defaultdict(lambda: [])
    printv("Constructing MultigridOp...")
    nvar = len(state.fields)
    from .multigrid import MultigridOp, SparseOperator
    mg = MultigridOp(domain.cshape,
                     nvar=nvar,
                     restriction=args.restriction,
                     mod=mod,
                     dtype=domain.dtype,
                     nlevels=args.nlvl)
    printv('levels: {}'.format(', '.join(map(str, mg.nnw))))
    for epoch in range(args.epochs + 1):
        s = problem.unpack_state(packed)
        domain.assign_active_state(state, s)
        if callback:
            callback(packed, epoch, opt)
        if epoch == args.epochs:
            break

        printv("Evaluating gradients...")
        epoch = mod.constant(epoch, dtype=domain.dtype)
        const, grads, field_desc, wgrads = problem._eval_grad(
            state.fields, state.weights, epoch)
        opt.last_residual = [np.mean(c**2)**0.5 for c in const]

        neqn = len(const)  # Number of equations.

        nw = domain.cshape
        stf = [[dict() for _ in range(nvar)] for _ in range(neqn)]
        for i in range(neqn):  # Loop over equations.
            for j in range(len(field_desc)):  # Loop over grid field variables.
                varindex, dw = field_desc[j]
                g = grads[i][j]
                if g is None:
                    continue
                if mod.sum(g**2) == 0:
                    continue
                stf[i][varindex][tuple(np.array(dw))] = mod.native(g.numpy())
        const = [mod.native(c.numpy()) for c in const]
        printv("Constructing SparseOperator...")
        mm = [[
            SparseOperator(s, nw, mod=mod, dtype=domain.dtype) for s in row
        ] for row in stf]
        del stf

        opt.evals += 1
        timer = Timer()
        timer.push('linsolver')
        printv("Computing rhs...")
        rhs = [
            -sum([mm[i][j].mul_transpose_field(const[i]) for i in range(neqn)])
            for j in range(nvar)
        ]
        printv("Computing matr...")
        matr = [[None for _ in range(nvar)] for _ in range(nvar)]
        for i in range(nvar):
            for j in range(nvar):
                matr[i][j] = mm[0][i].mul_transpose_op(mm[0][j])
                for e in range(1, neqn):
                    matr[i][j] = matr[i][j].add_elementwise(
                        mm[e][i].mul_transpose_op(mm[e][j]))
        del mm
        printv("Calling update_A()...")
        mg.update_A(matr)
        sol = [np.zeros_like(rhs[i]) for i in range(nvar)]
        for it in range(args.linsolver_maxiter):
            sol = mg.step(sol,
                          rhs,
                          pre=args.smooth_pre,
                          post=args.smooth_post,
                          ndirect=args.ndirect,
                          smoother=partial(mg.smoother_jacobi,
                                           omega=args.omega))
            if verbose > 1:
                printv('it={:d} r={}'.format(
                    it, ', '.join('{:.5g}'.format(np.mean(r**2)**0.5)
                                  for r in mg.residual(matr, sol, rhs))))
        dpacked = mod.numpy(mod.reshape(mod.stack(sol), [-1]))
        timer.pop()
        problem.timer_total.append(timer)
        problem.timer_last = timer
        packed += dpacked


def optimize_multigridop(args, problem, state, callback):
    import scipy.sparse
    mod = ModNumpy(np, scipy.sparse)
    opt = Optimizer(name='multigridop', displayname='MultigridOp')
    return optimize_multigridop_base(opt,
                                     args,
                                     problem,
                                     state,
                                     callback,
                                     mod=mod)


def optimize_multigridopcp(args, problem, state, callback):
    mod = import_cupy(args)
    opt = Optimizer(name='multigridopcp', displayname='MultigridOpCupy')
    return optimize_multigridop_base(opt,
                                     args,
                                     problem,
                                     state,
                                     callback,
                                     mod=mod)


def get_env_config():
    keys = [
        'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'ODIL_WARN', 'ODIL_BACKEND',
        'ODIL_JIT', 'ODIL_MT', 'ODIL_DTYPE'
    ]
    return {k: os.environ.get(k, '') for k in keys}


def setup_outdir(args, relpath_args=None):
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
    set_log_file(open("train.log", 'w'))

    # Update relative paths.
    if relpath_args is None:
        relpath_args = []
    for k in relpath_args:
        if getattr(args, k) is not None:
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
                  calc_func=None,
                  report_func=None,
                  history_func=None,
                  plot_func=None):
    cbinfo = argparse.Namespace()
    cbinfo.walltime = 0
    cbinfo.epoch = 0
    cbinfo.time_callback = 0
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

        task_report = (args.report_every and epoch % args.report_every == 0)
        task_history = (history is not None
                        and (epoch % args.history_every == 0
                             or epoch < args.history_full))
        task_plot = (epoch % args.plot_every == 0 and (epoch or args.frames))

        if task_report or task_history or task_plot:
            memusage = get_memory_usage_kb()
            if calc_func is not None:
                calc_func(problem=problem,
                          state=state,
                          epoch=epoch,
                          cbinfo=cbinfo)

        # Subtract first part of callback time.
        curtime = time.time()
        cbinfo.time_callback += curtime - time_prev
        time_prev = curtime
        walltime = curtime - cbinfo.time_start - cbinfo.time_callback

        if task_report:
            printlog("\nepoch={:05d}".format(epoch))
            if pinfo and 'norms' in pinfo:
                norms, names = pinfo['norms'], pinfo['names']
                printlog('residual: ' + ', '.join(
                    '{}:{:.5g}'.format(name or str(i), np.array(norm))
                    for i, (norm, name) in enumerate(zip(norms, names))))
            printlog("memory: {:} MiB".format(memusage // 1024))
            printlog("walltime: {:.3f} s".format(walltime))
            printlog("walltime+callback: {:.3f} s".format(  #
                walltime + cbinfo.time_callback))

            if report_func is not None:
                report_func(problem=problem,
                            state=state,
                            epoch=epoch,
                            cbinfo=cbinfo)
            if epoch > cbinfo.epoch:
                wte = (walltime - cbinfo.walltime) / (epoch - cbinfo.epoch)
                printlog("walltime/epoch: {:.3f} ms".format(wte * 1000))
                printlog("throughput: {:.3f}M cells/s".format(
                    np.prod(domain.cshape) / wte / 1e6))
                cbinfo.walltime = walltime
                cbinfo.epoch = epoch

        if task_history:
            history.append('epoch', epoch)
            history.append('frame', cbinfo.frame)
            if pinfo and 'norms' in pinfo:
                norms, names = pinfo['norms'], pinfo['names']
                for i, (norm, name) in enumerate(zip(norms, names)):
                    history.append('norm_{:}'.format(name or str(i)),
                                   np.array(norm))
            if pinfo and 'loss' in pinfo:
                history.append('loss', pinfo['loss'])
            history.append('walltime', walltime)
            history.append('memory', memusage / 1024)
            if history_func is not None:
                history_func(problem=problem,
                             state=state,
                             epoch=epoch,
                             history=history,
                             cbinfo=cbinfo)
            history.write()

        if task_plot:
            if plot_func is not None:
                plot_func(problem=problem,
                          state=state,
                          epoch=epoch,
                          frame=cbinfo.frame,
                          cbinfo=cbinfo)
            cbinfo.frame += 1

        # Subtract second part of callback time.
        curtime = time.time()
        cbinfo.time_callback += time.time() - time_prev
        time_prev = curtime

    callback.cbinfo = cbinfo
    return callback
