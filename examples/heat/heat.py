#!/usr/bin/env python3

import argparse
import numpy as np
import pickle

import odil
from odil import plotutil
import matplotlib.pyplot as plt
from odil import printlog
from odil.runtime import tf


def get_init_u(t, x):
    # Gaussian.
    def f(z):
        return np.exp(-(z - 0.5)**2 * 50)

    return f(x) - f(-0.5)


def get_ref_k(u, mod=np):
    # Gaussian.
    return 0.02 * (mod.exp(-(u - 0.5)**2 * 20))


def get_anneal_factor(epoch, period):
    return 0.5**(epoch / period) if period else 1


def transform_k(knet, mod, kmax):
    return mod.sigmoid(knet) * kmax


def operator_odil(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args
    dt, dx = ctx.step()
    it, ix = ctx.indices()
    nt, nx = ctx.size()
    epoch = ctx.tracers['epoch']

    def stencil_var(key, frozen=False):
        if not args.keep_frozen:
            frozen = False
        return [
            [
                ctx.field(key, 0, 0, frozen=frozen),
                ctx.field(key, 0, -1, frozen=frozen),
                ctx.field(key, 0, 1, frozen=frozen),
            ],
            [
                ctx.field(key, -1, 0, frozen=frozen),
                ctx.field(key, -1, -1, frozen=frozen),
                ctx.field(key, -1, 1, frozen=frozen),
            ],
        ]

    def apply_bc_u(st):
        # Apply boundary conditions by extrapolation to halo cells.
        if args.keep_init:
            # Initial conditions, linear extrapolation.
            u0 = extra.init_u
            q0 = [u0, mod.roll(u0, 1, axis=0), mod.roll(u0, -1, axis=0)]
            extrap = odil.core.extrap_linear
            q, qm = st
            for i in range(3):
                qm[i] = mod.where(it == 0, extrap(q[i], q0[i][None, :]), qm[i])
        # Zero Dirichlet conditions, quadratic extrapolation.
        extrap = odil.core.extrap_quadh
        for q in st:
            q[1] = mod.where(ix == 0, extrap(q[2], q[0], 0), q[1])
            q[2] = mod.where(ix == nx - 1, extrap(q[1], q[0], 0), q[2])
        return st

    u_st = stencil_var('u')
    apply_bc_u(u_st)

    q, qm = u_st
    u_t = (q[0] - qm[0]) / dt
    u_xm = ((q[0] + qm[0]) - (q[1] + qm[1])) / (2 * dx)
    u_xp = ((q[2] + qm[2]) - (q[0] + qm[0])) / (2 * dx)

    uf_st = stencil_var('u', frozen=True)
    apply_bc_u(uf_st)
    qf, qfm = uf_st
    ufxmh = ((qf[0] + qfm[0]) + (qf[1] + qfm[1])) * 0.25
    ufxph = ((qf[2] + qfm[2]) + (qf[0] + qfm[0])) * 0.25

    # Conductivity.
    if args.infer_k:
        km = transform_k(ctx.neural_net('k_net')(ufxmh)[0], mod, args.kmax)
        kp = transform_k(ctx.neural_net('k_net')(ufxph)[0], mod, args.kmax)
    else:
        km = get_ref_k(ufxmh, mod=mod)
        kp = get_ref_k(ufxph, mod=mod)

    # Heat equation.
    qm = u_xm * km
    qp = u_xp * kp
    q_x = (qp - qm) / dx
    fu = u_t - q_x
    if not args.keep_init:
        fu = mod.where(it == 0, ctx.cast(0), fu)
    res = [('fu', fu)]

    if extra.imp_size:
        u = u_st[0]
        # Rescale weight to the total number of points.
        k = args.kimp * (np.prod(ctx.size()) / extra.imp_size)**0.5
        fuimp = extra.imp_mask * (u_st[0][0] - extra.imp_u) * k
        res += [('imp', fuimp)]

    # Regularization.
    if args.kxreg:
        k = args.kxreg * get_anneal_factor(epoch, args.kxregdecay)
        u_x = (u_st[0][0] - u_st[0][1]) / dx
        u_x = mod.where(ix == 0, ctx.cast(0), u_x)
        fxreg = u_x * k
        res += [('xreg', fxreg)]

    if args.ktreg:
        k = args.ktreg * get_anneal_factor(epoch, args.ktregdecay)
        u_t = (u_st[0][0] - u_st[1][0]) / dt
        u_t = mod.where(it == 0, ctx.cast(0), u_t)
        ftreg = u_t * k
        res += [('treg', ftreg)]

    if args.kwreg and args.infer_k:
        domain = ctx.domain
        ww = domain.arrays_from_field(ctx.state.fields['k_net'])
        ww = mod.concatenate([mod.flatten(w) for w in ww], axis=0)
        k = args.kwreg * get_anneal_factor(epoch, args.kwregdecay)
        res += [('wreg', (mod.stop_gradient(ww) - ww) * k)]
    return res


def operator_pinn(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args

    # Inner points.
    inputs = [mod.constant(extra.t_inner), mod.constant(extra.x_inner)]
    u, = ctx.neural_net('u_net')(*inputs)

    def grad(f, *deriv):
        for i in range(len(deriv)):
            for _ in range(deriv[i]):
                f = tf.gradients(f, inputs[i])[0]
        return f

    u_t = grad(u, 1, 0)
    u_x = grad(u, 0, 1)

    # Conductivity.
    if args.infer_k:
        k = transform_k(ctx.neural_net('k_net')(u)[0], mod, args.kmax)
    else:
        k = get_ref_k(u, mod=mod)
    q = k * u_x
    q_x = grad(q, 0, 1)

    res = []

    # Heat equation.
    fu = u_t - q_x
    res += [('eqn', fu)]

    # Boundary conditions.
    u_net_bound, = ctx.neural_net('u_net')(extra.t_bound, extra.x_bound)
    fb = u_net_bound - extra.u_bound
    res += [('bound', fb)]

    # Initial conditions.
    if args.keep_init:
        u_net_init, = ctx.neural_net('u_net')(extra.t_init, extra.x_init)
        fi = u_net_init - extra.u_init
        res += [('init', fi)]

    # Imposed points.
    if extra.imp_size:
        imp_t, imp_x = extra.imp_points.T
        u_net_imp, = ctx.neural_net('u_net')(imp_t, imp_x)
        imp_indices = mod.reshape(extra.imp_indices, [-1, 1])
        u_imp = mod.gather_nd(mod.flatten(extra.imp_u), imp_indices)
        fimp = (u_net_imp - u_imp) * args.kimp
        res += [('imp', fimp)]

    return res


def get_imposed_indices(domain, args, iflat):
    iflat = np.array(iflat)
    rng = np.random.default_rng(args.seed)
    if args.imposed == 'random':
        imp_i = iflat.flatten()
        nimp = min(args.nimp, np.prod(imp_i.size))
        perm = rng.permutation(imp_i)
        imp_i = perm[:nimp]
    elif args.imposed == 'stripe':
        imp_i = iflat.flatten()
        t = np.array(domain.points('t')).flatten()
        imp_i = imp_i[abs(t[imp_i] - 0.5) < 1 / 6]
        nimp = min(args.nimp, np.prod(imp_i.size))
        perm = rng.permutation(imp_i)
        imp_i = perm[:nimp]
    elif args.imposed == 'none':
        imp_i = []
    else:
        raise ValueError("Unknown imposed=" + args.imposed)
    return imp_i


def get_imposed_mask(args, domain):
    mod = domain.mod
    size = np.prod(domain.cshape)
    row = range(size)
    iflat = np.reshape(row, domain.cshape)
    imp_i = get_imposed_indices(domain, args, iflat)
    imp_i = np.unique(imp_i)
    mask = np.zeros(size)
    if len(imp_i):
        mask[imp_i] = 1
        points = [mod.flatten(domain.points(i)) for i in range(domain.ndim)]
        points = np.array(points)[:, imp_i].T
    else:
        points = np.zeros((0, domain.ndim))
    mask = mask.reshape(domain.cshape)
    return mask, points, imp_i


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nt', type=int, default=64, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    parser.add_argument('--Nci',
                        type=int,
                        default=4096,
                        help="Number of collocation points inside domain")
    parser.add_argument('--Ncb',
                        type=int,
                        default=128,
                        help="Number of collocation points on each boundary")
    parser.add_argument('--arch_u',
                        type=int,
                        nargs="*",
                        default=[10, 10],
                        help="Network architecture for temperature in PINN")
    parser.add_argument('--arch_k',
                        type=int,
                        nargs="*",
                        default=[5, 5],
                        help="Network architecture for inferred conductivity")
    parser.add_argument('--solver',
                        type=str,
                        choices=('pinn', 'odil'),
                        default='odil',
                        help="Grid size in x")
    parser.add_argument('--infer_k',
                        type=int,
                        default=0,
                        help="Infer conductivity")
    parser.add_argument('--kxreg',
                        type=float,
                        default=0,
                        help="Space regularization weight")
    parser.add_argument('--kxregdecay',
                        type=float,
                        default=0,
                        help="Decay period of kxreg")
    parser.add_argument('--ktreg',
                        type=float,
                        default=0,
                        help="Time regularization weight")
    parser.add_argument('--ktregdecay',
                        type=float,
                        default=0,
                        help="Decay period of ktreg")
    parser.add_argument('--kwreg',
                        type=float,
                        default=0,
                        help="Regularization of neural network weights")
    parser.add_argument('--kwregdecay',
                        type=float,
                        default=0,
                        help="Decay period of kwreg")
    parser.add_argument('--kimp',
                        type=float,
                        default=2,
                        help="Weight of imposed points")
    parser.add_argument('--keep_frozen',
                        type=int,
                        default=1,
                        help="Respect frozen attribute for fields")
    parser.add_argument('--keep_init',
                        type=int,
                        default=1,
                        help="Impose initial conditions")
    parser.add_argument('--ref_path',
                        type=str,
                        help="Path to reference solution *.pickle")
    parser.add_argument('--imposed',
                        type=str,
                        choices=['random', 'stripe', 'none'],
                        default='none',
                        help="Set of points for imposed solution")
    parser.add_argument('--nimp',
                        type=int,
                        default=200,
                        help="Number of points for imposed=random")
    parser.add_argument('--noise',
                        type=float,
                        default=0,
                        help="Magnitude of perturbation of reference solution")
    parser.add_argument('--kmax',
                        type=float,
                        default=0.1,
                        help="Maximum value of conductivity")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_heat')
    parser.set_defaults(linsolver='direct')
    parser.set_defaults(optimizer='adam')
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=0)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(plotext='png', plot_title=1)
    parser.set_defaults(plot_every=2000,
                        report_every=500,
                        history_full=10,
                        history_every=100,
                        frames=10)
    return parser.parse_args()


@tf.function()
def eval_u_net(domain, net, arrays):
    domain.arrays_to_field(arrays, net)
    tt, xx = domain.points()
    net_u, = odil.core.eval_neural_net(net, [tt, xx], domain.mod)
    return net_u


def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_1d

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    title0 = "u epoch={:}".format(epoch) if args.plot_title else None
    title1 = "k epoch={:}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "k_{:05d}.{}".format(frame, args.plotext)
    printlog(path0, path1)

    if args.solver == 'odil':
        state_u = domain.field(state, 'u')
    elif args.solver == 'pinn':
        net = state.fields['u_net']
        arrays = domain.arrays_from_field(net)
        state_u = eval_u_net(domain, net, arrays)
    state_u = np.array(state_u)

    def callback(i, fig, ax, data, extent):
        if i == 0:
            imp_t, imp_x = extra.imp_points.T
            ax.scatter(imp_x,
                       imp_t,
                       s=0.5,
                       alpha=1,
                       edgecolor='none',
                       facecolor='k',
                       zorder=100)

    plot_1d(domain,
            np.array(extra.imp_u),
            state_u,
            path=path0,
            title=title0,
            cmap='YlOrBr',
            nslices=5,
            interpolation='bilinear',
            callback=callback,
            transpose=True,
            umin=0,
            umax=1)

    # Plot conductivity.
    fig, ax = plt.subplots(figsize=(1.7, 1.5))
    ref_uk = extra.ref_uk
    ref_k = get_ref_k(ref_uk)
    if args.infer_k:
        k, = domain.neural_net(state, 'k_net')(ref_uk)
        k = transform_k(k, mod, args.kmax)
    else:
        k = None
    if k is not None:
        ax.plot(ref_uk, k, zorder=10)
    ax.plot(ref_uk, ref_k, c='C2', lw=1.5, zorder=1)
    ax.set_xlabel('u')
    ax.set_ylabel('k')
    ax.set_ylim(0, 0.03)
    ax.set_title(title1)
    fig.savefig(path1, bbox_inches='tight')
    plt.close(fig)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        d = dict()
        d['state_u'] = state_u
        d['ref_u'] = extra.ref_u  # Reference without noise.
        d['imp_u'] = extra.imp_u  # Reference with noise.
        d['ref_uk'] = ref_uk
        d['k'] = k
        d['ref_k'] = ref_k
        d['imp_indices'] = extra.imp_indices
        d['imp_points'] = extra.imp_points
        d = odil.core.struct_to_numpy(mod, d)
        with open(path, 'wb') as f:
            pickle.dump(d, f)


def get_error(domain, extra, state, key):
    args = extra.args
    mod = domain.mod
    if key == 'u':
        if args.solver == 'odil':
            state_u = domain.field(state, key)
        elif args.solver == 'pinn':
            net = state.fields['u_net']
            arrays = domain.arrays_from_field(net)
            state_u = eval_u_net(domain, net, arrays)
        ref_u = extra.ref_u
        return np.sqrt(np.mean((state_u - ref_u)**2))
    elif key == 'k' and args.infer_k:
        k, = domain.neural_net(state, 'k_net')(extra.ref_uk)
        k = transform_k(k, mod, args.kmax)
        max_k = extra.ref_k.max()
        return np.sqrt(np.mean((k - extra.ref_k)**2)) / max_k
    return None


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    extra = problem.extra
    for key in ['u', 'k']:
        error = get_error(domain, extra, state, key)
        if error is not None:
            history.append('error_' + key, error)


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    extra = problem.extra
    res = dict()
    for key in ['u', 'k']:
        error = get_error(domain, extra, state, key)
        if error is not None:
            res[key] = error
    printlog('error: ' + ', '.join('{}:{:.5g}'.format(*item)
                                   for item in res.items()))


def load_fields_interp(path, keys, domain):
    '''
    Loads fields from file `path` and interpolates them to shape `domain.shape`.

    keys: `list` of `str`
        Keys of fields to load.
    '''
    from scipy.interpolate import RectBivariateSpline
    src_state = odil.State(fields={key: odil.Field() for key in keys})
    state = odil.State(fields={key: odil.Field() for key in keys})
    odil.core.checkpoint_load(domain, src_state, path)
    x1, y1 = domain.points_1d()
    for key in keys:
        src_u = src_state.fields[key]
        src_domain = odil.Domain(cshape=src_u.array.shape,
                                 dimnames=('x', 'y'),
                                 lower=domain.lower,
                                 upper=domain.upper,
                                 dtype=domain.dtype,
                                 mod=odil.backend.ModNumpy())
        src_u = src_domain.init_field(src_u)
        if src_domain.cshape != domain.cshape:
            src_x1, src_y1 = src_domain.points_1d()
            fu = RectBivariateSpline(src_x1, src_y1, src_u.array)
            state.fields[key].array = fu(x1, y1)
        else:
            state.fields[key] = src_u
    return state


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx),
                         dimnames=('t', 'x'),
                         multigrid=args.multigrid,
                         dtype=dtype)
    if domain.multigrid:
        printlog('multigrid levels:', domain.mg_cshapes)
    mod = domain.mod
    # Evaluate exact solution, boundary and initial conditions.
    tt, xx = domain.points()
    t1, x1 = domain.points_1d()
    init_u = get_init_u(x1 * 0, x1)

    # Load reference solution.
    if args.ref_path is not None:
        printlog("Loading reference solution from '{}'".format(args.ref_path))
        ref_state = load_fields_interp(args.ref_path, ['u'], domain)
        ref_u = domain.cast(ref_state.fields['u'].array)
    else:
        ref_u = get_init_u(tt, xx)

    # Add noise after choosing points with imposed values.
    imp_u = ref_u
    if args.noise:
        rng = np.random.default_rng(args.seed)
        imp_u += rng.normal(loc=0, scale=args.noise, size=ref_u.shape)

    imp_mask, imp_points, imp_indices = get_imposed_mask(args, domain)
    imp_size = len(imp_points)
    with open("imposed.csv", 'w') as f:
        f.write(','.join(domain.dimnames) + '\n')
        for p in imp_points:
            f.write('{:},{:}'.format(*p) + '\n')

    ref_uk = np.linspace(0, 1, 200).astype(domain.dtype)
    ref_k = get_ref_k(ref_uk)

    extra = argparse.Namespace()

    def add_extra(d, *keys):
        for key in keys:
            setattr(extra, key, d[key])

    add_extra(locals(), 'args', 'ref_u', 'ref_uk', 'ref_k', 'init_u',
              'imp_mask', 'imp_size', 'imp_u', 'imp_indices', 'imp_points')
    extra.epoch = mod.variable(domain.cast(0))

    if args.solver == 'pinn':
        t_inner, x_inner = domain.random_inner(args.Nci)
        t_bound0, x_bound0 = domain.random_boundary(1, 0, args.Ncb)
        t_bound1, x_bound1 = domain.random_boundary(1, 1, args.Ncb)
        t_bound = np.hstack((t_bound0, t_bound1))
        x_bound = np.hstack((x_bound0, x_bound1))
        t_init, x_init = domain.random_boundary(0, 0, args.Ncb)
        u_init = get_init_u(t_init, x_init)
        u_bound = get_init_u(t_bound, x_bound)
        printlog('Number of collocation points:')
        printlog('inner: {:}'.format(len(t_inner)))
        printlog('init: {:}'.format(len(t_init)))
        printlog('bound: {:}'.format(len(t_bound)))
        add_extra(locals(), 't_inner', 'x_inner', 't_bound', 'x_bound',
                  't_init', 'x_init', 'u_init', 'u_bound')

    state = odil.State()
    if args.solver == 'odil':
        operator = operator_odil
        state.fields['u'] = np.zeros(domain.cshape)
    elif args.solver == 'pinn':
        state.fields['u_net'] = domain.make_neural_net([2] + args.arch_u + [1])
        operator = operator_pinn
    else:
        raise RuntimeError(f'Unknown solver={solver}')

    if args.infer_k:
        state.fields['k_net'] = domain.make_neural_net([1] + args.arch_k + [1])

    state = domain.init_state(state)

    problem = odil.Problem(operator, domain, extra)

    if args.checkpoint is not None:
        printlog("Loading checkpoint '{}'".format(args.checkpoint))
        odil.core.checkpoint_load(domain, state, args.checkpoint)
        tpath = os.path.splitext(args.checkpoint)[0] + '_train.pickle'
        if args.checkpoint_train is None:
            assert os.path.isfile(tpath), "File not found '{}'".format(tpath)
            args.checkpoint_train = tpath

    if args.checkpoint_train is not None:
        printlog("Loading history from '{}'".format(args.checkpoint_train))
        history.load(args.checkpoint_train)
        args.epoch_start = history.get('epoch', [args.epoch_start])[-1]
        frame = history.get('frame', [args.frame_start])[-1]
        printlog("Starting from epoch={:} frame={:}".format(
            args.epoch_start, args.frame_start))
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(
        args, relpath_args=['checkpoint', 'checkpoint_train', 'ref_path'])
    problem, state = make_problem(args)
    callback = odil.make_callback(problem,
                                  args,
                                  plot_func=plot_func,
                                  history_func=history_func,
                                  report_func=report_func)
    odil.util.optimize(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
