#!/usr/bin/env python3

import argparse
import numpy as np
import pickle
import odil
from odil import plotutil
import matplotlib.pyplot as plt
from odil import printlog
"""
Solves the Poisson equation in a multi-dimensional cube
with zero Dirichlet boundary conditions.
"""


def get_ref_u(name, args, domain):
    xw = domain.points()
    ndim = len(xw)
    if name == 'hat':
        p = 5
        u = np.prod([(1 - x) * x * 5 for x in xw], axis=0)
        u = (u**p / (1 + u**p))**(1 / p)
    elif name == 'osc':
        pi = np.pi
        k = args.osc_k
        x, y = xw
        u = np.sin(pi * (k * x)**2) * np.sin(pi * y)
    else:
        raise ValueError("Unknown name=" + name)
    return u


def get_ref_rhs(name, args, domain):
    xw = domain.points()
    ndim = len(xw)
    if name == 'osc':
        pi, cos, sin = np.pi, np.cos, np.sin
        k = args.osc_k
        x, y = xw
        fu = (((-4 * k**4 * pi**2 * x**2) - pi**2) * sin(k**2 * pi * x**2) +
              2 * k**2 * pi * cos(k**2 * pi * x**2)) * sin(pi * y)
    else:
        raise ValueError("Unknown name=" + name)
    return fu


def split_wm_wp(st, dirs):
    q = st[0]
    qwm = [st[2 * i + 1] for i in dirs]
    qwp = [st[2 * i + 2] for i in dirs]
    return q, qwm, qwp


def apply_bc_u_mod(st, iw, nw, dirs, mod):
    'Applies zero-Dirichlet boundary conditions.'
    q, qwm, qwp = split_wm_wp(st, dirs)
    zero = mod.cast(0, q.dtype)
    for i in dirs:
        extrap = odil.core.extrap_quadh
        qm = mod.where(iw[i] == 0, extrap(qwp[i], q, zero), qwm[i])
        qp = mod.where(iw[i] == nw[i] - 1, extrap(qwm[i], q, zero), qwp[i])
        qwm[i], qwp[i] = qm, qp
    for i in dirs:
        st[2 * i + 1] = qwm[i]
        st[2 * i + 2] = qwp[i]


def get_discrete_rhs(u, domain, mod):
    ndim = domain.ndim
    dirs = range(ndim)
    dw = domain.step()
    iw = domain.indices()
    nw = domain.size()
    u_st = [None] * (2 * ndim + 1)
    u_st[0] = u
    for i in dirs:
        u_st[2 * i + 1] = mod.roll(u, 1, i)
        u_st[2 * i + 2] = mod.roll(u, -1, i)
    apply_bc_u_mod(u_st, iw, nw, dirs, mod)
    u, uwm, uwp = split_wm_wp(u_st, dirs)
    u_ww = [(uwp[i] - 2 * u + uwm[i]) / dw[i]**2 for i in dirs]
    fu = sum(u_ww)
    return fu


def operator(ctx):
    domain = ctx.domain
    extra = ctx.extra
    args = extra.args
    mod = domain.mod
    ndim = domain.ndim
    dirs = range(ndim)
    dw = ctx.step()
    iw = ctx.indices()
    nw = ctx.size()

    def stencil_var(key):
        st = [ctx.field(key)]
        for i in dirs:
            w = [-1 if j == i else 0 for j in dirs]
            st.append(ctx.field(key, *w))
            w = [1 if j == i else 0 for j in dirs]
            st.append(ctx.field(key, *w))
        return st

    u_st = stencil_var('u')
    apply_bc_u_mod(u_st, iw, nw, dirs, mod=mod)
    u, uwm, uwp = split_wm_wp(u_st, dirs)
    u_ww = [(uwp[i] - 2 * u + uwm[i]) / dw[i]**2 for i in dirs]
    fu = sum(u_ww) - extra.rhs
    res = [fu]

    if args.mgloss:
        from functools import partial
        restrict = partial(odil.core.restrict_to_coarser,
                           loc='c' * ndim,
                           mod=mod)
        for _ in range(args.mgloss):
            fu = restrict(fu)
            res += [fu]
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ndim',
                        type=int,
                        choices=[1, 2, 3, 4, 5, 6],
                        default=2,
                        help="Space dimension")
    parser.add_argument('--N', type=int, default=32, help="Grid size")
    parser.add_argument('--cellbased',
                        type=int,
                        default=1,
                        help="Cell-based fields")
    parser.add_argument('--dump_xmf',
                        type=int,
                        default=0,
                        help="Dump XMF+RAW files")
    parser.add_argument('--plot', type=int, default=0, help="Enable plotting")
    parser.add_argument('--ref',
                        type=str,
                        default='hat',
                        choices=('hat', 'osc'),
                        help="Reference solution")
    parser.add_argument('--rhs',
                        type=str,
                        default='discrete',
                        choices=('discrete', 'exact'),
                        help="Reference right-hand side")
    parser.add_argument('--osc_k',
                        type=float,
                        default=2,
                        help="Parameter for ref='osc'")
    parser.add_argument('--mgloss',
                        type=int,
                        default=0,
                        help="Use multigrid norm with mgloss terms in loss")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(frames=4,
                        report_every=100,
                        history_every=10,
                        plot_every=100,
                        history_full=50)

    parser.set_defaults(optimizer='adam')
    parser.set_defaults(multigrid=1)
    parser.set_defaults(lr=0.005)
    parser.set_defaults(double=1)

    parser.set_defaults(outdir='out_poisson')
    return parser.parse_args()


def write_field(u, name, path, domain, cellbased):
    ndim = domain.ndim
    dw = domain.step()
    axes = tuple(reversed(range(ndim)))
    u = np.transpose(u, axes)
    odil.write_raw_with_xmf(u, path, spacing=dw, name=name, cell=cellbased)


def plot_func(problem, state, epoch, frame, cbinfo):
    domain = problem.domain
    mod = domain.mod
    extra = problem.extra
    args = extra.args
    if args.frames == 0 and frame is not None:
        # Only plot the last frame.
        return
    ndim = domain.ndim
    key = 'u'
    paths = []
    suff = "" if frame is None else "_{:05d}".format(frame)
    if args.plot and ndim == 1:
        x = domain.points(0)
        fig, ax = plt.subplots()
        u = domain.field(state, key)
        ax.plot(x, u, label='epoch {:}'.format(epoch))
        ax.plot(x, extra.ref_u, label='reference')
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend(loc='upper left', bbox_to_anchor=(1., 1.))
        plotutil.savefig(fig, "u" + suff, pad_inches=0.01)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(x, extra.rhs)
        ax.set_xlabel('x')
        ax.set_ylabel('rhs')
        plotutil.savefig(fig, "rhs" + suff, pad_inches=0.01)
        plt.close(fig)

    if args.dump_xmf and ndim in [2, 3]:
        u = domain.field(state, key)
        path = key + '{}.xmf'.format(suff)
        write_field(u, key, path, domain, args.cellbased)
        paths.append(path)

    if args.dump_data:
        x = domain.points()
        u = domain.field(state, key)
        path = "data{}.pickle".format(suff)
        d = dict()
        d['x'] = x
        d['u'] = u
        d['ref_u'] = extra.ref_u
        d['rhs'] = extra.rhs
        if 0 and args.multigrid:  # XXX
            d['u_cumsum'] = [q.numpy() for q in us]
        d = odil.core.struct_to_numpy(mod, d)
        with open(path, 'wb') as f:
            pickle.dump(d, f)
        paths.append(path)

    printlog(' '.join(paths))


def get_error(domain, extra, state, key):
    mod = domain.mod
    state_u = domain.field(state, key)
    du = state_u - extra.ref_u
    return np.sqrt(np.mean(du**2))


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    extra = problem.extra
    for key in state.fields:
        error = get_error(domain, extra, state, key)
        history.append('error_' + key, error)


def report_func(problem, state, epoch, cbinfo):
    res = dict()
    domain = problem.domain
    extra = problem.extra
    for key in state.fields:
        res[key] = get_error(domain, extra, state, key)
    printlog('error: ' + ', '.join('{}:{:.5g}'.format(*item)
                                   for item in res.items()))


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    ndim = args.ndim
    domain = odil.Domain(cshape=[args.N] * ndim,
                         dimnames=['x', 'y', 'z', 'sx', 'sy', 'sz'][:ndim],
                         multigrid=args.multigrid,
                         dtype=dtype)
    if domain.multigrid:
        printlog('multigrid levels:', domain.mg_cshapes)
    mod = domain.mod

    cellbased = args.cellbased
    if cellbased:
        xw = domain.points(loc='c' * ndim)
    else:
        xw = domain.points(loc='n' * ndim)

    # Reference solution.
    ref_u = get_ref_u(args.ref, args, domain)
    # Reference right-hand side.
    if args.rhs == 'discrete':
        rhs = get_discrete_rhs(ref_u, domain, mod)
    else:
        rhs = get_ref_rhs(args.ref, args, domain)

    # Initial state.
    state = odil.State()
    state.fields['u'] = None
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.ref_u = ref_u
    extra.rhs = rhs
    extra.args = args
    if args.plot:
        write_field(extra.ref_u, 'u', 'ref_u.xmf', domain, cellbased)
        write_field(extra.rhs, 'rhs', 'rhs.xmf', domain, cellbased)

    problem = odil.Problem(operator, domain, extra)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(problem,
                                  args,
                                  plot_func=plot_func,
                                  history_func=history_func,
                                  report_func=report_func)
    odil.util.optimize(args, args.optimizer, problem, state, callback)
    plot_func(problem, state, 0, None, None)


if __name__ == "__main__":
    main()
