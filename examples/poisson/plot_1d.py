#!/usr/bin/env python3
"""
1D poisson
==========
"""

import argparse
import numpy as np
import pickle

import odil
from odil import plotutil
import matplotlib.pyplot as plt
from odil import printlog
from odil.core import Approx


def get_ref_u(args, domain):
    xw = domain.points()
    ndim = len(xw)
    p = 5
    u = np.prod([(1 - x) * x * 5 for x in xw], axis=0)
    u = (u**p / (1 + u**p))**(1 / p)
    return u

def split_wm_wp(st, dirs):
    q = st[0]
    qwm = [st[2 * i + 1] for i in dirs]
    qwp = [st[2 * i + 2] for i in dirs]
    return q, qwm, qwp


def apply_bc_u_mod(st, iw, nw, dirs, mod):
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
    return res


def plot_func(problem, state, epoch, frame, cbinfo):
    domain = problem.domain
    mod = domain.mod
    extra = problem.extra
    args = extra.args
    # Only plot the last frame.
    if args.frames == 0 and frame is not None:
        return
    ndim = domain.ndim
    key = 'u'
    paths = []
    suff = "" if frame is None else "_{:05d}".format(frame)
    x = domain.points(0)
    fig, ax = plt.subplots()
    u = domain.field(state, key)
    ax.plot(x, u, label='epoch {:}'.format(epoch))
    ax.plot(x, extra.ref_u, label='reference')
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.))
    plotutil.savefig(fig, "u{}".format(suff))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, extra.rhs)
    ax.set_xlabel('x')
    ax.set_ylabel('rhs')
    plotutil.savefig(fig, "rhs{}".format(suff))
    plt.close(fig)


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


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
N = 32
odil.util.add_arguments(parser)
odil.linsolver.add_arguments(parser)
parser.set_defaults(frames=4,
                    report_every=100,
                    history_every=1,
                    plot_every=100)
parser.set_defaults(history_full=50)
parser.set_defaults(multigrid=1)
parser.set_defaults(optimizer='lbfgsb')
parser.set_defaults(every_factor=1)
parser.set_defaults(lr=0.005)
parser.set_defaults(double=1)
parser.set_defaults(outdir='.')
args = parser.parse_args()
odil.setup_outdir(args)
dtype = np.float64 if args.double else np.float32
domain = odil.Domain(cshape=[N],
                     dimnames=['x'],
                     multigrid=args.multigrid,
                     dtype=dtype)
mod = domain.mod
xw = domain.points(loc='c')
ref_u = get_ref_u(args, domain)
rhs = get_discrete_rhs(ref_u, domain, mod)
state = odil.State()
state.fields['u'] = None
state = domain.init_state(state)
extra = argparse.Namespace()
extra.ref_u = ref_u
extra.rhs = rhs
extra.args = args
problem = odil.Problem(operator, domain, extra)
callback = odil.make_callback(problem,
                              args,
                              plot_func=plot_func,
                              history_func=history_func)
odil.util.optimize(args, args.optimizer, problem, state, callback)
plot_func(problem, state, 0, None, None)
