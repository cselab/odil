#!/usr/bin/env python3
"""
1D poisson
==========
"""

import argparse
import numpy as np
import odil
import matplotlib.pyplot as plt

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

N = 32
parser = argparse.ArgumentParser()
parser.set_defaults(frames=4,
                    report_every=100,
                    history_every=1,
                    plot_every=100)
odil.util.add_arguments(parser)
odil.linsolver.add_arguments(parser)
parser.set_defaults(multigrid=1)
parser.set_defaults(optimizer='lbfgsb')
parser.set_defaults(every_factor=1)
parser.set_defaults(lr=0.005)
parser.set_defaults(double=1)
parser.set_defaults(outdir='.')
args = parser.parse_args()
odil.setup_outdir(args)
domain = odil.Domain(cshape=[N],
                     dimnames=['x'],
                     multigrid=args.multigrid,
                     dtype=np.float64)
mod = domain.mod
xw = domain.points(loc='c')
ndim = len(xw)
p = 5
u = np.prod([(1 - x) * x * 5 for x in xw], axis=0)
ref_u = (u**p / (1 + u**p))**(1 / p)

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
rhs = sum(u_ww)

state = odil.State()
state.fields['u'] = None
state = domain.init_state(state)
extra = argparse.Namespace()
extra.ref_u = ref_u
extra.rhs = rhs
extra.args = args
problem = odil.Problem(operator, domain, extra)
callback = odil.make_callback(problem, args)
odil.util.optimize(args, args.optimizer, problem, state, callback)
x = domain.points(0)
fig, ax = plt.subplots()
u = domain.field(state, "u")
ax.plot(x, u, label="ODIL")
ax.plot(x, extra.ref_u, label='reference')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()
fig, ax = plt.subplots()
ax.plot(x, extra.rhs)
ax.set_xlabel('x')
ax.set_ylabel('rhs')
plt.show()
