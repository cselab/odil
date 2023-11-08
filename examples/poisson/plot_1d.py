#!/usr/bin/env python3
"""
1D poisson
==========
"""

import argparse
import numpy as np
import odil
import matplotlib.pyplot as plt

def split_wm_wp(st):
    return st[0], [st[1]], [st[2]]

def apply_bc_u_mod(st, iw, nw, mod):
    q, qwm, qwp = split_wm_wp(st)
    zero = mod.cast(0, q.dtype)
    extrap = odil.core.extrap_quadh
    qm = mod.where(iw[0] == 0, extrap(qwp[0], q, zero), qwm[0])
    qp = mod.where(iw[0] == nw[0] - 1, extrap(qwm[0], q, zero), qwp[0])
    qwm[0], qwp[0] = qm, qp
    st[1] = qwm[0]
    st[2] = qwp[0]


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
    apply_bc_u_mod(u_st, iw, nw, mod=mod)
    u, uwm, uwp = split_wm_wp(u_st)
    u_ww = [(uwp[i] - 2 * u + uwm[i]) / dw[i]**2 for i in dirs]
    fu = sum(u_ww) - extra.rhs
    res = [fu]
    return res

N = 32
parser = argparse.ArgumentParser()
odil.util.add_arguments(parser)
odil.linsolver.add_arguments(parser)
parser.set_defaults(multigrid=1)
parser.set_defaults(optimizer='lbfgsb')
parser.set_defaults(outdir='.')
args = parser.parse_args()
odil.setup_outdir(args)
domain = odil.Domain(cshape=[N],
                     multigrid=args.multigrid,
                     dtype=np.float64)
mod = domain.mod
x, = domain.points(loc='c')
p = 5
u = (1 - x) * x * 5
ref_u = (u**p / (1 + u**p))**(1 / p)
dw = domain.step()
iw = domain.indices()
nw = domain.size()
u_st = [u, mod.roll(u, 1, 0), mod.roll(u, -1, 0)]
apply_bc_u_mod(u_st, iw, nw, mod)
u, uwm, uwp = split_wm_wp(u_st)
u_ww = [(uwp[0] - 2 * u + uwm[0]) / dw[0]**2]
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
fig, ax = plt.subplots()
x = domain.points(0)
u = domain.field(state, "u")
ax.plot(x, u, label='ODIL')
ax.plot(x, extra.ref_u, label='reference')
ax.set_xlabel('x')
ax.set_ylabel('u')
plt.show()
