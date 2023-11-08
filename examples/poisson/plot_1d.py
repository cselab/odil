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
    extra = ctx.extra
    mod = ctx.domain.mod
    dw = ctx.step()
    iw = ctx.indices()
    nw = ctx.size()
    u_st = [ctx.field("u"), ctx.field("u", -1), ctx.field("u", 1)]
    apply_bc_u_mod(u_st, iw, nw, mod=mod)
    u, uwm, uwp = split_wm_wp(u_st)
    u_ww = [(uwp[0] - 2 * u + uwm[0]) / dw[0]**2]
    return [sum(u_ww) - extra.rhs]

N = 32
p = 5
parser = argparse.ArgumentParser()
odil.util.add_arguments(parser)
odil.linsolver.add_arguments(parser)
parser.set_defaults(multigrid=1)
parser.set_defaults(optimizer='lbfgsb')
args = parser.parse_args()
odil.setup_outdir(args)
domain = odil.Domain(cshape=[N],
                     multigrid=args.multigrid,
                     dtype=np.float64)
mod = domain.mod
x, = domain.points(loc='c')
u0 = 5 * (1 - x) * x
ref_u = (u0**p / (1 + u0**p))**(1 / p)
dw = domain.step()
iw = domain.indices()
nw = domain.size()
u_st = [ref_u, mod.roll(ref_u, 1, 0), mod.roll(ref_u, -1, 0)]
apply_bc_u_mod(u_st, iw, nw, mod)
u_ww = [(u_st[2] - 2 * u_st[0] + u_st[1]) / dw[0]**2]
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
ax.plot(x, u, 'o', label='ODIL')
ax.plot(x, extra.ref_u, label='reference')
ax.set_xlabel('x')
ax.set_ylabel('u')
plt.show()
