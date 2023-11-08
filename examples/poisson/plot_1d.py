#!/usr/bin/env python3
"""
1D Poisson
==========
"""

import odil
import matplotlib.pyplot as plt

def apply_bc(domain, st):
    ix, = domain.indices()
    nx, = domain.size()
    mod = domain.mod
    extrap = odil.core.extrap_quadh
    st[1] = mod.where(ix == 0, extrap(st[2], st[0], 0), st[1])
    st[2] = mod.where(ix == nx - 1, extrap(st[1], st[0], 0), st[2])


def operator(ctx):
    dx, = ctx.step()
    u_st = [ctx.field('u'), ctx.field('u', -1), ctx.field('u', 1)]
    apply_bc(ctx.domain, u_st)
    u_xx = (u_st[2] - 2 * u_st[0] + u_st[1]) / dx**2
    return [u_xx - ctx.extra['rhs']]

def get_ref_u(x):
    p = 5
    v = 5 * (1 - x) * x
    return (v**p / (1 + v**p))**(1 / p)

N = 32
domain = odil.Domain(cshape=[N], multigrid=True)
mod = domain.mod
x, = domain.points()
dx, = domain.step()
ref_u = get_ref_u(x)
u_st = [ref_u, mod.roll(ref_u, 1, 0), mod.roll(ref_u, -1, 0)]
apply_bc(domain, u_st)
rhs = (u_st[2] - 2 * u_st[0] + u_st[1]) / dx**2
state = odil.State(fields={'u': None})
state = domain.init_state(state)
problem = odil.Problem(operator, domain, extra={'rhs': rhs})
opt = odil.optimizer.make_optimizer('lbfgsb', dtype=domain.dtype, mod=mod)


def loss_grad(arrays):
    domain.arrays_to_state(arrays, state)
    loss, grads, terms, names, norms = problem.eval_loss_grad(state)
    pinfo = {'terms': terms, 'names': names, 'norms': norms, 'loss': loss}
    return loss, grads, pinfo


opt.run(domain.arrays_from_state(state), loss_grad=loss_grad, epochs=33)
fig, ax = plt.subplots()
u = domain.field(state, 'u')
ax.plot(x, u, 'o', label='ODIL')
ax.plot(x, ref_u, label='reference')
ax.set_xlabel('x')
ax.set_ylabel('u')
plt.show()
