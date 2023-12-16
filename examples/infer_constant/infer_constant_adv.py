#!/usr/bin/env python3

import argparse
import numpy as np

import odil
from odil import printlog
from odil import plotutil
"""
Inference of unknown constant parameters in the advection-diffusion equation
from the solution at the initial and final time.
"""

C_DIFF = 0.01  # Diffusivity.
C_SRC = 0.1  # Uniform source.
C_VEL = 0.2  # Advection velocity.


def get_exact(t, x):
    '''
    Returns solution of the equation:
      u_t = nu * u_xx + src
    '''
    t = np.array(t)
    x = np.array(x)
    u = np.zeros_like(x)
    nu = C_DIFF
    xx = x - t * C_VEL
    ii = [1, 2, 3]
    for i in ii:
        k = 2 * i * np.pi
        u += np.cos(xx * k) * np.exp(-nu * k**2 * t)
    u /= 2 * len(ii)
    # Add uniform source.
    src = C_SRC
    u += src * t
    return u


def transform_u(u, extra, mod):
    u = mod.concatenate(
        [extra.u_init[None, :], u[1:-1], extra.u_final[None, :]], axis=0)
    return u


def operator_adv(ctx):
    mod = ctx.mod
    dt, dx = ctx.step('t', 'x')
    x = ctx.points('x')
    it, ix = ctx.indices('t', 'x')
    nt, nx = ctx.size('t', 'x')
    coeff = ctx.field('coeff')
    extra = ctx.extra

    def stencil_roll(q):
        return [
            mod.roll(q, shift=np.negative(s), axis=(0, 1))
            for s in [(0, 0), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)]
        ]

    u = transform_u(ctx.field('u'), extra, mod)
    u_st = stencil_roll(u)
    u, uxm, uxp, um, umxm, umxp = u_st

    u_t = (u - um) / dt
    um_xx = (umxm - 2 * um + umxp) / (dx**2)
    u_xx = (uxm - 2 * u + uxp) / (dx**2)
    u_xx = 0.5 * (u_xx + um_xx)
    um_x = (um - umxm) / dx
    u_x = (u - uxm) / dx
    u_x = 0.5 * (u_x + um_x)

    # Discretization.
    fu = u_t - coeff[0] * u_xx - coeff[1] + coeff[2] * u_x
    res = [fu[1:]]
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nt', type=int, default=64, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(plot_every=50, report_every=50, history_every=50)
    parser.set_defaults(frames=10)
    parser.set_defaults(optimizer='lbfgsb')
    parser.set_defaults(double=1)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(outdir='out_infer_constant_adv')
    return parser.parse_args()


def plot(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    uu_exact = extra.uu_exact

    # Print current values of coefficients.
    printlog('diff={:.5g}, src={:.5g}, vel={:.5g}'.format(
        *np.array(state.fields['coeff'].array)))

    path0 = "u_{:05d}.png".format(frame)
    printlog(path0)

    uu = domain.field(state, 'u')
    uu = np.array(transform_u(uu, extra, domain.mod))

    umax = max(abs(np.max(uu_exact)), abs(np.min(uu_exact)))
    odil.plot.plot_1d(domain,
                      uu_exact,
                      uu,
                      path=path0,
                      cmap='RdBu_r',
                      nslices=5,
                      transpose=True,
                      transparent=False,
                      umin=-umax,
                      umax=umax)


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx),
                         dimnames=('t', 'x'),
                         lower=(0, -1),
                         upper=(1, 1),
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_nlvl=args.nlvl)

    # Evaluate exact solution, boundary and initial conditions.
    tt, xx = domain.points()
    # Node-based in time, cell-based in space.
    t1 = domain.points('t', loc='n.')
    x1 = domain.points('x', loc='.c')
    uu_exact = get_exact(tt, xx)
    u_init = get_exact(x1 * 0 + domain.lower[0], x1)
    u_final = get_exact(x1 * 0 + domain.upper[0], x1)

    # Initial state.
    state = odil.State(fields={
        'coeff': [0, 0, 0.001],
        'u': odil.Field(None, loc='nc'),
    })
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.uu_exact = uu_exact
    extra.u_init = u_init
    extra.u_final = u_final
    extra.args = args

    problem = odil.Problem(operator_adv, domain, extra)
    return problem, state


def main():
    global problem, args

    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)

    callback = odil.make_callback(problem, args, plot_func=plot)
    odil.optimize(args, args.optimizer, problem, state, callback)

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
