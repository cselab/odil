#!/usr/bin/env python3

import argparse
import numpy as np
import odil
from odil import printlog
from odil import plotutil
import matplotlib.pyplot as plt
"""
Infers unknown constant parameters of an advection-diffusion equation
from the solution at the initial and final time.
"""


def get_ref_u(t, x, args):
    '''
    Returns an exact solution of the equation:
      u_t + c_vel * u_x = c_diff * u_xx + c_src
    '''
    t = np.array(t)
    x = np.array(x)
    u = np.zeros_like(x)
    nu = args.c_diff
    xx = x - t * args.c_vel
    ii = [1, 2, 3]
    for i in ii:
        k = 2 * i * np.pi
        u += np.cos(xx * k) * np.exp(-nu * k**2 * t)
    u /= 2 * len(ii)
    # Add uniform source.
    src = args.c_src
    u += src * t
    return u


def transform_u(u, extra, mod):
    # Impose initial and final conditions exactly.
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
    parser.add_argument(  #
        '--c_diff', type=float, default=0.01, help="Diffusivity")
    parser.add_argument(  #
        '--c_src', type=float, default=0.1, help="Uniform source")
    parser.add_argument(  #
        '--c_vel', type=float, default=0.2, help="Advection velocity")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(frames=3,
                        plot_every=50,
                        report_every=50,
                        history_every=10)
    parser.set_defaults(optimizer='lbfgsb')
    parser.set_defaults(multigrid=1)
    parser.set_defaults(double=1)
    parser.set_defaults(outdir='out_infer_constant')
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    ref_u = extra.ref_u

    state_u = domain.field(state, 'u')
    state_u = np.array(transform_u(state_u, extra, domain.mod))
    coeff = np.array(domain.field(state, 'coeff'))
    title = 'epoch={:}, diff={:.3g}, src={:.3g}, vel={:.3g}'.format(
        epoch, *coeff)
    umax = max(abs(np.max(ref_u)), abs(np.min(ref_u)))
    fig = odil.plot.plot_1d(domain,
                            ref_u,
                            state_u,
                            cmap='RdBu_r',
                            nslices=5,
                            title=title,
                            transpose=True,
                            transparent=False,
                            umin=-umax,
                            umax=umax)
    plotutil.savefig(fig, "u_{:05d}".format(frame), printf=printlog)
    plt.close(fig)


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    coeff = np.array(domain.field(state, 'coeff'))
    # Print current parameters.
    printlog('diff={:.5g}, src={:.5g}, vel={:.5g}'.format(*coeff))


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    coeff = np.array(domain.field(state, 'coeff'))
    # Add current parameters to `train.csv`.
    history.append('c_diff', coeff[0])
    history.append('c_src', coeff[1])
    history.append('c_vel', coeff[2])


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
    tone = domain.points_1d('t', loc='n')
    xone = domain.points_1d('x', loc='c')
    ref_u = get_ref_u(tt, xx, args)
    u_init = get_ref_u(xone * 0 + domain.lower[0], xone, args)
    u_final = get_ref_u(xone * 0 + domain.upper[0], xone, args)

    # Initial state.
    state = odil.State(
        fields={
            'coeff': odil.Array([0, 0, 0.001]),
            # None will be converted to a zero field.
            'u': odil.Field(None, loc='nc'),
        })
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.ref_u = ref_u
    extra.u_init = u_init
    extra.u_final = u_final
    extra.args = args

    problem = odil.Problem(operator_adv, domain, extra)
    return problem, state


def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(problem,
                                  args,
                                  plot_func=plot_func,
                                  report_func=report_func,
                                  history_func=history_func)
    odil.optimize(args, args.optimizer, problem, state, callback)


if __name__ == "__main__":
    main()
