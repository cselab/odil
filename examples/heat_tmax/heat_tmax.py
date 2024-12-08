#!/usr/bin/env python3

import argparse
import numpy as np
import odil
from odil import printlog
from odil import plotutil
import matplotlib.pyplot as plt


def get_ref_u(t, x, args):
    '''
    Returns an exact solution of the equation:
      u_t = u_xx
    '''
    t = np.array(t) * args.tmax_ref
    x = np.array(x)
    u = np.sin(x) * np.exp(-t)
    return u


def transform_u(u, extra, mod):
    # Impose initial conditions exactly.
    u = mod.concatenate([extra.u_init[None, :], u[1:]], axis=0)
    return u


def operator_heat(ctx):
    mod = ctx.mod
    dt, dx = ctx.step('t', 'x')
    x = ctx.points('x')
    it, ix = ctx.indices('t', 'x', loc='nc')
    nt, nx = ctx.size('t', 'x')
    coeff = ctx.field('coeff')
    extra = ctx.extra
    args = extra.args

    def roll(u, shift):
        return mod.roll(u, shift, (0, 1))

    def stencil(name):
        # Offsets along t and x.
        offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)]
        # Shifted source fields, using ctx.field() to support Newton.
        fields = [ctx.field(name, *otx) for otx in offsets]
        # Cancel the shift, transform the field, and shift back.
        fields = [
            roll(transform_u(roll(u, otx), extra, mod), np.negative(otx))
            for u, otx in zip(fields, offsets)
        ]
        return fields

    u, uxm, uxp, um, umxm, umxp = stencil('u')

    # Impose zero boundary conditions.
    uxm = mod.where(ix == 0, -u, uxm)
    uxp = mod.where(ix == nx - 1, -u, uxp)
    umxm = mod.where(ix == 0, -um, umxm)
    umxp = mod.where(ix == nx - 1, -um, umxp)

    # Apply tmax.
    dt *= coeff[0]

    u_t = (u - um) / dt
    um_xx = (umxm - 2 * um + umxp) / (dx**2)
    u_xx = (uxm - 2 * u + uxp) / (dx**2)
    u_xx = 0.5 * (u_xx + um_xx)

    # Discretization.
    fu = u_t - u_xx
    fu = mod.where(it == 0, ctx.cast(0), fu)  # Valid values are for it > 0.
    res = [('eqn', fu)]

    # Impose a value at t=tmax.
    ixc = nx // 2
    res += [('imp', args.kimp * (u[-1, ixc] - extra.u_final[ixc]))]
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Nt', type=int, default=64, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    parser.add_argument('--kimp', type=float, default=1)
    parser.add_argument('--tmax_ref', type=float, default=4.5)
    parser.add_argument('--tmax_init', type=float, default=1)
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(frames=4,
                        plot_every=1000,
                        report_every=1000,
                        history_every=200)
    parser.set_defaults(optimizer='lbfgsb')
    parser.set_defaults(multigrid=1)
    parser.set_defaults(double=1)
    parser.set_defaults(echo=1)
    parser.set_defaults(outdir='out_heat_tmax')
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    ref_u = extra.ref_u

    state_u = domain.field(state, 'u')
    state_u = np.array(transform_u(state_u, extra, domain.mod))
    coeff = np.array(domain.field(state, 'coeff'))

    ixc = domain.size('x') // 2
    title = 'epoch={:}, tmax={:.8g}\nu(pi/2, tmax) / u(pi/2, 0) = {:.5g}'.format(
        epoch, coeff[0], state_u[-1, ixc] / state_u[0, ixc])
    umax = np.max(ref_u)
    fig = odil.plot.plot_1d(domain,
                            ref_u,
                            state_u,
                            cmap='Spectral_r',
                            nslices=5,
                            title=title,
                            transpose=True,
                            transparent=False,
                            interpolation='none',
                            umin=0,
                            umax=umax)
    plotutil.savefig(fig, "u_{:05d}".format(frame), printf=printlog)
    plt.close(fig)


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    coeff = np.array(domain.field(state, 'coeff'))
    # Print current parameters.
    printlog('tmax={:.5g}'.format(*coeff))


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    coeff = np.array(domain.field(state, 'coeff'))
    # Add current parameters to `train.csv`.
    history.append('tmax', coeff[0])


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx),
                         dimnames=('t', 'x'),
                         lower=(0, 0),
                         upper=(1, np.pi),
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_nlvl=args.nlvl)

    # Evaluate exact solution, boundary and initial conditions.
    tt, xx = domain.points(loc='nc')
    # Node-based in time, cell-based in space.
    tone = domain.points_1d('t', loc='n')
    xone = domain.points_1d('x', loc='c')
    ref_u = get_ref_u(tt, xx, args)
    u_init = get_ref_u(np.full_like(xone, domain.lower[0]), xone, args)
    u_final = get_ref_u(np.full_like(xone, domain.upper[0]), xone, args)

    # Initial state.
    state = odil.State(
        fields={
            'u': odil.Field(np.tile(u_init, [args.Nt + 1, 1]), loc='nc'),
            'coeff': odil.Array([args.tmax_init]),
        })
    state = domain.init_state(state)

    extra = argparse.Namespace()
    extra.ref_u = ref_u
    extra.u_init = u_init
    extra.u_final = u_final
    extra.args = args

    problem = odil.Problem(operator_heat, domain, extra)
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
