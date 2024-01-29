#!/usr/bin/env python3

import argparse
import numpy as np
import odil
from odil import printlog
from odil import plotutil
import matplotlib.pyplot as plt


def u_init_smooth(x, y):
    r = x**2 + y**2
    res = np.exp(1 - 1 / (1 - r))
    res = np.where(r < 1, res, 0)
    return res


def u_init_blob(x, y, t):
    # Single blob advected by uniform velocity field.
    u0 = 0.2
    v0 = 0.2
    r0 = 0.2
    dx = x - u0 * t - 0.3
    dy = y - v0 * t - 0.3
    k = 1 + t
    dx *= k
    dy /= k
    res = np.maximum(0, 1 - (dx**2 + dy**2) / r0**2)
    res = res**0.2
    return res


def operator_advection(ctx):
    mod = ctx.mod
    extra = ctx.extra
    args = extra.args
    dt, dx, dy = ctx.step()
    x, y = ctx.points('x', 'y', loc='ncc')
    it, ix, iy = ctx.indices(loc='ncc')
    nt, ny, ny = ctx.size()

    def single_var(key, shift_t=0, shift_x=0, shift_y=0, frozen=False):
        u = ctx.field(key, shift_t, shift_x, shift_y, frozen=frozen)
        return u

    def stencil_var(key, shift_t=0, frozen=False):
        st = [
            ctx.field(key, shift_t, 0, 0, frozen=frozen),
            ctx.field(key, shift_t, -1, 0, frozen=frozen),
            ctx.field(key, shift_t, 1, 0, frozen=frozen),
            ctx.field(key, shift_t, 0, -1, frozen=frozen),
            ctx.field(key, shift_t, 0, 1, frozen=frozen),
        ]
        return st

    def laplace(st):
        q, qxm, qxp, qym, qyp = st
        q_xx = (qxp - 2 * q + qxm) / dx**2
        q_yy = (qyp - 2 * q + qym) / dy**2
        q_lap = q_xx + q_yy
        return q_lap

    def deriv_fou(um, u, up, v):
        '''
        Returns first-order derivative at i
        given tracer fields um=u(i-1), u=u(i), etc
        and velocity field v=v(i)
        '''
        # | um | u | up |
        du = mod.where(
            v > 0,
            u - um,
            mod.where(v < 0, up - u, (up - um) * 0.5),
        )
        return du

    # Face velocity.
    vx_st = stencil_var('vx')
    vxf_st = stencil_var('vx', frozen=True)
    vx = vx_st[0]
    vxf = vxf_st[0]
    vxmh = vx
    vxph = vx
    vxmhf = (vxf + mod.roll(vxf, shift=1, axis=1)) * 0.5
    vxphf = (vxf + mod.roll(vxf, shift=-1, axis=1)) * 0.5

    vy_st = stencil_var('vy')
    vyf_st = stencil_var('vy', frozen=True)
    vy = vy_st[0]
    vyf = vyf_st[0]
    vymh = vy
    vyph = vy
    vymhf = (vyf + mod.roll(vyf, shift=1, axis=2)) * 0.5
    vyphf = (vyf + mod.roll(vyf, shift=-1, axis=2)) * 0.5

    st = stencil_var('u', shift_t=-1)
    u_x = deriv_fou(st[1], st[0], st[2], vxf)
    u_y = deriv_fou(st[3], st[0], st[4], vyf)
    vu_x = vx * u_x / dx
    vu_y = vy * u_y / dy

    u = single_var('u')
    um = st[0]
    um = mod.where(it == 1, extra.u_init[None, :], um)
    # Time derivative.
    u_t = (u - um) / dt

    # Advection equation.
    fu = u_t + vu_x + vu_y
    fu = mod.where(it == 0, (u - extra.u_init[None, :]) / dx, fu)

    zero = ctx.cast(0)

    # Imposed tracer values.
    fimp = mod.where(it == nt - 1, (u - extra.u_final[None, :]) / dx, zero)

    res = [fu, fimp * args.kimp]

    # Velocity regualization.
    if args.kxreg:
        freg_vx = laplace(vx_st) * args.kxreg
        freg_vy = laplace(vy_st) * args.kxreg
        res += [freg_vx, freg_vy]

    if args.ktreg:
        k = args.ktreg / dt
        ftreg_vx = (single_var('vx') - single_var('vx', -1)) * k
        ftreg_vx = mod.where(it == 0, zero, ftreg_vx)
        ftreg_vy = (single_var('vy') - single_var('vy', -1)) * k
        ftreg_vy = mod.where(it == 0, zero, ftreg_vy)
        res += [ftreg_vx, ftreg_vy]

    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nt', type=int, default=None, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    parser.add_argument('--Ny', type=int, default=None, help="Grid size in y")
    parser.add_argument('--kxreg',
                        type=float,
                        default=0.01,
                        help="Laplacian regularization weight")
    parser.add_argument('--ktreg',
                        type=float,
                        default=1,
                        help="Time regularization weight")
    parser.add_argument('--kimp',
                        type=float,
                        default=10,
                        help="Imposed values weight")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)

    parser.set_defaults(outdir='out_veltracer')
    parser.set_defaults(frames=5)
    parser.set_defaults(plot_every=100, report_every=100, history_every=10)
    parser.set_defaults(optimizer='adam')
    parser.set_defaults(lr=0.01)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(mg_interp='conv')

    # Relevant if using optimizer=newton.
    parser.set_defaults(linsolver='multigrid')
    parser.set_defaults(linsolver_maxiter=10)
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    extra = problem.extra
    args = extra.args
    path0 = "u_{:05d}.png".format(frame)
    path1 = "vx_{:05d}.png".format(frame)
    path2 = "vy_{:05d}.png".format(frame)
    printlog(path0)

    slices_it = np.linspace(0, domain.cshape[0], 5, dtype=int)
    slices_t = domain.points_1d(0, loc='n')[slices_it]

    state_u = domain.field(state, 'u')
    state_vx = domain.field(state, 'vx')
    state_vy = domain.field(state, 'vy')

    def callback_quiver(i, j, ax, fig):
        plt.setp(ax.spines.values(), linewidth=0.25)
        ax.yaxis.label.set_size(7)
        xx, yy = domain.points('x', 'y', loc='.cc')
        skip = domain.cshape[1] // 8
        offset = skip // 2 - 1
        x = np.array(xx[offset::skip, offset::skip]).flatten()
        y = np.array(yy[offset::skip, offset::skip]).flatten()
        vx = state_vx
        vy = state_vy
        vx = np.array(vx[slices_it[j], offset::skip, offset::skip]).flatten()
        vy = np.array(vy[slices_it[j], offset::skip, offset::skip]).flatten()
        ax.quiver(x, y, vx, vy, scale=5, color='k')

    odil.plot.plot_2d(domain,
                      extra.exact_uu,
                      state_u,
                      slices_it,
                      slices_t,
                      path0,
                      cmap='YlOrBr',
                      umin=0,
                      umax=1,
                      callback=callback_quiver,
                      interpolation='bilinear',
                      title="epoch={:}".format(epoch))
    odil.plot.plot_2d(domain,
                      state_vx,
                      state_vy,
                      slices_it,
                      slices_t,
                      path1,
                      umin=-0.5,
                      umax=0.5,
                      cmap='PuOr_r',
                      interpolation='bilinear',
                      ylabel_exact='vx',
                      ylabel_pred='vy')


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx, args.Ny),
                         dimnames=('t', 'x', 'y'),
                         lower=(0, 0, 0),
                         upper=(1, 1, 1),
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_nlvl=args.nlvl)
    if domain.multigrid:
        printlog('multigrid levels:', domain.mg_cshapes)

    # Evaluate exact solution, boundary and initial conditions.
    x, y = domain.points('x', 'y', loc='.cc')
    u_init = u_init_blob(x, y, 0)
    u_final = u_init_blob(x, y, 1)

    # Initial state.
    state = odil.State()
    # loc='ncc' defines the location of field values along (t,x,y) axes:
    # node-centered (n) in t, and cell-centered (c) in x,y.
    state.fields['u'] = odil.Field(None, loc='ncc')
    state.fields['vx'] = odil.Field(None, loc='ncc')
    state.fields['vy'] = odil.Field(None, loc='ncc')
    state = domain.init_state(state)

    # Reference solution.
    exact_uu = np.zeros(domain.get_field_shape(loc='ncc'))
    exact_uu[0] = u_init
    exact_uu[-1] = u_final

    extra = argparse.Namespace()
    extra.u_init = u_init
    extra.u_final = u_final
    extra.exact_uu = exact_uu
    extra.args = args
    problem = odil.Problem(operator_advection, domain, extra)
    return problem, state


def main():
    args = parse_args()
    args.Nt = args.Nt or args.Nx
    args.Ny = args.Ny or args.Nx
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(problem, args, plot_func=plot_func)
    odil.optimize(args, args.optimizer, problem, state, callback)


if __name__ == "__main__":
    main()
