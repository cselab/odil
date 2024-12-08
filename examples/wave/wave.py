#!/usr/bin/env python3

import argparse
import numpy as np
import pickle

import odil
from odil import plotutil
import matplotlib.pyplot as plt
from odil import printlog
from odil.runtime import tf


def get_exact(args, t, x):
    t = tf.Variable(t)
    x = tf.Variable(x)
    u = tf.zeros_like(x)
    with tf.GradientTape() as tape:
        ii = [1, 2, 3, 4, 5]
        for i in ii:
            k = i * np.pi
            u += tf.cos((x - t + 0.5) * k)
            u += tf.cos((x + t - 0.5) * k)
        u /= 2 * len(ii)
    ut = tape.gradient(u, t).numpy()
    u = u.numpy()
    return u, ut


def operator_wave(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args
    dt, dx = ctx.step()
    it, ix = ctx.indices()
    nt, nx = ctx.size()
    x = ctx.points('x')

    def stencil_var(key):
        st = [
            ctx.field(key),
            ctx.field(key, -1, 0),
            ctx.field(key, -2, 0),
            ctx.field(key, -1, -1),
            ctx.field(key, -1, 1)
        ]
        return st

    left_utm = mod.roll(extra.left_u, 1, axis=0)
    right_utm = mod.roll(extra.right_u, 1, axis=0)

    def apply_bc_u(st):
        extrap = odil.core.extrap_quadh
        st[3] = mod.where(  #
            ix == 0, extrap(st[4], st[1], left_utm[:, None]), st[3])
        st[4] = mod.where(  #
            ix == nx - 1, extrap(st[3], st[1], right_utm[:, None]), st[4])
        return st

    u_st = stencil_var('u')
    apply_bc_u(u_st)
    u, utm, utmm, uxm, uxp = u_st

    u_t_tm = (u - utm) / dt
    u_t_tmm = (utm - utmm) / dt
    u_t_tmm = mod.where(it == 1, extra.init_ut[None, :], u_t_tmm)

    u_tt = (u_t_tm - u_t_tmm) / dt
    u_xx = (uxm - 2 * utm + uxp) / (dx**2)

    fu = u_tt - u_xx

    u0 = extra.init_u + 0.5 * dt * extra.init_ut
    fu = mod.where(it == 0, (u - u0[None, :]) * args.kimp, fu)

    res = [('fu', fu)]

    return res


def get_uut(domain, init_u, uu):
    from odil.core import extrap_quad, extrap_quadh
    dt = domain.step('t')
    u = uu
    utm = np.roll(u, 1, axis=0)
    utp = np.roll(u, -1, axis=0)
    utm[0, :] = extrap_quadh(utp[0, :], u[0, :], init_u)
    utp[-1, :] = extrap_quad(u[-3, :], u[-2, :], u[-1, :])
    uut = (utp - utm) / (2 * dt)
    return uut


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nt', type=int, default=64, help="Grid size in t")
    parser.add_argument('--Nx', type=int, default=64, help="Grid size in x")
    parser.add_argument('--kimp',
                        type=float,
                        default=1,
                        help="Factor to impose initial conditions")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(double=1)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(outdir='out_wave')
    parser.set_defaults(linsolver='direct')
    parser.set_defaults(optimizer='lbfgsb')
    parser.set_defaults(lr=0.001)
    parser.set_defaults(plotext='png', plot_title=1)
    parser.set_defaults(plot_every=100,
                        report_every=10,
                        history_full=5,
                        history_every=10,
                        frames=2)
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None):
    from odil.plot import plot_1d

    domain = problem.domain
    extra = problem.extra
    mod = domain.mod
    args = extra.args

    title0 = "u epoch={:05d}".format(epoch) if args.plot_title else None
    title1 = "ut epoch={:05d}".format(epoch) if args.plot_title else None
    path0 = "u_{:05d}.{}".format(frame, args.plotext)
    path1 = "ut_{:05d}.{}".format(frame, args.plotext)
    printlog(path0, path1)

    ref_u, ref_ut = extra.ref_u, extra.ref_ut

    state_u = np.array(domain.field(state, 'u'))
    state_ut = get_uut(domain, extra.init_u, state_u)

    if args.dump_data:
        path = "data_{:05d}.pickle".format(frame)
        d = dict()
        d['upper'] = domain.upper
        d['lower'] = domain.lower
        d['cshape'] = domain.cshape
        d['state_u'] = state_u
        d['state_ut'] = state_ut
        d['ref_u'] = ref_u
        d['ref_ut'] = ref_ut
        d = odil.core.struct_to_numpy(mod, d)
        with open(path, 'wb') as f:
            pickle.dump(d, f)

    umax = max(abs(np.max(ref_u)), abs(np.min(ref_u)))
    plot_1d(domain,
            extra.ref_u,
            state_u,
            path=path0,
            title=title0,
            cmap='RdBu_r',
            nslices=5,
            transpose=True,
            umin=-umax,
            umax=umax)

    umax = max(abs(np.max(ref_ut)), abs(np.min(ref_ut)))
    plot_1d(domain,
            ref_ut,
            state_ut,
            path=path1,
            title=title1,
            cmap='RdBu_r',
            nslices=5,
            transpose=True,
            umin=-umax,
            umax=umax)


def get_error(domain, extra, state, key):
    args = extra.args
    mod = domain.mod
    if key == 'u':
        state_u = domain.field(state, key)
        ref_u = extra.ref_u
        return np.sqrt(np.mean((state_u - ref_u)**2))
    return None


def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    extra = problem.extra
    for key in ['u', 'k']:
        error = get_error(domain, extra, state, key)
        if error is not None:
            history.append('error_' + key, error)


def report_func(problem, state, epoch, cbinfo):
    domain = problem.domain
    extra = problem.extra
    res = dict()
    for key in ['u', 'k']:
        error = get_error(domain, extra, state, key)
        if error is not None:
            res[key] = error
    printlog('error: ' + ', '.join('{}:{:.5g}'.format(*item)
                                   for item in res.items()))


def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(cshape=(args.Nt, args.Nx),
                         dimnames=('t', 'x'),
                         lower=(0, -1),
                         upper=(1, 1),
                         multigrid=args.multigrid,
                         dtype=dtype)
    if domain.multigrid:
        printlog('multigrid levels:', domain.mg_cshapes)
    mod = domain.mod
    # Evaluate exact solution, boundary and initial conditions.
    tt, xx = domain.points()
    t1, x1 = domain.points_1d()
    ref_u, ref_ut = get_exact(args, tt, xx)
    left_u, _ = get_exact(args, t1, t1 * 0 + domain.lower[1])
    right_u, _ = get_exact(args, t1, t1 * 0 + domain.upper[1])
    init_u, init_ut = get_exact(args, x1 * 0 + domain.lower[0], x1)

    extra = argparse.Namespace()

    def add_extra(d, *keys):
        for key in keys:
            setattr(extra, key, d[key])

    add_extra(locals(), 'args', 'ref_u', 'ref_ut', 'left_u', 'right_u',
              'init_u', 'init_ut')

    state = odil.State()
    state.fields['u'] = np.zeros(domain.cshape)
    state = domain.init_state(state)
    problem = odil.Problem(operator_wave, domain, extra)
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

    with open('done', 'w') as f:
        pass


if __name__ == "__main__":
    main()
