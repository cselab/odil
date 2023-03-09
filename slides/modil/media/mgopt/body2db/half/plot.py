#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
from argparse import Namespace
from glob import glob
import os
import re

def savefig(fig, path, **kwargs):
    for ext in [
            #'pdf',
            'svg',
    ]:
        metadata = {
            'Date': None
        } if ext == 'svg' else {
            'DateModified': None
        } if ext == 'pdf' else {}
        p = path + '.' + ext
        print(p)
        fig.savefig(p, metadata=metadata, **kwargs)

def savelegend(fig, ax, path, **kwargs):
    figleg, axleg = plt.subplots()
    handles, labels = ax.get_legend_handles_labels()
    legend = axleg.legend(handles, labels, loc='center', frameon=False)
    axleg.set_axis_off()
    figleg.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    savefig(figleg, path, bbox_inches=bbox, **kwargs)

def calc_lossroot(u):
    res = np.empty(u.shape, dtype=u.dtype.descr + [('lossroot', float)])
    for name in u.dtype.names:
        res[name] = u[name]
    res['lossroot'] = res['loss']**0.5
    return res

def calc_epochtime(u, width=3):
    t = u['walltime']
    e = u['epoch']
    width = min(len(t), width)
    tm = np.hstack([[np.nan] * width, t[:-width]])
    em = np.hstack([[np.nan] * width, e[:-width]])
    res = np.empty(u.shape, dtype=u.dtype.descr + [('epochtime', float)])
    for name in u.dtype.names:
        res[name] = u[name]
    # Time of one epoch in milliseconds.
    res['epochtime'] = (t - tm) / (e - em) * 1000
    return res


def plot(lines, args):
    uu = [np.genfromtxt(line[0], delimiter=',', names=True) for line in lines]
    uu = [calc_lossroot(u) for u in uu]
    uu = [calc_epochtime(u, width=args.timewidth) for u in uu]
    for key, name, ylbl in [
        ('lossroot', 'loss', 'loss square root'),
        ('vx_err_l2', 'error', 'velocity error'),
        ('chi_err_l2', 'error_chi', 'body fraction error'),
        #('epochtime', 'epochtime', 'epoch time [ms]'),
    ]:
        fig, ax = plt.subplots()
        for i, line in enumerate(lines):
            u = uu[i]
            kwargs = {'label': line[1]}
            if len(line) > 2:
                kwargs['c'], kwargs['ls'] = line[2][:2], line[2][2:]
            if key not in u.dtype.names:
                print("skip unknown key '{}' for '{}'".format(key, line[1]))
                ax.plot([], [], **kwargs)
            else:
                ax.plot(u['epoch'] + 1, u[key], **kwargs)
        ax.set_xlabel('epoch')
        ax.set_xscale('log')
        ax.set_xticks(10 ** np.arange(0, 5.1))
        locmin = matplotlib.ticker.LogLocator(base=10.,
                                              subs=np.arange(0.1, 0.99, 0.1),
                                              numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        ax.set_ylabel(ylbl)
        if name == 'epochtime':
            ax.set_ylim(bottom=0)
        elif name in ['error']:
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
        elif name in ['loss']:
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 100)
        else:
            ax.set_yscale('log')
        savefig(fig, os.path.join(args.dir, 'train_' + name))
        plt.close(fig)
    savelegend(fig, ax, 'train_leg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        type=str,
                        default='.',
                        nargs='?',
                        help="Base directory")
    parser.add_argument('--timewidth',
                        type=int,
                        default=5,
                        help="Epoch time timewidth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lines = [
        ('bfgs/train.csv', 'L-BFGS (inverse)'),
        ('bfgs_forw/train.csv', 'L-BFGS (forward)'),
        ('ref/train.csv', 'Newton (forward)'),
    ]
    plot(lines, args)
