#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import argparse
from argparse import Namespace
from glob import glob
import os
import re


def savefig(fig, path, **kwargs):
    for ext in ['svg']:
        p = path + '.' + ext
        print(p)
        fig.savefig(p, **kwargs)


def calc_loss(u):
    s = np.zeros_like(u['loss'])
    for i in range(10):
        k = 'loss' + str(i)
        if k in u.dtype.names:
            s += u[k]**2
    u['loss'] = s**0.5
    return u


def calc_epochtime(u, width=3):
    t = u['walltime']
    e = u['epoch']
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
    uu = [u[1:] for u in uu]
    uu = [calc_loss(u) for u in uu]
    uu = [calc_epochtime(u, width=args.timewidth) for u in uu]
    for key, name, ylbl in [
        ('loss', 'loss', 'loss'),
        ('epochtime', 'epochtime', 'epoch time [ms]'),
    ]:
        if key not in uu[0].dtype.names:
            print("skip unknown key='{}'".format(key))
            continue
        fig, ax = plt.subplots()
        for i, line in enumerate(lines):
            lbl = line[1]
            u = uu[i]
            y = u[key]
            #if "129" in line[1] and key == 'loss': y *= 4
            ax.plot(u['epoch'], y, label=lbl)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylbl)
        if name == 'epochtime':
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale('log')
        ax.legend(loc='upper left', bbox_to_anchor=(1., 1.))
        savefig(fig, os.path.join(args.dir, 'train_' + name))
        plt.close(fig)


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
        ('nx257_lvl8/train.csv', 'N=257, 8 levels'),
        ('nx257_lvl7/train.csv', 'N=257, 7 levels'),
        ('nx257_lvl6/train.csv', 'N=257, 6 levels'),
        ('nx257_lvl5/train.csv', 'N=257, 5 levels'),
        ('nx129_lvl7/train.csv', 'N=129, 7 levels'),
    ]
    plot(lines, args)
