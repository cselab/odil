#!/usr/bin/env python3

import numpy as np
import argparse
import os
import pickle
from odil import plotutil
import matplotlib.pyplot as plt


def load_csv(path):
    if path and os.path.isfile(path):
        return np.genfromtxt(path, delimiter=',', names=True)


parser = argparse.ArgumentParser()
parser.add_argument("--odil", default="out_odil/train.csv")
parser.add_argument("--odiln", default="out_odiln/train.csv")
parser.add_argument("--pinn", default="out_pinn/train.csv")
parser.add_argument("--out", type=str, default='heat_')
args = parser.parse_args()

u0 = load_csv(args.pinn)
u1 = load_csv(args.odil)
u2 = load_csv(args.odiln)

for key, name, ylbl in [
    ('error_u', 'u', 'temperature error'),
    ('error_k', 'k', 'conductivity error'),
]:
    fig, ax = plt.subplots(figsize=(1.5, 1.3))
    for u, lbl, c, m in [
        (u0, 'PINN, Adam', 'C0', 'o'),
        (u1, 'ODIL, Adam', 'C1', 's'),
        (u2, 'ODIL, Newton', 'C3', '^'),
    ]:
        if u is None:
            continue
        error = u[key]
        print("Last {} from {}: {:.6g} after {:.0f} iterations".format(
            key, lbl, error[-1], u['epoch'][-1]))
        ax.plot(u['epoch'], error, label=lbl, c=c)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylbl)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(10.**np.arange(0, 7))
    if name == 'u':
        yticks = 10.**np.arange(-3, 0.1)
    elif name == 'k':
        yticks = 10.**np.arange(-2, 1.1)
    ax.set_yticks(yticks)
    ax.set_ylim(min(yticks), max(yticks))
    plotutil.set_log_ticks(ax.xaxis)
    plotutil.set_log_ticks(ax.yaxis)
    plotutil.savefig(fig, args.out + 'train_' + name)
