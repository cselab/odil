#!/usr/bin/env python3

import numpy as np
import os
import argparse
from odil import plotutil
import matplotlib.pyplot as plt
plotutil.set_extlist(['png'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="out_poisson/train.csv")
    parser.add_argument("--out", type=str, default='out_poisson/train')
    parser.add_argument("--vmax", type=float, default=1)
    args = parser.parse_args()
    odil = np.genfromtxt(args.data, delimiter=',', names=True)
    fig, ax = plt.subplots(figsize=(1.5, 1.3))
    ax.plot(odil['epoch'] + 1, odil['error_u'], label='ODIL', c='C1')
    # x-axis
    ax.set_xlabel('epoch')
    ax.set_xscale('log')
    ax.set_xticks(10**np.arange(0, 4.1, 1))
    # y-axis
    ax.set_ylabel(r'error')
    ax.set_yscale('log')
    vmin = -3 if odil['error_u'].min() < 1e-2 else -2
    ax.set_ylim(10**vmin, 10)
    ax.set_yticks(10**np.arange(vmin, 1.1))
    plotutil.set_log_ticks(ax.yaxis)
    plotutil.apply_clip_box(ax, ax.lines, upper=(1.05, 1.05))
    plotutil.savefig(fig, args.out)
