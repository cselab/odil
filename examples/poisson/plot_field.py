#!/usr/bin/env python3

import numpy as np
import pickle
import argparse
import math
from odil import plotutil
import matplotlib.pyplot as plt
plotutil.set_extlist(['png'])


def plot_field(path, u_odil, u_ref, vmax):
    extent = [0, 1, 0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    for ax, u, title in zip(axes, [u_odil, u_ref], ['ODIL', 'reference']):
        ax.spines[:].set_visible(True)
        ax.spines[:].set_linewidth(0.25)
        ax.imshow(u.T,
                  interpolation='bilinear',
                  cmap='PuOr_r',
                  vmin=-vmax,
                  vmax=vmax,
                  extent=extent,
                  origin='lower',
                  aspect='equal')
        ax.set_title(title, y=0.97)
        ax.set_xticks([])
        ax.set_yticks([])
    plotutil.savefig(fig, path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='out_poisson/data.pickle')
    parser.add_argument("--out", type=str, default='out_poisson/field')
    parser.add_argument("--vmax", type=float, default=1)
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data = pickle.load(f)
        u_odil = data['u']
        u_ref = data['ref_u']

    plot_field(args.out, u_odil, u_ref, vmax=args.vmax)


if __name__ == "__main__":
    main()
