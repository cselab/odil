import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import os

import numpy as np


def plot_1d(
    domain,
    u_ref,
    u_state,
    path=None,
    title=None,
    umin=None,
    umax=None,
    slice_lim=0.1,
    transpose=False,
    invertx=False,
    nslices=6,
    dpi=300,
    transparent=True,
    figsize=(3, 2.5),
    aspect='auto',
    callback=None,
    interpolation='nearest',
    cmap=None,
    cref='C2',
    cstate='C0',
):
    if transpose:
        # Index zero drawn as vertical, rotate by 90 degrees counterclockwise.
        ix = 1
        iy = 0
        u_ref = u_ref.T
        u_state = u_state.T
    else:
        # Index zero drawn as horizontal.
        ix = 0
        iy = 1
    extent = [
        domain.lower[ix], domain.upper[ix], domain.lower[iy], domain.upper[iy]
    ]
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)
    spec = fig.add_gridspec(2 * nslices, 3)
    xx, yy = domain.points_1d(ix, iy)
    xx, yy = np.array(xx), np.array(yy)
    xlim = (domain.lower[ix], domain.upper[ix])
    ylim = (domain.lower[iy], domain.upper[iy])
    if umin is None:
        umin = u_ref.min()
    if umax is None:
        umax = u_ref.max()
    if cmap is None:
        cmap = 'viridis'
    ulim = (umin, umax)
    ptp = umax - umin
    slim = (umin - ptp * slice_lim, umax + ptp * slice_lim)
    if title is not None:
        fig.suptitle(title, fontsize=8)
    axes = [None, None]
    for data, i in (u_state, 0), (u_ref, 1):
        axes[i] = fig.add_subplot(spec[1:-1, i])
        ax = axes[i]
        ax.spines[:].set_visible(True)
        ax.spines[:].set_linewidth(0.25)
        ax.imshow(data.T,
                  interpolation=interpolation,
                  cmap=cmap,
                  vmin=ulim[0],
                  vmax=ulim[1],
                  extent=extent,
                  origin='lower',
                  aspect=aspect)
        if callback is not None:
            callback(i, fig, ax, data, extent)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if invertx:
            ax.invert_xaxis()

    shift = 0.22
    spec = fig.add_gridspec(2 * nslices, 3, left=shift)
    for i in range(nslices):
        yslice = i * (domain.cshape[iy] - 1) // max(1, nslices - 1)
        ns = nslices - 1 - i
        ax = fig.add_subplot(spec[2 * ns:2 * ns + 2, 2])
        ax.spines[:].set_visible(True)
        ax.spines[:].set_linewidth(0.25)
        l0, = ax.plot(xx,
                      u_ref[:, yslice],
                      c=cref,
                      ls='-',
                      label="reference",
                      linewidth=0.9)
        l1, = ax.plot(xx,
                      u_state[:, yslice],
                      c=cstate,
                      ls='-',
                      label="inferred",
                      linewidth=0.6)
        # ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(slim)
        if invertx:
            ax.invert_xaxis()
        ax.arrow(-0.025,
                 0.5,
                 -0.05,
                 0,
                 overhang=0,
                 head_width=0.05,
                 head_length=0.03,
                 linewidth=0.5,
                 transform=ax.transAxes,
                 facecolor='black',
                 clip_on=False)
    ax.legend(handles=[l1, l0],
              loc=(-2.15 - shift, 0.5),
              columnspacing=2.2,
              ncol=2,
              frameon=False,
              handletextpad=0.5,
              fontsize=7)

    if path is not None:
        fig.savefig(path, dpi=dpi, pad_inches=0.01, transparent=transparent)
        plt.close(fig)
    else:
        return fig


def plot_2d(
    domain,
    exact_uu,
    pred_uu,
    slices_it,
    path,
    title=None,
    umin=None,
    umax=None,
    dpi=300,
    figsizey=3.,
    hspace=0.05,
    cmap=None,
    callback=None,
    xlabel='{:.2f}',
    ylabel_exact='reference',
    ylabel_pred='inferred',
    transparent=False,
    interpolation='nearest',
):
    '''
    exact_uu: `array`
        3D fields (t, x, y) with reference solution.
    pred_uu: `array`
        3D fields (t, x, y) with inferred solution.
    slices_it: `list`
        List of indices along direction t
    '''
    nslices = len(slices_it)
    figsize = (figsizey * nslices * 0.5, figsizey)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=hspace)
    spec = fig.add_gridspec(2, nslices)
    extent = [
        domain.lower[1], domain.upper[1], domain.lower[2], domain.upper[2]
    ]
    if title:
        fig.suptitle(title)
    tt = domain.cell_center_1d(0)
    axes = [[None] * nslices for i in range(2)]
    for j in range(nslices):
        it = slices_it[j]
        for i, data in enumerate((exact_uu[it], pred_uu[it])):
            ax = fig.add_subplot(spec[i, j])
            axes[i][j] = ax
            ax.spines[['left', 'right', 'bottom', 'top']].set_visible(True)
            ax.spines[['left', 'right', 'bottom', 'top']].set_linewidth(0.25)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:4])
            ax.imshow(data.T,
                      interpolation=interpolation,
                      cmap=cmap,
                      vmin=umin,
                      vmax=umax,
                      extent=extent,
                      origin='lower',
                      aspect='equal')
            if i == 1:
                if xlabel:
                    ax.set_xlabel(xlabel.format(tt[it]))
            if j == 0 and i == 0:
                if ylabel_exact:
                    ax.set_ylabel(ylabel_exact)
            if j == 0 and i == 1:
                if ylabel_pred:
                    ax.set_ylabel(ylabel_pred)

            if callback:
                callback(i, j, ax, fig)

    fig.savefig(path,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.01,
                transparent=transparent)
    plt.close(fig)


g_colormap_names = [
    "rainbow",
    "coolwarm",
    "yellow",
    "geo",
]


def get_colormap_data(name):
    data = None
    assert name in g_colormap_names
    if name == "rainbow":
        data = [
            0.0, 0.278431372549, 0.278431372549, 0.858823529412, 0.143, 0.0,
            0.0, 0.360784313725, 0.285, 0.0, 1.0, 1.0, 0.429, 0.0,
            0.501960784314, 0.0, 0.571, 1.0, 1.0, 0.0, 0.714, 1.0,
            0.380392156863, 0.0, 0.857, 0.419607843137, 0.0, 0.0, 1.0,
            0.878431372549, 0.301960784314, 0.301960784314
        ]
    elif name == "coolwarm":
        data = [
            0.0, 0.0, 0.0, 0.34902, 0.03125000000000003, 0.039216, 0.062745,
            0.380392, 0.06250000000000006, 0.062745, 0.117647, 0.411765,
            0.09374999999999994, 0.090196, 0.184314, 0.45098,
            0.12499999999999997, 0.12549, 0.262745, 0.501961, 0.15625,
            0.160784, 0.337255, 0.541176, 0.18750000000000003, 0.2, 0.396078,
            0.568627, 0.21875000000000006, 0.239216, 0.454902, 0.6,
            0.24999999999999994, 0.286275, 0.521569, 0.65098,
            0.28124999999999994, 0.337255, 0.592157, 0.701961, 0.3125,
            0.388235, 0.654902, 0.74902, 0.34375, 0.466667, 0.737255, 0.819608,
            0.37500000000000006, 0.572549, 0.819608, 0.878431, 0.40625,
            0.654902, 0.866667, 0.909804, 0.4375, 0.752941, 0.917647, 0.941176,
            0.46875, 0.823529, 0.956863, 0.968627, 0.5, 0.988235, 0.960784,
            0.901961, 0.5, 0.941176, 0.984314, 0.988235, 0.52, 0.988235,
            0.945098, 0.85098, 0.54, 0.980392, 0.898039, 0.784314, 0.5625,
            0.968627, 0.835294, 0.698039, 0.59375, 0.94902, 0.733333, 0.588235,
            0.625, 0.929412, 0.65098, 0.509804, 0.65625, 0.909804, 0.564706,
            0.435294, 0.6875, 0.878431, 0.458824, 0.352941, 0.71875, 0.839216,
            0.388235, 0.286275, 0.7500000000000001, 0.760784, 0.294118,
            0.211765, 0.78125, 0.701961, 0.211765, 0.168627, 0.8125, 0.65098,
            0.156863, 0.129412, 0.84375, 0.6, 0.094118, 0.094118, 0.875,
            0.54902, 0.066667, 0.098039, 0.9062500000000001, 0.501961, 0.05098,
            0.12549, 0.9375, 0.45098, 0.054902, 0.172549, 0.96875, 0.4,
            0.054902, 0.192157, 1.0, 0.34902, 0.070588, 0.211765
        ]
    elif name == "geo":
        data = [
            0.0, 0.0, 0.6039215686274509, 0.8705882352941177, 0.5, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.12156862745098039, 0.3568627450980392
        ]
    elif name == "yellow":
        data = [
            0.0, 1.0, 1.0, 0.988235, 0.002, 1.0, 1.0, 0.988235,
            0.05000000000000001, 0.984314, 0.988235, 0.843137,
            0.10000000000000002, 0.988235, 0.988235, 0.741176, 0.15, 0.980392,
            0.968627, 0.654902, 0.20000000000000004, 0.980392, 0.945098,
            0.576471, 0.25, 0.968627, 0.905882, 0.486275, 0.3, 0.968627,
            0.862745, 0.388235, 0.3499999999999999, 0.960784, 0.803922,
            0.286275, 0.4000000000000001, 0.94902, 0.741176, 0.219608, 0.45,
            0.941176, 0.678431, 0.14902, 0.5, 0.929412, 0.607843, 0.094118,
            0.55, 0.921569, 0.545098, 0.054902, 0.6, 0.909804, 0.486275,
            0.035294, 0.65, 0.890196, 0.411765, 0.019608, 0.6999999999999998,
            0.8, 0.305882, 0.0, 0.7500000000000001, 0.760784, 0.239216, 0.0,
            0.8000000000000002, 0.678431, 0.180392, 0.011765, 0.85, 0.6,
            0.121569, 0.023529, 0.9, 0.501961, 0.054902, 0.031373, 0.95, 0.4,
            0.039216, 0.058824, 1.0, 0.301961, 0.047059, 0.090196
        ]
    else:
        assert False, "Unknown colormap=" + name
    data = np.reshape(data, (-1, 4))
    return data


def get_cmap(name):
    data = get_colormap_data(name)
    nodes = data[:, 0]
    colors = data[:, 1:]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name, list(zip(nodes, colors)))
    return cmap
