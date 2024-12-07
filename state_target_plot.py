import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

parser = argparse.ArgumentParser(description="Input the variables for running the script.")
parser.add_argument("problem", type=int, choices=[1, 2, 3], help="Specify the problem number (1, 2, or 3)")
args = parser.parse_args()
problem = args.problem

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

impath = "./paper_plots/"


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=300, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


######################################################################################################################
# FOM
impath_ex = "./data/FOM/problem=" + str(problem) + "/"

qs = np.load(impath_ex + 'qs_org.npy')
qs_target = np.load(impath_ex + 'qs_target.npy')

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
im1 = ax1.pcolormesh(qs.T, cmap='YlOrRd')
ax1.axis('off')
ax1.axis('auto')
ax1.set_title(r"$Q$")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(122)
im2 = ax2.pcolormesh(qs_target.T, cmap='YlOrRd')
ax2.axis('off')
ax2.axis('auto')
ax2.set_title(r"$Q_\mathrm{t}$")
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im2, cax=cax, orientation='vertical')

fig.supylabel(r"time $t$")
fig.supxlabel(r"space $x$")

fig.savefig(impath + "State_target_p" + str(problem), dpi=300, transparent=True)
