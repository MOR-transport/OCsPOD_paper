import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import re

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
os.makedirs(impath, exist_ok=True)

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


# Plotting the modes vs cost functional plot
immpath_FOM = "./data/FOM/problem=" + str(problem) + "/"
immpath_PODG = "./data/PODG/problem=" + str(problem) + "/"
immpath_sPODG = "./data/sPODG/problem=" + str(problem) + "/"

if problem == 1:
    PODG_modes = np.asarray([4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40])
    sPODG_modes = np.asarray([4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40])
else:
    PODG_modes = np.asarray(
        [4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    sPODG_modes = np.asarray([4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40])

file_J = "J_opt_FOM_list.npy"

PODG_J = []
sPODG_J = []

# Regex pattern to extract real numbers from file names
number_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

# Function to extract the number from a string (like a file name)
def extract_number(dir_name):
    match = number_pattern.search(dir_name)
    if match:
        return float(match.group())  # Convert to float if a match is found
    return float('inf')  # Return a very large number if no match is found


for root, dirs, files in os.walk(immpath_PODG):
    dirs.sort(key=lambda dir_name: extract_number(dir_name))
    if file_J in files:
        file_path = os.path.join(root, file_J)
        PODG_J.append(np.load(file_path)[-1])

for root, dirs, files in os.walk(immpath_sPODG):
    dirs.sort(key=lambda dir_name: extract_number(dir_name))
    if file_J in files:
        file_path = os.path.join(root, file_J)
        sPODG_J.append(np.load(file_path)[-1])

del PODG_J[:2]
del sPODG_J[:2]

modes_with_min_cost_PODG = PODG_modes[np.argmin(PODG_J)]
modes_with_min_cost_sPODG = sPODG_modes[np.argmin(sPODG_J)]


impath_FOM = "./data/FOM/problem=" + str(problem) + "/"
impath_sPODG = "./data/sPODG/problem=" + str(problem) + "/" + "modes=" + str(modes_with_min_cost_sPODG) + "/"  # Best sPODG
impath_PODG1 = "./data/PODG/problem=" + str(problem) + "/" + "modes=" + str(modes_with_min_cost_PODG) + "/"  # Best PODG
impath_PODG2 = "./data/PODG/problem=" + str(problem) + "/" + "modes=" + str(modes_with_min_cost_sPODG) + "/"  # PODG with best sPODG modes


f_opt_FOM = np.load(impath_FOM + 'f_opt.npy')
qs_adj_opt_FOM = np.load(impath_FOM + 'qs_adj_opt.npy')
qs_opt_FOM = np.load(impath_FOM + 'qs_opt.npy')

f_opt_PODG1 = np.load(impath_PODG1 + 'f_opt.npy')
qs_adj_opt_PODG1 = np.load(impath_PODG1 + 'qs_adj_opt.npy')
qs_opt_PODG1 = np.load(impath_PODG1 + 'qs_opt.npy')

f_opt_PODG2 = np.load(impath_PODG2 + 'f_opt.npy')
qs_adj_opt_PODG2 = np.load(impath_PODG2 + 'qs_adj_opt.npy')
qs_opt_PODG2 = np.load(impath_PODG2 + 'qs_opt.npy')

f_opt_sPODG = np.load(impath_sPODG + 'f_opt.npy')
qs_adj_opt_sPODG = np.load(impath_sPODG + 'qs_adj_opt.npy')
qs_opt_sPODG = np.load(impath_sPODG + 'qs_opt.npy')


f_min = np.min(f_opt_FOM)
f_max = np.max(f_opt_FOM)
adj_min = np.min(qs_adj_opt_FOM)
adj_max = np.max(qs_adj_opt_FOM)
state_min = np.min(qs_opt_FOM)
state_max = np.max(qs_opt_FOM)

##############################################################################################################
fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(3,4,1)
im1 = ax1.pcolormesh(f_opt_FOM.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax1.axis('off')
ax1.axis('auto')
ax1.set_title(r"$FOM$")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(3,4,2)
im2 = ax2.pcolormesh(f_opt_sPODG.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax2.axis('off')
ax2.axis('auto')
ax2.set_title(r"$sPOD-G, \: modes = $" + str(modes_with_min_cost_sPODG))
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax3 = fig.add_subplot(3,4,3)
im3 = ax3.pcolormesh(f_opt_PODG1.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax3.axis('off')
ax3.axis('auto')
ax3.set_title(r"$POD-G, \: modes = $" + str(modes_with_min_cost_PODG))
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im3, cax=cax, orientation='vertical')

ax4 = fig.add_subplot(3,4,4)
im4 = ax4.pcolormesh(f_opt_PODG2.T, cmap='YlOrRd', vmin=f_min, vmax=f_max)
ax4.axis('off')
ax4.axis('auto')
ax4.set_title(r"$POD-G, \: modes = $" + str(modes_with_min_cost_sPODG))
divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im4, cax=cax, orientation='vertical')

##############################################################################################################
ax5 = fig.add_subplot(3,4,5)
im5 = ax5.pcolormesh(qs_adj_opt_FOM.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax5.axis('off')
ax5.axis('auto')
divider = make_axes_locatable(ax5)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im5, cax=cax, orientation='vertical')

ax6 = fig.add_subplot(3,4,6)
im6 = ax6.pcolormesh(qs_adj_opt_sPODG.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax6.axis('off')
ax6.axis('auto')
divider = make_axes_locatable(ax6)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im6, cax=cax, orientation='vertical')

ax7 = fig.add_subplot(3,4,7)
im7 = ax7.pcolormesh(qs_adj_opt_PODG1.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax7.axis('off')
ax7.axis('auto')
divider = make_axes_locatable(ax7)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im7, cax=cax, orientation='vertical')

ax8 = fig.add_subplot(3,4,8)
im8 = ax8.pcolormesh(qs_adj_opt_PODG2.T, cmap='YlOrRd', vmin=adj_min, vmax=adj_max)
ax8.axis('off')
ax8.axis('auto')
divider = make_axes_locatable(ax8)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im8, cax=cax, orientation='vertical')

##############################################################################################################
ax9 = fig.add_subplot(3,4,9)
im9 = ax9.pcolormesh(qs_opt_FOM.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax9.axis('off')
ax9.axis('auto')
divider = make_axes_locatable(ax9)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im9, cax=cax, orientation='vertical')

ax10 = fig.add_subplot(3,4,10)
im10 = ax10.pcolormesh(qs_opt_sPODG.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax10.axis('off')
ax10.axis('auto')
divider = make_axes_locatable(ax10)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im10, cax=cax, orientation='vertical')

ax11 = fig.add_subplot(3,4,11)
im11 = ax11.pcolormesh(qs_opt_PODG1.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax11.axis('off')
ax11.axis('auto')
divider = make_axes_locatable(ax11)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im11, cax=cax, orientation='vertical')

ax12 = fig.add_subplot(3,4,12)
im12 = ax12.pcolormesh(qs_opt_PODG2.T, cmap='YlOrRd', vmin=state_min, vmax=state_max)
ax12.axis('off')
ax12.axis('auto')
divider = make_axes_locatable(ax12)
cax = divider.append_axes('right', size='10%', pad=0.08)
fig.colorbar(im12, cax=cax, orientation='vertical')

fig.supylabel(r"time $t$")
fig.supxlabel(r"space $x$")

fig.savefig(impath + 'Result_p' + str(problem), dpi=300, transparent=True)
