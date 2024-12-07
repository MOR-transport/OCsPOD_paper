import numpy as np
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
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


immpath_FOM = "./data/FOM/problem=" + str(problem) + "/"
immpath_PODG = "./data/PODG/problem=" + str(problem) + "/"
immpath_sPODG = "./data/sPODG/problem=" + str(problem) + "/"
FOM_J = np.load(immpath_FOM + "J_opt_list.npy")
FOM_t = np.load(immpath_FOM + "running_time.npy")

if problem == 1:
    PODG_modes = np.asarray([4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40])
    sPODG_modes = np.asarray([4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40])
else:
    PODG_modes = np.asarray(
        [4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    sPODG_modes = np.asarray([4, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40])


file_J = "J_opt_FOM_list.npy"
file_t = "running_time.npy"

PODG_J = []
PODG_t = []
sPODG_J = []
sPODG_t = []

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
    # print(f"Current directory: {root}")
    # print(f"Subdirectories: {dirs}")
    # print(f"Files: {files}")
    if file_J in files:
        file_path = os.path.join(root, file_J)
        PODG_J.append(np.load(file_path))
    if file_t in files:
        file_path = os.path.join(root, file_t)
        PODG_t.append(np.load(file_path))


for root, dirs, files in os.walk(immpath_sPODG):
    dirs.sort(key=lambda dir_name: extract_number(dir_name))
    # print(f"Current directory: {root}")
    # print(f"Subdirectories: {dirs}")
    # print(f"Files: {files}")
    if file_J in files:
        file_path = os.path.join(root, file_J)
        sPODG_J.append(np.load(file_path))
    if file_t in files:
        file_path = os.path.join(root, file_t)
        sPODG_t.append(np.load(file_path))

del PODG_t[:2]
del sPODG_t[:2]
del PODG_J[:2]
del sPODG_J[:2]


# Interpolate for common grid spec (PODG)
t_min_1 = min([min(t) for t in PODG_t])
t_max_1 = max([max(t) for t in PODG_t])
t_min_2 = min([min(t) for t in sPODG_t])
t_max_2 = max([max(t) for t in sPODG_t])
t_min_3 = min(FOM_t)
t_max_3 = max(FOM_t)
t_min = min(t_min_1, t_min_2, t_min_3)
t_max = max(t_max_1, t_max_2, t_max_3)
common_t_grid = np.linspace(t_min, t_max, 1000000)

# Interpolate each J array to the common time grid
PODG_J_interpolated = []
for t, J in zip(PODG_t, PODG_J):
    len_J = len(J)
    len_t = len(t)
    length = min(len_J, len_t)
    interp_func = interp1d(t[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)
    PODG_J_interpolated.append(interp_func(common_t_grid))

# Interpolate each J array to the common time grid
sPODG_J_interpolated = []
for t, J in zip(sPODG_t, sPODG_J):
    len_J = len(J)
    len_t = len(t)
    length = min(len_J, len_t)
    interp_func = interp1d(t[:length], J[:length], kind='linear', fill_value=(J[0], J[-1]), bounds_error=False)
    sPODG_J_interpolated.append(interp_func(common_t_grid))

# Interpolate each J array to the common time grid
interp_func = interp1d(FOM_t, FOM_J, kind='linear', fill_value=(FOM_J[0], FOM_J[-1]), bounds_error=False)
FOM_interpolated = interp_func(common_t_grid)


PODG_J_min = np.nanmin(np.stack(PODG_J_interpolated), axis=0)
sPODG_J_min = np.nanmin(np.stack(sPODG_J_interpolated), axis=0)

fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(111)
ax1.loglog(common_t_grid, FOM_interpolated, color='sienna', linestyle='-', label="FOM")
ax1.loglog(common_t_grid, PODG_J_min, color='green', linestyle='--', label="POD-G")
ax1.loglog(common_t_grid, sPODG_J_min, color='red', linestyle='dashdot', label="sPOD-G")
ax1.set_xlabel(r"run time $(\mathrm{s})$")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_title(r"$\mathcal{J}$ vs run time")
ax1.legend()
ax1.grid()

fig.savefig(impath + 'J_vs_runtime_P' + str(problem), dpi=300, transparent=True)