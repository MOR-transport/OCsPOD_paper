import numpy as np
import os
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


##########################################################################
# Plotting the modes vs cost functional plot
immpath_FOM = "./data/FOM/problem=" + str(problem) + "/"
immpath_PODG = "./data/PODG/problem=" + str(problem) + "/"
immpath_sPODG = "./data/sPODG/problem=" + str(problem) + "/"
FOM_J = np.load(immpath_FOM + "J_opt_list.npy")[-1]

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

POD_J_tol = PODG_J[:2]
sPOD_J_tol = sPODG_J[:2]
POD_J_modes = PODG_J[2:]
sPODG_J_modes = sPODG_J[2:]

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(111)
ax1.axhline(y=FOM_J, color='sienna', linestyle='-', label="FOM")
ax1.scatter(PODG_modes, POD_J_modes, marker="s", label="POD-G", color="green")
ax1.scatter(sPODG_modes, sPODG_J_modes, marker="*", label="sPOD-G", color="red")
ax1.set_xlabel(r"truncated modes")
ax1.set_ylabel(r"$\mathcal{J}$")
ax1.set_yscale('log')
ax1.grid()
ax1.legend()
fig.savefig(impath + 'ModesVsCost_P' + str(problem), dpi=300, transparent=True)



# Tolerance values for table
print(f"Cost functional reached for POD-G with tolerance = 0.01:  {POD_J_tol[1]}")
print(f"Cost functional reached for POD-G with tolerance = 0.001:  {POD_J_tol[0]}")
print(f"Cost functional reached for sPOD-G with tolerance = 0.01:  {sPOD_J_tol[1]}")
print(f"Cost functional reached for sPOD-G with tolerance = 0.001:  {sPOD_J_tol[0]}")


print("\n\n")

P_Nm_avg_PODG_tol_0_01 = np.load("./data/PODG/problem=" + str(problem) + "/tol=0.01/trunc_modes_list.npy")
P_Nm_avg_PODG_tol_0_001 = np.load("./data/PODG/problem=" + str(problem) + "/tol=0.001/trunc_modes_list.npy")
P_Nm_avg_sPODG_tol_0_01 = np.load("./data/sPODG/problem=" + str(problem) + "/tol=0.01/trunc_modes_list.npy")
P_Nm_avg_sPODG_tol_0_001 = np.load("./data/sPODG/problem=" + str(problem) + "/tol=0.001/trunc_modes_list.npy")

print(f"Average number of modes per iteration for POD-G with tolerance = 0.01:  {int(sum(P_Nm_avg_PODG_tol_0_01) / len(P_Nm_avg_PODG_tol_0_01))}")
print(f"Average number of modes per iteration for POD-G with tolerance = 0.001:  {int(sum(P_Nm_avg_PODG_tol_0_001) / len(P_Nm_avg_PODG_tol_0_001))}")
print(f"Average number of modes per iteration for sPOD-G with tolerance = 0.01:  {int(sum(P_Nm_avg_sPODG_tol_0_01) / len(P_Nm_avg_sPODG_tol_0_01))}")
print(f"Average number of modes per iteration for sPOD-G with tolerance = 0.001:  {int(sum(P_Nm_avg_sPODG_tol_0_001) / len(P_Nm_avg_sPODG_tol_0_001))}")
