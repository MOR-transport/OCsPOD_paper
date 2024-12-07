import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import sys
sys.path.append('./sPOD/lib/')
from sPOD_algo import give_interpolation_error
from transforms import Transform

impath = "./paper_plots/"
os.makedirs(impath, exist_ok=True)


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


def objective(c, X, qs_0, qs_frame):
    interpolated_f = np.interp(X + c, X, qs_frame, period=X[-1])
    error = interpolated_f - qs_0

    squared_error = np.linalg.norm(error)
    return squared_error


def calc_shift(qs, qs_0, X, t):
    # Initial guess for c
    Nt = len(t)
    initial_guess_c = 0
    optimal_c = np.zeros((1, Nt))

    for i in range(Nt):
        qs_frame = qs[:, i]
        # Minimize the objective function with respect to c
        result = optimize.minimize(objective, np.asarray([initial_guess_c]), args=(X, qs_0, qs_frame,))

        # Extract the optimal value of c
        optimal_c[:, i] = -result.x[0]
        initial_guess_c = result.x[0]

    return optimal_c


def get_T(delta_s, X, t, interp_order):
    Nx = len(X)
    Nt = len(t)

    data_shape = [Nx, 1, 1, Nt]
    dx = X[1] - X[0]
    L = [X[-1]]

    # Create the transformations
    trafo_1 = Transform(data_shape, L, shifts=delta_s[0],
                        dx=[dx],
                        use_scipy_transform=False,
                        interp_order=interp_order)

    return trafo_1.shifts_pos, trafo_1


L = 10
Nx = 2000
Nt = 2000
mu = 0.9
var = 0.01
offset = 20
cfl = 1.0

x = np.arange(1, Nx + 1) * L / Nx
dx = x[1] - x[0]
dt = dx * cfl
t = dt * np.arange(Nt)
print(dx, dt, x[0], x[-1], t[0], t[-1])
XX, TT = np.meshgrid(x, t)


# Initial condition
Q0 = np.exp(-((x - L / offset) ** 2) / var)


# Traveling wave
QT = np.exp(-((XX.T - mu * TT.T - L / offset) ** 2) / var)


# Stationary wave
delta = calc_shift(QT, Q0, x, t)
_, T = get_T(delta, x, t, interp_order=5)
print(2 * give_interpolation_error(QT, T))
QS = T.reverse(QT)


# Singular value decay of waves:
rank = 100
U_T, SIG_T, VH_T = np.linalg.svd(QT, full_matrices=False)
U_S, SIG_S, VH_S = np.linalg.svd(QS, full_matrices=False)
SIG_T_k_SIG_T_0 = SIG_T[:rank] / SIG_T[0]
SIG_S_k_SIG_S_0 = SIG_S[:rank] / SIG_S[0]


# PLOT the combined figure
# Create figure and axis for the line plots
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the lines
ax.semilogy(np.arange(1, rank + 1), SIG_T_k_SIG_T_0, label='Traveling wave', color='green', marker="*")
ax.semilogy(np.arange(1, rank + 1), SIG_S_k_SIG_S_0, label='Stationary wave', color='orange', marker="*")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\sigma_k/\sigma_0$")
ax.grid()
ax.legend()

# Set the images on the plot in non-overlapping positions
# Adjust positions as necessary to avoid overlap

# Add first image in the upper left corner of the plot
newax1 = fig.add_axes([0.4, 0.55, 0.1, 0.2], anchor='NW', zorder=5)
newax1.pcolormesh(QT.T, cmap='YlGn')
newax1.set_xlabel(r"$x$")
newax1.set_ylabel(r"$t$")
newax1.set_yticks([], [])
newax1.set_xticks([], [])
newax1.set_title("Traveling wave")


# Add second image in the upper right corner of the plot
newax2 = fig.add_axes([0.4, 0.25, 0.1, 0.2], anchor='SE', zorder=5)
newax2.pcolormesh(QS.T, cmap='YlOrRd')
newax2.set_xlabel(r"$x$")
newax2.set_ylabel(r"$t$")
newax2.set_yticks([], [])
newax2.set_xticks([], [])
newax2.set_title("Stationary wave")

fig.savefig(impath + "Trav_vs_Stat", dpi=300, transparent=True)