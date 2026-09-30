# GETTING STARTED (no previous Python experience required)
# 1. If needed, install Python 3 from https://www.python.org/downloads/.
#    On Windows, select "Add Python to PATH" in the installer.
# 2. Open a terminal (Command Prompt or PowerShell on Windows) and type:
#        pip install numpy matplotlib
#    If pip is not found, try: python -m pip install numpy matplotlib
# 3. Save this entire file as transient_1d_explicit.py, not as a .txt file.
# 4. Open a terminal in that folder: on Windows, open the folder in File
#    Explorer, type cmd in its address bar, and press Enter. Alternatively,
#    use: cd "path/to/the/folder" in an existing terminal.
# 5. Run: python transient_1d_explicit.py
#    Some computers use py or python3 instead of python.
# 6. Edit the inputs below to explore material properties, mesh size, time
#    step, boundary temperatures, and total solution time. Save and rerun.
#    The contour PNG is saved beside this file; no figure window opens.

from pathlib import Path  # Part of standard Python; constructs file paths.
from time import perf_counter  # High-resolution elapsed-time clock.
import numpy as np       # NumPy supplies numerical arrays (abbreviated np).
import matplotlib
matplotlib.use("Agg")  # Write image files without opening figure windows.
import matplotlib.pyplot as plt  # Matplotlib supplies plotting commands.


# ----------------------------------------------------------------------
# 1. USER INPUTS: constant properties and a uniform one-dimensional rod
# ----------------------------------------------------------------------
# Model: rho*cp*dT/dt = k*d2T/dx2, or dT/dt = alpha*d2T/dx2.
# There is no internal heat generation or heat loss through the rod's sides.
# Properties are illustrative steel-like values, independent of temperature.
k = 45.0                 # Thermal conductivity [W/(m K)].
rho = 7800.0             # Density [kg/m^3].
cp = 500.0               # Specific heat capacity [J/(kg K)].
L = 0.10                 # Rod length [m].
# Fine grid: 200 spatial intervals (dx = 0.0005 m) and 60,000 time steps.
# Decrease dt when refining dx to keep FTCS stable. These settings give
# Fo = 0.230769 and store about 96 MB of temperature history per script.
number_of_nodes = 201    # Includes BOTH end nodes; integer >= 3.
dt = 0.005               # Time step [s]; stable for BOTH example scripts.
total_time = 300.0       # Total simulation time [s]; use a multiple of dt.
T_initial = 20.0         # Initial temperature of interior nodes [deg C].
T_left = 100.0           # Prescribed temperature at x = 0 [deg C].
T_right = 20.0           # Prescribed temperature at x = L [deg C].

# Temperature differences in degrees Celsius equal differences in kelvin.
if min(k, rho, cp, L, dt, total_time) <= 0:
    raise ValueError("Properties, length, time step, and total time must be positive.")
if not isinstance(number_of_nodes, int) or number_of_nodes < 3:
    raise ValueError("number_of_nodes must be an integer of at least 3.")

alpha = k / (rho * cp)   # Thermal diffusivity [m^2/s].
dx = L / (number_of_nodes - 1)  # Spacing between neighboring nodes [m].
Fo = alpha * dt / dx**2  # Fourier number (dimensionless); ** means power.
print(f"Thermal diffusivity alpha = {alpha:.6g} m^2/s")
print(f"Fourier number Fo = {Fo:.6f}")

# Standard 1D FTCS with these Dirichlet boundaries requires Fo <= 0.5
# for stability. An unstable time step amplifies errors instead of damping
# them. A finer mesh (smaller dx) therefore requires a smaller time step.
if Fo > 0.5:
    raise ValueError(
        f"UNSTABLE explicit time step: Fo = {Fo:.6f} > 0.5. "
        f"Choose dt <= {0.5 * dx**2 / alpha:.6g} s."
    )

# Use an integer number of equal steps so the final saved time is total_time.
number_of_steps = int(round(total_time / dt))
if number_of_steps < 1 or not np.isclose(
    number_of_steps * dt, total_time, rtol=1e-10, atol=1e-12
):
    raise ValueError("Choose total_time to be a positive integer multiple of dt.")


# ----------------------------------------------------------------------
# 2. SPATIAL GRID, TIME GRID, AND INITIAL/BOUNDARY CONDITIONS
# ----------------------------------------------------------------------
x = np.linspace(0.0, L, number_of_nodes)  # Equally spaced positions [m].
time = np.arange(number_of_steps + 1) * dt  # Saved times, including t = 0.

# T_history has one ROW per saved time and one COLUMN per spatial node.
# T_history[n, i] is the temperature at time[n] and position x[i].
# Python indexing begins at zero; index -1 denotes the last entry.
# np.zeros allocates storage; each later row will be filled by the solver.
T_history = np.zeros((number_of_steps + 1, number_of_nodes))
T_history[0, :] = T_initial  # ':' selects every spatial column in row zero.
T_history[:, 0] = T_left    # Prescribe the left end at every saved time.
T_history[:, -1] = T_right  # Prescribe the right end at every saved time.
# The left end is suddenly raised to T_left at t = 0; the interior still
# starts at T_initial. Thus the boundary values take precedence at t = 0.


# ----------------------------------------------------------------------
# 3. EXPLICIT FTCS: forward time, centered space
# ----------------------------------------------------------------------
# Approximate dT/dt by (T_i^(n+1) - T_i^n)/dt and d2T/dx2 by
# (T_(i-1)^n - 2*T_i^n + T_(i+1)^n)/dx^2. Rearranging gives:
# T_i^(n+1) = T_i^n + Fo*(T_(i-1)^n - 2*T_i^n + T_(i+1)^n).
# Superscripts n and n+1 label times; subscripts i label spatial nodes.
# Every value on the right is already known at OLD time n. No linear
# system is needed. Read only row n while writing the separate row n+1;
# updating the old row in place would accidentally change the method.
# Time only solver setup and time advancement, excluding imports, common
# grid/history allocation, plotting and PNG saving.
# Compare these teaching implementations, not the methods' best possible
# speed: explicit uses Python loops; implicit uses NumPy's dense solver.
# Run scripts individually for timing comparisons; run_both.bat launches
# them concurrently, so they can compete for CPU time. Repeat for context.
solver_start = perf_counter()
for n in range(number_of_steps):  # range stops BEFORE number_of_steps.
    for i in range(1, number_of_nodes - 1):  # Interior nodes only.
        T_history[n + 1, i] = T_history[n, i] + Fo * (
            T_history[n, i - 1]
            - 2.0 * T_history[n, i]
            + T_history[n, i + 1]
        )
    # End temperatures were already assigned for every row above.

solver_seconds = perf_counter() - solver_start
print(f"Explicit FTCS solver time: {solver_seconds:.6f} s")
print(f"Average time per step: {1e6 * solver_seconds / number_of_steps:.2f} microseconds")
print(f"Grid: {number_of_nodes} nodes, dx = {dx:g} m, dt = {dt:g} s")


# ----------------------------------------------------------------------
# 4. OPTIONAL BOUNDARY EQUATIONS (EXPLANATION ONLY; NOT EXECUTED)
# ----------------------------------------------------------------------
# The running example keeps BOTH specified-temperature (Dirichlet) ends.
# To change a boundary, remove its fixed assignment above and instead
# update that surface node at each time step. Let N = number_of_nodes - 1
# denote the RIGHT surface, with i = N-1 its neighbor inside the rod.
# The following equations use a centered derivative and a fictitious
# "ghost node" T_(N+1) outside the rod. Eliminating the ghost node gives a
# surface update that includes the heat capacity of a half-width end cell.
# Initialize a newly unfixed surface to an appropriate initial temperature.
#
# (1) SPECIFIED HEAT FLUX / NEUMANN, at the RIGHT end:
#     -k*dT/dx = q_flux [W/m^2]; positive q_flux removes heat to the right.
#     -k*(T_(N+1)^n - T_(N-1)^n)/(2*dx) = q_flux
#     => T_(N+1)^n = T_(N-1)^n - 2*dx*q_flux/k.
#     Substitute into FTCS to obtain:
#     T_N^(n+1) = T_N^n + 2*Fo*(T_(N-1)^n - T_N^n - dx*q_flux/k).
#     INSULATED: q_flux = 0, so the last term vanishes (zero heat loss).
#     This ghost-node flux update also requires Fo <= 0.5.
#
# (2) CONVECTION / ROBIN, at the RIGHT end:
#     -k*dT/dx = h*(T_surface - T_infinity).
#     h is the convection coefficient [W/(m^2 K)]; T_infinity is the
#     ambient fluid temperature [deg C]. Define Bi = h*dx/k (mesh Biot
#     number, not one based on the entire rod length). Then:
#     T_N^(n+1) = T_N^n + 2*Fo*(T_(N-1)^n
#                              - (1+Bi)*T_N^n + Bi*T_infinity).
#     For h >= 0, the conservative nonnegative-coefficient restriction is
#     Fo <= 1/[2*(1+Bi)], stricter than the interior Fo <= 0.5 condition.
#     The stability check above would need updating for this alternative.
#
# SIGN CONVENTION AT THE LEFT END:
# The outward direction there is -x. Physical convection therefore reads
# +k*dT/dx = h*(T_surface - T_infinity) at x = 0. With outward heat flux
# positive, the same surface updates apply with N replaced by 0 and N-1
# replaced by 1. If instead q_flux is defined in the fixed +x direction
# using -k*dT/dx = q_flux at the LEFT end, replace -dx*q_flux/k by
# +dx*q_flux/k in its surface update. Always specify the flux direction.


# ----------------------------------------------------------------------
# 5. PLOT THE COMPLETE TEMPERATURE HISTORY AND SAVE A PNG
# ----------------------------------------------------------------------
# meshgrid repeats x along rows and time along columns, making coordinate
# arrays with the same shape as T_history: (saved times, spatial nodes).
X, Time = np.meshgrid(x, time)
# FILLED CONTOUR PLOT: a direct 2D view of temperature versus x and time.
# contourf fills the bands between temperature levels; contour draws lines
# of constant temperature. This is a position-time map, not a second
# spatial dimension. coolwarm uses blue for cooler and red for hotter.
T_min = min(T_initial, T_left, T_right)
T_max = max(T_initial, T_left, T_right)
if T_min == T_max:  # Keep increasing contour levels for a uniform solution.
    T_min -= 0.5
    T_max += 0.5
levels = np.linspace(T_min, T_max, 21)  # 20 filled temperature bands.
contour_fig, contour_ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
filled = contour_ax.contourf(X, Time, T_history, levels=levels, cmap="coolwarm")
if np.ptp(T_history) > 0:
    lines = contour_ax.contour(
        X, Time, T_history, levels=levels[2:-1:2],
        colors="black", linewidths=0.4, alpha=0.5,
    )
    contour_ax.clabel(lines, inline=True, fontsize=8, fmt="%.0f")
contour_ax.set_xlabel("Position x [m]")
contour_ax.set_ylabel("Time [s]")
contour_ax.set_title("1D heat conduction: explicit FTCS temperature contours")
contour_ax.grid(True, color="gray", alpha=0.25)
contour_fig.colorbar(filled, ax=contour_ax, label="Temperature [deg C]")
contour_path = Path(__file__).resolve().parent / "transient_1d_explicit_contour.png"
contour_fig.savefig(contour_path, dpi=300)
print(f"Completed {number_of_steps} steps to t = {time[-1]:g} s.")
print(f"Saved {contour_path.name} in {contour_path.parent}")
plt.close(contour_fig)  # Release the figure after writing its PNG.
