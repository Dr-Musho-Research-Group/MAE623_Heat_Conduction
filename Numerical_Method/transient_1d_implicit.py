# GETTING STARTED (no previous Python experience required)
# 1. If needed, install Python 3 from https://www.python.org/downloads/.
#    On Windows, select "Add Python to PATH" in the installer.
# 2. Open a terminal (Command Prompt or PowerShell on Windows) and type:
#        pip install numpy matplotlib
#    If pip is not found, try: python -m pip install numpy matplotlib
# 3. Save this entire file as transient_1d_implicit.py, not as a .txt file.
# 4. Open a terminal in that folder: on Windows, open the folder in File
#    Explorer, type cmd in its address bar, and press Enter. Alternatively,
#    use: cd "path/to/the/folder" in an existing terminal.
# 5. Run: python transient_1d_implicit.py
#    Some computers use py or python3 instead of python.
# 6. Edit the inputs below to explore material properties, mesh size, time
#    step, boundary temperatures, and total solution time. Save and rerun.
#    The contour PNG is saved beside this file; no figure window opens.

from pathlib import Path  # Part of standard Python; constructs file paths.
from time import perf_counter  # High-resolution elapsed-time clock.
import numpy as np       # NumPy supplies arrays and a linear-system solver.
import matplotlib
matplotlib.use("Agg")  # Write image files without opening figure windows.
import matplotlib.pyplot as plt  # Matplotlib supplies plotting commands.


# ----------------------------------------------------------------------
# 1. USER INPUTS: same physical problem and grid as the explicit example
# ----------------------------------------------------------------------
# Model: rho*cp*dT/dt = k*d2T/dx2, or dT/dt = alpha*d2T/dx2.
# Homogeneous rod, constant properties, no internal heat generation, and
# no heat loss through the sides. The values are illustrative steel-like
# properties. Temperature differences in degrees Celsius equal kelvin.
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

if min(k, rho, cp, L, dt, total_time) <= 0:
    raise ValueError("Properties, length, time step, and total time must be positive.")
if not isinstance(number_of_nodes, int) or number_of_nodes < 3:
    raise ValueError("number_of_nodes must be an integer of at least 3.")

alpha = k / (rho * cp)   # Thermal diffusivity [m^2/s].
dx = L / (number_of_nodes - 1)  # Uniform spatial spacing [m].
Fo = alpha * dt / dx**2  # Fourier number (dimensionless); ** means power.
print(f"Thermal diffusivity alpha = {alpha:.6g} m^2/s")
print(f"Fourier number Fo = {Fo:.6f}")
# Backward Euler is unconditionally stable for this linear diffusion
# problem: it does not require Fo <= 0.5. Stability does NOT ensure good
# accuracy, however; large time steps obscure rapid temperature changes.
# Both schemes are first-order accurate in time and second-order in space
# at interior nodes. Their finite-dt results need not be exactly identical.

number_of_steps = int(round(total_time / dt))
if number_of_steps < 1 or not np.isclose(
    number_of_steps * dt, total_time, rtol=1e-10, atol=1e-12
):
    raise ValueError("Choose total_time to be a positive integer multiple of dt.")


# ----------------------------------------------------------------------
# 2. SPATIAL GRID, TIME GRID, AND INITIAL/BOUNDARY CONDITIONS
# ----------------------------------------------------------------------
x = np.linspace(0.0, L, number_of_nodes)  # Positions including both ends.
time = np.arange(number_of_steps + 1) * dt  # Times including t = 0 [s].

# Each ROW of T_history is a saved time; each COLUMN is a spatial node.
# T_history[n, i] means temperature at time[n] and x[i]. Python starts
# indexing at zero; -1 means the last entry, and ':' selects all entries.
T_history = np.zeros((number_of_steps + 1, number_of_nodes))
T_history[0, :] = T_initial
T_history[:, 0] = T_left
T_history[:, -1] = T_right
# At t = 0 the left end is suddenly raised to T_left. Boundary values
# override the uniform initial value at the endpoints; the interior
# remains at T_initial initially.


# ----------------------------------------------------------------------
# 3. BUILD THE BACKWARD-EULER TRIDIAGONAL COEFFICIENT MATRIX
# ----------------------------------------------------------------------
# Unlike FTCS, evaluate the spatial derivative at the NEW time n+1:
# (T_i^(n+1) - T_i^n)/dt = alpha*(T_(i-1)^(n+1)
#                               - 2*T_i^(n+1) + T_(i+1)^(n+1))/dx^2.
# Rearranging gives one equation for each interior node:
# -Fo*T_(i-1)^(n+1) + (1+2*Fo)*T_i^(n+1) - Fo*T_(i+1)^(n+1) = T_i^n.
# The neighboring new temperatures are unknown too, so all new values
# must be solved simultaneously. Write these equations as A*T_new = b.
# A has nonzero entries only on its main diagonal and the two adjacent
# diagonals: it is TRIDIAGONAL. We store it as a regular 2D NumPy array
# for clarity and use np.linalg.solve, rather than an optimized solver.
# With fixed properties, grid, and dt, A is the same at every time step.
# Time only solver setup and time advancement, excluding imports, common
# grid/history allocation, plotting and PNG saving.
# Compare these teaching implementations, not the methods' best possible
# speed: explicit uses Python loops; implicit uses NumPy's dense solver.
# Run scripts individually for timing comparisons; run_both.bat launches
# them concurrently, so they can compete for CPU time. Repeat for context.
solver_start = perf_counter()
A = np.zeros((number_of_nodes, number_of_nodes))
for i in range(1, number_of_nodes - 1):  # Only interior equation rows.
    A[i, i - 1] = -Fo          # Coefficient of the left neighbor.
    A[i, i] = 1.0 + 2.0 * Fo   # Coefficient of the node itself.
    A[i, i + 1] = -Fo          # Coefficient of the right neighbor.

# Boundary rows express T_0^(n+1) = T_left and T_N^(n+1) = T_right.
# Their only nonzero coefficient is 1 on the main diagonal.
A[0, 0] = 1.0
A[-1, -1] = 1.0


# ----------------------------------------------------------------------
# 4. ADVANCE IN TIME BY SOLVING THE LINEAR SYSTEM AT EACH STEP
# ----------------------------------------------------------------------
for n in range(number_of_steps):  # n runs from 0 through steps - 1.
    # Interior right-hand sides are OLD temperatures. copy() ensures that
    # changing b does not change the stored old-temperature row.
    b = T_history[n, :].copy()
    b[0] = T_left    # Right-hand side of the left boundary equation.
    b[-1] = T_right  # Right-hand side of the right boundary equation.
    # solve returns the new vector of temperatures, including both ends.
    # Do not form an inverse of A; solve is the direct NumPy operation.
    T_history[n + 1, :] = np.linalg.solve(A, b)

solver_seconds = perf_counter() - solver_start
print(f"Implicit Backward Euler solver time: {solver_seconds:.6f} s")
print(f"Average time per step: {1e6 * solver_seconds / number_of_steps:.2f} microseconds")
print(f"Grid: {number_of_nodes} nodes, dx = {dx:g} m, dt = {dt:g} s")


# ----------------------------------------------------------------------
# 5. OPTIONAL BOUNDARY EQUATIONS (EXPLANATION ONLY; NOT EXECUTED)
# ----------------------------------------------------------------------
# The running example keeps BOTH specified-temperature (Dirichlet) ends.
# The alternatives below replace the RIGHT boundary row of A and b,
# and its fixed T_history assignment. Let N = number_of_nodes - 1.
# Initialize a newly unfixed surface to an appropriate initial temperature.
# These equations eliminate a fictitious "ghost node" outside the rod
# using a centered boundary derivative. They include heat storage in a
# half-width end cell. All boundary fluxes are evaluated at NEW time n+1.
# Clear the existing boundary row before inserting its new coefficients.
#
# (1) SPECIFIED HEAT FLUX / NEUMANN, at the RIGHT end:
#     -k*dT/dx = q_flux [W/m^2]; positive q_flux removes heat to the right.
#     T_(N+1)^(n+1) = T_(N-1)^(n+1) - 2*dx*q_flux/k.
#     Substituting into Backward Euler gives:
#     -2*Fo*T_(N-1)^(n+1) + (1+2*Fo)*T_N^(n+1)
#         = T_N^n - 2*Fo*dx*q_flux/k.
#     Thus A[N,N-1] = -2*Fo, A[N,N] = 1+2*Fo, and on EVERY time step
#     b[N] = T_history[n,N] - 2*Fo*dx*q_flux/k (instead of T_right).
#     INSULATED: q_flux = 0, so b[N] = T_history[n,N].
#
# (2) CONVECTION / ROBIN, at the RIGHT end:
#     -k*dT/dx = h*(T_surface - T_infinity).
#     h [W/(m^2 K)] is the convection coefficient; T_infinity [deg C]
#     is the ambient fluid temperature. Define Bi = h*dx/k (mesh Biot
#     number, not one based on the entire rod length). Then:
#     -2*Fo*T_(N-1)^(n+1) + [1+2*Fo*(1+Bi)]*T_N^(n+1)
#         = T_N^n + 2*Fo*Bi*T_infinity.
#     Thus A[N,N-1] = -2*Fo, A[N,N] = 1+2*Fo*(1+Bi), and EVERY step
#     b[N] = T_history[n,N] + 2*Fo*Bi*T_infinity (instead of T_right).
#     For constant h the matrix can still be built once before the loop.
#
# SIGN CONVENTION AT THE LEFT END:
# The outward direction is -x there, so physical convection reads
# +k*dT/dx = h*(T_surface - T_infinity). With outward heat flux positive,
# the same equations apply with N replaced by 0 and N-1 replaced by 1.
# If q_flux instead denotes fixed +x-directed flux, -k*dT/dx = q_flux
# at the LEFT end, its right-hand side is T_0^n + 2*Fo*dx*q_flux/k.
# Always specify the flux direction before choosing the sign.


# ----------------------------------------------------------------------
# 6. PLOT THE COMPLETE TEMPERATURE HISTORY AND SAVE A PNG
# ----------------------------------------------------------------------
# meshgrid creates coordinate arrays matching the shape of T_history:
# rows correspond to saved times, columns to spatial positions.
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
contour_ax.set_title("1D heat conduction: implicit Backward Euler temperature contours")
contour_ax.grid(True, color="gray", alpha=0.25)
contour_fig.colorbar(filled, ax=contour_ax, label="Temperature [deg C]")
contour_path = Path(__file__).resolve().parent / "transient_1d_implicit_contour.png"
contour_fig.savefig(contour_path, dpi=300)
print(f"Completed {number_of_steps} steps to t = {time[-1]:g} s.")
print(f"Saved {contour_path.name} in {contour_path.parent}")
plt.close(contour_fig)  # Release the figure after writing its PNG.
