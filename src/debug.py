#%%
# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
domain_size = 30.0  # um
wavelength = 0.532  # um
k0 = 2 * np.pi / wavelength  # Free space wavevector
n0 = 1.5  # Refractive index
dx = domain_size / 2**8  # Grid spacing
dz = dx  # Choose dz proportional to dx for stability

# Grid size
Nx = 2**7
Nz = 200  # Number of propagation steps

# Initialize field with a **localized** 1D Gaussian beam at z = 0
x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
z = np.linspace(0, dz * Nz, Nz)
X, Z = np.meshgrid(x, z, indexing="ij")

beam_width = 2.0  # Beam waist in um

# Initialize E field with nonzero values only at z = 0 (source plane)
E = np.zeros((Nx, Nz), dtype=np.complex128)
E[:, 0] = np.exp(-x**2 / beam_width**2)  # 1D Gaussian beam at z = 0

# Finite difference coefficients
laplacian_coeff = 1 / dx**2
propagation_coeff = dz / (2j * k0 * n0)

# Function to compute dE/dz (BPM update step)
def compute_dE_dz(E_slice):
    laplacian_E = np.roll(E_slice, 1, axis=0) - 2 * E_slice + np.roll(E_slice, -1, axis=0)
    laplacian_E *= laplacian_coeff
    return laplacian_E * propagation_coeff

# Storage for stability check and snapshots
max_magnitude = []
snapshots = []
snapshot_intervals = np.linspace(1, Nz-1, 6, dtype=int)  # Capture 6 snapshots including initial and final

# BPM Propagation Loop with RK4
for zi in range(1, Nz):  # Start from zi=1 since zi=0 is the source
    E_prev = E[:, zi-1]

    k1 = compute_dE_dz(E_prev)
    k2 = compute_dE_dz(E_prev + k1 / 2)
    k3 = compute_dE_dz(E_prev + k2 / 2)
    k4 = compute_dE_dz(E_prev + k3)

    # RK4 update step
    E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Record max field magnitude to check divergence
    max_magnitude.append(np.max(np.abs(E)))

    # Store snapshots
    if zi in snapshot_intervals:
        print(zi)
        snapshots.append(np.abs(E.copy()))

# Plot BPM Propagation Snapshots
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    im = ax.imshow(snapshots[i], 
            extent=[0, domain_size, 0, dz * Nz], aspect="auto",
            cmap="inferno", vmin=0, vmax=1)
    ax.set_title(f"Step {snapshot_intervals[i]}")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Propagation z (um)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("BPM Propagation Snapshots (RK4)")

# Plot Stability Check
plt.figure(figsize=(6, 4))
plt.plot(max_magnitude)
plt.xlabel("Propagation Step")
plt.ylabel("Max |E|")
plt.title("Numerical Stability Check (RK4)")

plt.show()

# %%
