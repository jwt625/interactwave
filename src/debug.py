#%%
# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
domain_size = 100.0  # um
wavelength = 0.532  # um
k0 = 2 * np.pi / wavelength  # Free space wavevector

# Grid size
Nx = 2**9
dx = domain_size / Nx  # Grid spacing
dz = dx  # Choose dz proportional to dx for stability
Nz = 1000  # Number of propagation steps

# Initialize field with a **localized** 1D Gaussian beam at z = 0
x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
z = np.linspace(0, dz * Nz, Nz)
X, Z = np.meshgrid(x, z, indexing="ij")

beam_width = 20.0  # Beam waist in um

# Typical lens parameters
lens_diameter = 30.0        # microns
lens_thickness = 3.0        # microns
R1 = 80.0                   # radius of curvature for first surface
R2 = 80.0                   # radius of curvature for second surface
n_lens = 1.50              # lens refractive index
n0 = 1.00                   # background refractive index
lens_center_z = 30.0        # center of lens along z in microns
x_lens = 0.0        # center of lens along x in microns

# Create an array for n_r^2 everywhere; default to n0^2 (outside the lens)
n_r2 = np.full((Nx, Nz), n0**2, dtype=np.float64)

# For each x in [-lens_diameter/2, lens_diameter/2], compute the lens surfaces
# Vectorized approach: we will define z_first(x) and z_second(x) for all x,
# then fill n_r2(x,z) = n_lens^2 where z_first <= z <= z_second.

# Precompute surfaces (1D in x)
z1 = lens_center_z - lens_thickness/2
z2 = lens_center_z + lens_thickness/2

# We'll create arrays z_first(x) and z_second(x).
z_first = z1 + (R1 - np.sqrt(R1**2 - (x-x_lens)**2))  # shape = (Nx,)
z_second = z2 - (R2 - np.sqrt(R2**2 - (x-x_lens)**2)) # shape = (Nx,)

# Now fill n_r2 where we are "inside" the lens
for ix in range(Nx):
    # skip if x is outside lens diameter
    if abs(x[ix]-x_lens) > lens_diameter/2:
        continue
    
    # The lens surfaces in z for this x
    zf = z_first[ix]
    zs = z_second[ix]
    
    # Find indices in the z-array that lie between these surfaces
    # (assuming z is a 1D array of shape (Nz,))
    in_lens = (z >= zf) & (z <= zs)
    
    # Fill n_lens^2 in that region
    n_r2[ix, in_lens] = n_lens**2


# Initialize E field with nonzero values only at z = 0 (source plane)
E = np.zeros((Nx, Nz), dtype=np.complex128)
E[:, 0] = np.exp(-x**2 / beam_width**2)  # 1D Gaussian beam at z = 0

# Finite difference coefficients
laplacian_coeff = 1 / dx**2
propagation_coeff = dz / (2j * k0 * n0)
laplacian_factor = 1j / (2 * k0 * n0)
index_factor     = 1j * (k0 / (2 * n0))


# Function to compute dE/dz (BPM update step)
def compute_dE_dz(E_slice, n_r2_slice):
    # Finite-difference Laplacian in x
    laplacian_E = (np.roll(E_slice, 1, axis=0) 
                   - 2*E_slice 
                   + np.roll(E_slice, -1, axis=0)) / dx**2
    
    # Laplacian term
    laplacian_term = laplacian_factor * laplacian_E
    
    # Index term
    # n_r2_slice - n0^2 for each x
    index_term = index_factor * (n_r2_slice - n0**2) * E_slice
    
    return laplacian_term + index_term


# Modify the source to include a small quadratic phase for focusing

# Define the focal length for the quadratic phase
focal_length = -200.0  # Adjust to control focusing behavior

# Apply a quadratic phase at the source plane (z = 0)
quadratic_phase = np.exp(-1j * (k0 / (2 * focal_length)) * x**2)
E[:, 0] *= quadratic_phase  # Multiply initial beam by phase factor

# Storage for stability check and snapshots
max_magnitude = []
snapshots = []
snapshot_intervals = np.linspace(1, Nz-1, 6, dtype=int)  # Capture 6 snapshots including initial and final

# BPM Propagation Loop with RK4
for zi in range(1, Nz):
    E_prev = E[:, zi-1]
    # Grab n_r2 at this step (i.e., the same z-slice)
    n_r2_slice = n_r2[:, zi-1]
    
    k1 = dz * compute_dE_dz(E_prev, n_r2_slice)
    k2 = dz * compute_dE_dz(E_prev + k1/2, n_r2_slice)
    k3 = dz * compute_dE_dz(E_prev + k2/2, n_r2_slice)
    k4 = dz * compute_dE_dz(E_prev + k3,     n_r2_slice)

    E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4)/6

    # Record max field magnitude to check divergence
    max_magnitude.append(np.max(np.abs(E)))

    # Store snapshots
    if zi in snapshot_intervals:
        print(zi)
        snapshots.append(np.abs(E.copy()))

# Plot BPM Propagation Snapshots
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    if i == 0:
        # Plot n_r2. Note the transpose (.T) to put x on the horizontal axis.
        im = ax.imshow(np.sqrt(n_r2.T), 
                        extent=[x[0], x[-1], z[0], z[-1]], 
                        origin='lower', 
                        aspect='auto', 
                        cmap='viridis')
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title("Lens Refractive Index Distribution (n_r)")
        continue
    im = ax.imshow(snapshots[i].T, 
            extent=[x[0], x[-1], z[0], z[-1]], aspect="auto",
            cmap="inferno", vmin=0, vmax=1,
            origin="lower")
    ax.set_title(f"Step {snapshot_intervals[i]}")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Propagation z (um)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("BPM Propagation with Quadratic Phase (Focusing)")

# Plot Stability Check
plt.figure(figsize=(6, 4))
plt.plot(max_magnitude)
plt.xlabel("Propagation Step")
plt.ylabel("Max |E|")
plt.title("Numerical Stability Check (Focusing)")

plt.show()

# %% plot n
plt.figure(figsize=(7, 5))

# Plot n_r2. Note the transpose (.T) to put x on the horizontal axis.
im = plt.imshow(np.sqrt(n_r2.T), 
                extent=[x[0], x[-1], z[0], z[-1]], 
                origin='lower', 
                aspect='auto', 
                cmap='viridis')

plt.colorbar(im, label="n_r")
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("Lens Refractive Index Distribution (n_r)")
plt.show()

# %% big plot of the final beam
plt.figure(figsize=(7, 7))

# Plot n_r2. Note the transpose (.T) to put x on the horizontal axis.
im = plt.imshow(snapshots[-1].T, 
                extent=[x[0], x[-1], z[0], z[-1]], 
                origin='lower', 
                aspect='auto', 
                # cmap='viridis',
                cmap="inferno")

# plt.colorbar(im)
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.show()


# %%
