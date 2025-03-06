#%%
# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Refractive index generation functions
# ================================

def generate_lens_n_r2(x, z, lens_diameter, lens_thickness, R1, R2, n_lens, n0, lens_center_z, x_lens):
    """
    Generate the squared refractive index (n_r^2) distribution for a spherical lens.
    
    The lens is defined by two spherical surfaces.
    """
    Nx = len(x)
    Nz = len(z)
    n_r2 = np.full((Nx, Nz), n0**2, dtype=np.float64)
    
    # Precompute the z-positions of the lens surfaces (1D, for each x)
    z1 = lens_center_z - lens_thickness / 2.0
    z2 = lens_center_z + lens_thickness / 2.0
    
    # To avoid taking sqrt of negative numbers, use np.maximum(...,0)
    z_first = z1 + (R1 - np.sqrt(np.maximum(R1**2 - (x - x_lens)**2, 0)))
    z_second = z2 - (R2 - np.sqrt(np.maximum(R2**2 - (x - x_lens)**2, 0)))
    
    # Fill in the lens region where |x - x_lens| <= lens_diameter/2
    for ix in range(Nx):
        if abs(x[ix] - x_lens) > lens_diameter / 2:
            continue
        # Determine z indices where we are inside the lens
        in_lens = (z >= z_first[ix]) & (z <= z_second[ix])
        n_r2[ix, in_lens] = n_lens**2
    return n_r2

def generate_waveguide_n_r2(x, z, l, L, w, n_WG, n0):
    """
    Generate the squared refractive index (n_r^2) distribution for a weakly guided S-bend waveguide.
    
    The waveguide center is defined by a smooth S-bend trajectory:
    
        x_c(z) = (l/2) * [1 - cos(pi*z/L)]
    
    so that x_c(0)=0 and x_c(L)=l.
    The waveguide extends laterally between x = x_c(z) - w/2 and x = x_c(z) + w/2.
    Outside the waveguide, n_r = n0.
    """
    Nx = len(x)
    Nz = len(z)
    n_r2 = np.full((Nx, Nz), n0**2, dtype=np.float64)
    
    # Compute the waveguide center x_c for each z using the new sine-based S-bend:
    # x_c(z) = (l/L)*z - (l/(2*pi)) * sin((2*pi/L)*z), with endpoints clipped to [0, l]
    x_c = (l / L) * z - (l / (2 * np.pi)) * np.sin((2 * np.pi / L) * z)
    x_c = np.clip(x_c, 0, l)

    # For each propagation step (each z), mark the waveguide region in x
    for iz in range(Nz):
        lower_edge = x_c[iz] - w / 2.0
        upper_edge = x_c[iz] + w / 2.0
        in_wg = (x >= lower_edge) & (x <= upper_edge)
        n_r2[in_wg, iz] = n_WG**2
    return n_r2

def fundamental_mode_source(x, w, n_WG, n0, wavelength):
    """
    Returns the normalized fundamental TE mode profile for a symmetric slab waveguide.
    
    The mode is given by:
      E(x) =  cos(kx*x)                                  for |x| <= w/2
              cos(kx*(w/2)) * exp[-kappa*(|x|-w/2)]       for |x| > w/2
             
    where:
      k0 = 2*pi / wavelength
      beta is the propagation constant for the fundamental mode,
      kx = sqrt(n_WG^2 * k0^2 - beta^2),
      kappa = sqrt(beta^2 - n0^2 * k0^2),
      and beta satisfies:  kx * tan(kx*w/2) = kappa.

    We scan all possible roots and pick the largest beta, which corresponds
    to the fundamental (lowest-order) TE mode.
    """
    k0 = 2 * np.pi / wavelength

    def f(beta):
        """Function whose root(s) define valid TE slab modes."""
        # If beta is out of range or leads to imaginary kx/kappa, skip
        if beta < n0*k0 or beta > n_WG*k0:
            return None
        inside = n_WG**2 * k0**2 - beta**2
        outside = beta**2 - n0**2 * k0**2
        if inside <= 0 or outside <= 0:
            return None
        
        kx = np.sqrt(inside)
        kappa = np.sqrt(outside)
        return kx * np.tan(kx * w / 2) - kappa

    # 1) Find all roots via scanning
    #    We'll sample a range of betas from n0*k0 up to n_WG*k0
    N = 2000
    beta_vals = np.linspace(n0*k0, n_WG*k0, N)
    f_vals = []
    for b in beta_vals:
        val = f(b)
        # If f(b) is invalid (None) or not real, store None
        if val is None or np.isnan(val):
            f_vals.append(None)
        else:
            f_vals.append(val)

    # 2) Identify intervals where sign changes occur
    #    We'll store (b_left, b_right) for each sign change
    sign_change_intervals = []
    for i in range(N-1):
        if (f_vals[i] is not None and f_vals[i+1] is not None):
            if f_vals[i] * f_vals[i+1] < 0:
                sign_change_intervals.append((beta_vals[i], beta_vals[i+1]))

    # 3) For each interval, do a local bisection to refine the root
    roots = []
    for (b_left, b_right) in sign_change_intervals:
        for _ in range(50):  # up to 50 bisection iterations
            b_mid = 0.5*(b_left + b_right)
            val_mid = f(b_mid)
            if val_mid is None:
                # If f(b_mid) is invalid, shrink the interval
                b_right = b_mid
                continue
            if abs(val_mid) < 1e-9:
                # Converged
                roots.append(b_mid)
                break
            val_left = f(b_left)
            # If val_left is None, treat it as same sign as val_mid
            if val_left is None or val_left*val_mid > 0:
                b_left = b_mid
            else:
                b_right = b_mid
        else:
            # If we never 'break' from the bisection, store the midpoint
            roots.append(b_mid)

    # 4) Pick the largest valid root (this is the fundamental TE0 mode)
    if not roots:
        raise ValueError("No valid slab mode found in [n0*k0, n_WG*k0].")

    beta = max(roots)  # fundamental has the largest beta
    # Now compute kx and kappa from that beta
    kx = np.sqrt(n_WG**2 * k0**2 - beta**2)
    kappa = np.sqrt(beta**2 - n0**2 * k0**2)

    # 5) Construct the field profile
    E = np.zeros_like(x, dtype=np.complex128)
    for i, xi in enumerate(x):
        if abs(xi) <= w / 2:
            E[i] = np.cos(kx * xi)
        else:
            E[i] = np.cos(kx * (w / 2)) * np.exp(-kappa * (abs(xi) - w / 2))

    # 6) Normalize the mode
    norm = np.sqrt(np.trapz(np.abs(E)**2, x))
    E /= norm
    return E


# ================================
# Main simulation setup
# ================================

# Physical parameters
domain_size = 100.0  # um
wavelength = 0.532   # um
k0 = 2 * np.pi / wavelength  # Free space wavevector

# Grid size
Nx = 2**9
dx = domain_size / Nx     # Grid spacing
dz = 3*dx                 # Choose dz proportional to dx for stability
Nz = 300               # Number of propagation steps

# Create spatial grids
x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
z = np.linspace(0, dz * Nz, Nz)
X, Z = np.meshgrid(x, z, indexing="ij")


# ================================
# Refractive index distribution setup
# ================================

# Uncomment one of the following blocks depending on the desired structure

# ----- Option 1: Spherical Lens -----
# Typical lens parameters
# lens_diameter = 30.0        # microns
# lens_thickness = 3.0        # microns
# R1 = 80.0                   # radius of curvature for first surface
# R2 = 80.0                   # radius of curvature for second surface
# n_lens = 1.50               # lens refractive index
# n0 = 1.00                   # background refractive index
# lens_center_z = 30.0        # center of lens along z in microns
# x_lens = 0.0                # center of lens along x in microns
#
# n_r2 = generate_lens_n_r2(x, z, lens_diameter, lens_thickness, R1, R2, n_lens, n0, lens_center_z, x_lens)

# ----- Option 2: S-bend Waveguide -----
# For testing the waveguide, we use n0=1 and n_WG=1.1.
n0 = 1.0
n_WG = 1.1
# S-bend parameters:
l = 10.0   # total lateral offset in microns at z=L
L = 150.0   # length of the S-bend in microns
w = 1    # waveguide width in microns

n_r2 = generate_waveguide_n_r2(x, z, l, L, w, n_WG, n0)

# ================================
# Initialize the input field E (Gaussian beam at z=0)
# ================================
E = np.zeros((Nx, Nz), dtype=np.complex128)

# ===========
# mode source
# ===========
# Gaussian beam with curvature
# beam_width = 4.0  # Beam waist in um
# E[:, 0] = np.exp(-x**2 / beam_width**2)
# # Optionally apply a quadratic phase for additional focusing
# focal_length = -200.0  # Adjust to control focusing behavior
# quadratic_phase = np.exp(-1j * (k0 / (2 * focal_length)) * x**2)
# E[:, 0] *= quadratic_phase

# waveguide mode
E[:, 0] = fundamental_mode_source(x, w, n_WG, n0, wavelength)


# ================================
# BPM propagation parameters and function
# ================================
laplacian_factor = 1j / (2 * k0 * n0)
index_factor     = 1j * (k0 / (2 * n0))

def compute_dE_dz(E_slice, n_r2_slice):
    """
    Compute the derivative dE/dz using the BPM equation:
    
        ∂E/∂z = (i/(2k0 n0)) (∂^2 E/∂x^2) + i (k0/(2n0)) [n_r^2(x,z) - n0^2] E.
    """
    # Finite-difference Laplacian in x
    laplacian_E = (np.roll(E_slice, 1, axis=0) 
                   - 2 * E_slice 
                   + np.roll(E_slice, -1, axis=0)) / dx**2
    
    laplacian_term = laplacian_factor * laplacian_E
    index_term = index_factor * (n_r2_slice - n0**2) * E_slice
    
    return laplacian_term + index_term

# ================================
# BPM Propagation Loop with RK4
# ================================
max_magnitude = []
snapshots = []
snapshot_intervals = np.linspace(1, Nz-1, 6, dtype=int)  # Capture 6 snapshots

for zi in range(1, Nz):
    E_prev = E[:, zi-1]
    n_r2_slice = n_r2[:, zi-1]
    
    k1 = dz * compute_dE_dz(E_prev, n_r2_slice)
    k2 = dz * compute_dE_dz(E_prev + k1/2, n_r2_slice)
    k3 = dz * compute_dE_dz(E_prev + k2/2, n_r2_slice)
    k4 = dz * compute_dE_dz(E_prev + k3, n_r2_slice)
    
    E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    max_magnitude.append(np.max(np.abs(E)))
    
    if zi in snapshot_intervals:
        print("Snapshot at step:", zi)
        snapshots.append(np.abs(E.copy()))

# ================================
# Plotting
# ================================

# Plot BPM Propagation Snapshots
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i == 0:
        # Plot the refractive index distribution (n_r = sqrt(n_r2))
        im = ax.imshow(np.sqrt(n_r2.T),
                       extent=[x[0], x[-1], z[0], z[-1]],
                       origin='lower',
                       aspect='auto',
                       cmap='viridis')
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title("Refractive Index Distribution (n_r)")
    else:
        im = ax.imshow(snapshots[i].T,
                       extent=[x[0], x[-1], z[0], z[-1]],
                       origin="lower",
                       aspect="auto",
                       cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"Step {snapshot_intervals[i]}")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("Propagation z (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("BPM Propagation with Quadratic Phase (Focusing)")
plt.show()

# Plot Stability Check
plt.figure(figsize=(6, 4))
plt.plot(max_magnitude)
plt.xlabel("Propagation Step")
plt.ylabel("Max |E|")
plt.title("Numerical Stability Check")
plt.show()

# Plot the final beam intensity
plt.figure(figsize=(7, 7))
im = plt.imshow(snapshots[-1].T,
                extent=[x[0], x[-1], z[0], z[-1]],
                origin='lower',
                aspect='auto',
                cmap="inferno")
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("Final Beam Intensity")
plt.show()

# %% check mode source
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.io as pio


# Create a matplotlib figure with a legend and custom font sizes
plt.rcParams.update({'font.size': 14})  # Global font size for text elements
fig, ax = plt.subplots()
ax.plot(x, E[:, 0])
# ax.plot(x, n_r2_slice)


# Convert the matplotlib figure to a Plotly figure
plotly_fig = tls.mpl_to_plotly(fig)

# Optionally update the legend font size in the Plotly figure layout
plotly_fig.update_layout(
    legend=dict(font=dict(size=12))
)

# Display the Plotly figure
pio.show(plotly_fig)
# %%
