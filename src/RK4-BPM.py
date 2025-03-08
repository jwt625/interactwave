#%%
# Re-import necessary libraries
import numpy as np
import warnings
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


def generate_MMI_n_r2(x, z, z_MMI_start, L_MMI, w_MMI, w_wg, d, n_WG, n_MMI, n0):
    """
    Generate the squared refractive index (n_r^2) distribution for an MMI-based splitter.
    
    The structure consists of:
      - Two input waveguides in the region z < z_MMI_start, with centers at x = -d/2 and x = d/2
        and width w_wg (refractive index n_WG).
      - A central MMI region for z_MMI_start <= z <= z_MMI_start + L_MMI,
        which is a rectangular region of width w_MMI (refractive index n_MMI).
      - Two output waveguides in the region z > (z_MMI_start + L_MMI), with the same positions
        and dimensions as the input waveguides (n_WG).
    
    Parameters
    ----------
    x : 1D numpy array
        The transverse coordinate (in µm).
    z : 1D numpy array
        The propagation coordinate (in µm).
    z_MMI_start : float
        z position where the MMI region begins.
    L_MMI : float
        Length of the MMI region along z (in µm).
    w_MMI : float
        Width of the MMI region (in µm).
    w_wg : float
        Width of the input and output waveguides (in µm).
    d : float
        Center-to-center separation between the two waveguides (in µm). (Waveguide centers are at –d/2 and d/2.)
    n_WG : float
        Refractive index of the input/output waveguides.
    n_MMI : float
        Refractive index of the MMI region.
    n0 : float
        Background refractive index.
    
    Returns
    -------
    n_r2 : 2D numpy array of shape (len(x), len(z))
        The squared refractive index distribution.
    """
    import numpy as np
    Nx = len(x)
    Nz = len(z)
    
    # Create a 2D grid with x as rows and z as columns.
    X, Z = np.meshgrid(x, z, indexing="ij")
    
    # Initialize with background index squared.
    n_r2 = np.full((Nx, Nz), n0**2, dtype=np.float64)
    
    # Define z position where the MMI region ends.
    z_MMI_end = z_MMI_start + L_MMI
    
    # Create mask for input waveguides: z < z_MMI_start.
    # The waveguide centers are at x = -d/2 and x = d/2.
    mask_input = (Z < z_MMI_start) & (((np.abs(X + d/2) <= w_wg/2) | (np.abs(X - d/2) <= w_wg/2)))
    
    # Create mask for the MMI region: z between z_MMI_start and z_MMI_end and |x| <= w_MMI/2.
    mask_MMI = (Z >= z_MMI_start) & (Z <= z_MMI_end) & (np.abs(X) <= w_MMI/2)
    
    # Create mask for output waveguides: z > z_MMI_end.
    mask_output = (Z > z_MMI_end) & (((np.abs(X + d/2) <= w_wg/2) | (np.abs(X - d/2) <= w_wg/2)))
    
    # Assign refractive index squared to the designated regions.
    n_r2[mask_input] = n_WG**2
    n_r2[mask_output] = n_WG**2
    n_r2[mask_MMI]   = n_MMI**2
    
    return n_r2


def slab_mode_source(x, w, n_WG, n0, wavelength, ind_m=0, x0=0):
    """
    Returns the normalized TE mode profile of a symmetric slab waveguide for the
    specified mode index (ind_m). Modes are computed from the dispersion relations:
    
      Even modes: f_even(beta) = kx * tan(kx*w/2) - kappa = 0,
      Odd  modes: f_odd(beta)  = -kx * cot(kx*w/2) - kappa = 0,
      
    where:
      k0 = 2*pi / wavelength,
      kx = sqrt(n_WG^2 * k0^2 - beta^2),
      kappa = sqrt(beta^2 - n0^2 * k0^2).

    This solver partitions the search into continuous branches by checking that the
    parameter theta = kx*w/2 lies in the appropriate interval:
      - For even modes: theta in [j*pi, j*pi + pi/2] for some even j (j=0 for fundamental).
      - For odd modes:  theta in [j*pi + pi/2, (j+1)*pi] for some even j.
      
    If the requested mode index is out of range, a warning is issued.
    """
    k0 = 2 * np.pi / wavelength

    def f_even(beta):
        if beta < n0*k0 or beta > n_WG*k0:
            return None
        inside = n_WG**2 * k0**2 - beta**2
        outside = beta**2 - n0**2 * k0**2
        if inside <= 0 or outside <= 0:
            return None
        kx = np.sqrt(inside)
        kappa = np.sqrt(outside)
        return kx * np.tan(kx * w / 2) - kappa

    def f_odd(beta):
        if beta < n0*k0 or beta > n_WG*k0:
            return None
        inside = n_WG**2 * k0**2 - beta**2
        outside = beta**2 - n0**2 * k0**2
        if inside <= 0 or outside <= 0:
            return None
        kx = np.sqrt(inside)
        kappa = np.sqrt(outside)
        sin_term = np.sin(kx * w / 2)
        if abs(sin_term) < 1e-12:
            return None
        return - kx * (np.cos(kx * w / 2) / sin_term) - kappa
    
    def valid_even(beta):
        inside = n_WG**2 * k0**2 - beta**2
        if inside <= 0:
            return False
        kx = np.sqrt(inside)
        theta = kx * w / 2
        m = int(np.floor(2 * theta / np.pi))
        # For even modes, require that m is even.
        # Additionally, for the fundamental (m == 0), reject if theta is too close to pi/2.
        if m % 2 == 0:
            if m == 0 and theta > (np.pi/2 - 0.1):
                return False
            return True
        return False

    def valid_odd(beta):
        inside = n_WG**2 * k0**2 - beta**2
        if inside <= 0:
            return False
        kx = np.sqrt(inside)
        theta = kx * w / 2
        m = int(np.floor(2 * theta / np.pi))
        # Accept candidate if its branch index m is odd.
        return (m % 2 == 1)


    N = 2000
    beta_scan = np.linspace(n0*k0, n_WG*k0, N)
    even_intervals = []
    odd_intervals = []
    f_even_vals = [f_even(b) for b in beta_scan]
    f_odd_vals = [f_odd(b) for b in beta_scan]
    for i in range(N-1):
        if (f_even_vals[i] is not None) and (f_even_vals[i+1] is not None):
            if f_even_vals[i] * f_even_vals[i+1] < 0:
                even_intervals.append((beta_scan[i], beta_scan[i+1]))
        if (f_odd_vals[i] is not None) and (f_odd_vals[i+1] is not None):
            if f_odd_vals[i] * f_odd_vals[i+1] < 0:
                odd_intervals.append((beta_scan[i], beta_scan[i+1]))
                
    def refine_root(f, b_left, b_right):
        for _ in range(50):
            b_mid = 0.5*(b_left+b_right)
            val_mid = f(b_mid)
            if val_mid is None:
                b_right = b_mid
                continue
            if abs(val_mid) < 1e-9:
                return b_mid
            val_left = f(b_left)
            if val_left is None or val_left*val_mid > 0:
                b_left = b_mid
            else:
                b_right = b_mid
        return b_mid

    even_roots = []
    for (b_left, b_right) in even_intervals:
        root = refine_root(f_even, b_left, b_right)
        if valid_even(root):
            even_roots.append(root)
    odd_roots = []
    for (b_left, b_right) in odd_intervals:
        root = refine_root(f_odd, b_left, b_right)
        if valid_odd(root):
            odd_roots.append(root)

    modes = [("even", r) for r in even_roots] + [("odd", r) for r in odd_roots]
    modes_sorted = sorted(modes, key=lambda tup: tup[1], reverse=True)
    if len(modes_sorted) == 0:
        raise ValueError("No guided slab modes found in [n0*k0, n_WG*k0].")
    if ind_m >= len(modes_sorted):
        warnings.warn(
            f"Requested mode index {ind_m} >= found modes ({len(modes_sorted)}). Using highest mode index {len(modes_sorted)-1}.",
            UserWarning
        )
        ind_m = len(modes_sorted) - 1

    parity, beta_chosen = modes_sorted[ind_m]
    inside = n_WG**2 * k0**2 - beta_chosen**2
    outside = beta_chosen**2 - n0**2 * k0**2
    kx = np.sqrt(inside)
    kappa = np.sqrt(outside)

    E = np.zeros_like(x, dtype=np.complex128)
    if parity == "even":
        for i, xi in enumerate(x):
            # Shift the coordinate by x0.
            xp = xi - x0
            if abs(xp) <= w/2:
                E[i] = np.cos(kx * xp)
            else:
                E[i] = np.cos(kx * (w/2)) * np.exp(-kappa * (abs(xp)-w/2))
    else:
        for i, xi in enumerate(x):
            xp = xi - x0
            if abs(xp) <= w/2:
                E[i] = np.sin(kx * xp)
            else:
                E[i] = np.sign(xp) * np.sin(kx * (w/2)) * np.exp(-kappa * (abs(xp)-w/2))
    norm = np.sqrt(np.trapz(np.abs(E)**2, x))
    E /= norm
    return E

# ================================
# Main simulation setup
# ================================

# Physical parameters
domain_size = 50  # um
wavelength = 0.532   # um
k0 = 2 * np.pi / wavelength  # Free space wavevector

# Grid size
Nx = 2**8
dx = domain_size / Nx     # Grid spacing
dz = 2*dx                 # Choose dz proportional to dx for stability
Nz = 600               # Number of propagation steps

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
# # For testing the waveguide, we use n0=1 and n_WG=1.1.
# n0 = 1.0
# n_WG = 1.1
# # S-bend parameters:
# l = 10.0   # total lateral offset in microns at z=L
# L = 200.0   # length of the S-bend in microns
# w = 1    # waveguide width in microns
# n_r2 = generate_waveguide_n_r2(x, z, l, L, w, n_WG, n0)

# ----- Option 3: MMI splitter -----
# MMI structure parameters:
z_MMI_start = 50.0    # MMI region begins at z = 50 um
L_MMI = 130.0          # MMI region length = 40 um
w_MMI = 8.0          # MMI region width = 40 um
w_wg = 2.0            # input/output waveguide width = 4 um
d_wg = 4.0              # center-to-center separation of waveguides = 12 um

# Refractive indices:
n0 = 1.0
n_WG = 1.1    # waveguide (input/output) index
n_MMI = 1.1   # MMI region index (can be different, if desired)

# Generate the refractive index distribution.
n_r2 = generate_MMI_n_r2(x, z, z_MMI_start, L_MMI, w_MMI,
        w_wg, d_wg, n_WG, n_MMI, n0)



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
E[:, 0] = slab_mode_source(x, w=w_wg, n_WG=n_WG, n0=n0,
    wavelength=wavelength, ind_m=1, x0=d_wg/2)



# PML
# Define the PML thickness (in grid points) and compute its physical width.
pml_thickness = int(5 * wavelength / dx)  # e.g., 5 wavelengths thick
pml_width = pml_thickness * dx

# Determine the x position at which the PML starts (on both sides).
x_edge = domain_size / 2 - pml_width

# Create the damping profile sigma(x) for the transverse coordinate.
# sigma is zero in the interior and rises smoothly (quadratically) in the PML region.
sigma_max = 0.5  # Adjustable absorption strength
sigma_x = np.where(np.abs(x) > x_edge,
                   sigma_max * ((np.abs(x) - x_edge) / pml_width) ** 2,
                   0)

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
    damping_term = - sigma_x * E_slice  # PML damping term
    
    return laplacian_term + index_term + damping_term

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

# %% Plot the final beam intensity
plt.figure(figsize=(7, 7))
im = plt.imshow(snapshots[-1].T,
                extent=[x[0], x[-1], z[0], z[-1]],
                origin='lower',
                aspect='auto',
                cmap="inferno", vmin=0, vmax=1)
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("Final Beam Intensity")
plt.show()


#%% debug PML
import numpy as np
import matplotlib.pyplot as plt

# Assuming the following variables are defined:
# wavelength, dx, domain_size, and x (the x-grid)

# Define PML parameters
pml_thickness = int(5 * wavelength / dx)  # e.g., 5 wavelengths thick
pml_width = pml_thickness * dx
x_edge = domain_size / 2 - pml_width       # x position where PML begins
sigma_max = 0.5                          # maximum damping strength

# Compute sigma_x: zero in interior, quadratic rise in PML regions.
sigma_x = np.where(np.abs(x) > x_edge,
                   sigma_max * ((np.abs(x) - x_edge) / pml_width) ** 2,
                   0)

# Plot sigma_x versus x
plt.figure(figsize=(8, 4))
plt.plot(x, sigma_x, label=r'$\sigma(x)$', lw=2)
plt.xlabel("x (µm)")
plt.ylabel(r'$\sigma(x)$')
plt.title("PML Damping Profile")
plt.legend()
plt.grid(True)
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







# %% check waveguide modes
import numpy as np
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.io as pio
import warnings

# Define transverse coordinate and waveguide parameters
x = np.linspace(-10, 10, 1000)  # Adjust the range and resolution as needed
w = 5.0        # Waveguide width in microns
n_WG = 1.1     # Core refractive index
n0 = 1.0       # Cladding refractive index
wavelength = 0.532  # in microns

# Calculate up to 10 modes (indices 0 through 9)
num_modes = 5
mode_fields = []
mode_indices = []

for m in range(num_modes):
    try:
        E_mode = slab_mode_source(x, w, n_WG, n0, wavelength, ind_m=m)
        mode_fields.append(E_mode)
        mode_indices.append(m)
        # print(tmp)
    except Exception as err:
        warnings.warn(f"Mode {m} not found: {err}. Stopping mode search.")
        break

# Plot the real part of each mode on the same figure
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(8, 6))
for i, E_mode in enumerate(mode_fields):
    ax.plot(x, np.real(E_mode) + i, label=f"Mode {mode_indices[i]}")
ax.set_xlabel("x (µm)")
ax.set_ylabel("Field amplitude (Real part)")
ax.set_title("Slab Waveguide TE Modes")
ax.legend()

# Convert the matplotlib figure to a Plotly figure for interactive viewing
plotly_fig = tls.mpl_to_plotly(fig)
plotly_fig.update_layout(legend=dict(font=dict(size=12)))
pio.show(plotly_fig)


# %% check MMI
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
domain_size = 100.0   # um, transverse extent
z_total = 200.0       # um, propagation distance
Nx = 512
Nz = 512
x = np.linspace(-domain_size/2, domain_size/2, Nx)
z = np.linspace(0, z_total, Nz)

# MMI structure parameters:
z_MMI_start = 50.0    # MMI region begins at z = 50 um
L_MMI = 40.0          # MMI region length = 40 um
w_MMI = 40.0          # MMI region width = 40 um
w_wg = 4.0            # input/output waveguide width = 4 um
d = 12.0              # center-to-center separation of waveguides = 12 um

# Refractive indices:
n0 = 1.0      # background
n_WG = 1.1    # waveguide (input/output) index
n_MMI = 1.1   # MMI region index (can be different, if desired)

# Generate the refractive index distribution.
n_r2 = generate_MMI_n_r2(x, z, z_MMI_start, L_MMI, w_MMI, w_wg, d, n_WG, n_MMI, n0)

#%%
# Plot the refractive index distribution.
plt.figure(figsize=(8, 6))
plt.imshow(np.sqrt(n_r2).T, extent=[x[0], x[-1], z[0], z[-1]], origin='lower', aspect='auto', cmap='viridis')
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("MMI Splitter Refractive Index Distribution (n_r)")
plt.colorbar(label="n_r")
plt.show()


# %%
