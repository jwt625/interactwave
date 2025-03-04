
#%%

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
domain_size = 30.0  # um
wavelength = 0.532  # um
k0 = 2 * np.pi / wavelength  # Free space wavevector
n0 = 1.5  # Refractive index
dx = domain_size / 2**7  # Grid spacing
dz = dx  # Choose dz proportional to dx for stability

# Grid size
Nx = 2**7
Nz = 160  # Number of propagation steps

# Initialize field with a Gaussian beam
x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
z = np.linspace(0, dz * Nz, Nz)
X, Z = np.meshgrid(x, z, indexing="ij")

beam_width = 5.0  # Beam waist in um
E = np.exp(-X**2 / beam_width**2).astype(np.complex128)  # Envelope field

# Finite difference coefficients
laplacian_coeff = 1 / dx**2
propagation_coeff = dz / (2j * k0 * n0)

# Storage for stability check
max_magnitude = []

# BPM Propagation Loop
for zi in range(Nz):
    # Second derivative in x using central difference
    laplacian_E = np.roll(E, 1, axis=0) - 2 * E + np.roll(E, -1, axis=0)
    laplacian_E *= laplacian_coeff

    # BPM update step
    dE_dz = (laplacian_E) * propagation_coeff
    E += dE_dz

    # Record max field magnitude to check divergence
    max_magnitude.append(np.max(np.abs(E)))

# Plot Results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(E), extent=[0, domain_size, 0, dz * Nz], aspect="auto", cmap="inferno")
plt.colorbar(label="Field Magnitude")
plt.title("Final Beam Profile")

plt.subplot(1, 2, 2)
plt.plot(max_magnitude)
plt.xlabel("Propagation Step")
plt.ylabel("Max |E|")
plt.title("Numerical Stability Check")

plt.show()

# %%
