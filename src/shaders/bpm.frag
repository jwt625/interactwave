precision mediump float;
varying vec2 uv;

uniform int N;
uniform sampler2D texture;
uniform float k0;
uniform float dx;
uniform float dz;
uniform float n0;
uniform sampler2D refractiveIndexTex;

#define PI 3.14159265358979

#include ./utils/pack.frag

void main() {
    // Fetch the field at (x, z + dz) (previous step in z, moving downward)
    vec2 offsetz = vec2(0, 1.0 / float(N) * dz / dx);
    // vec2 offsetx = vec2(1.0 / float(N) * dx / dz, 0.0);
    vec2 offsetx = vec2(1.0 / float(N), 0.0);
    
    vec2 E_prev = rgba2vec(texture2D(texture, uv + offsetz ));

    // Fetch the refractive index at (x, z)
    float n_i_encoded = texture2D(refractiveIndexTex, uv).a;
    float n_i = n_i_encoded * 5.0; // Decode refractive index
    // debug with n = 1
    // float n_i = 1.0;

    // Fetch neighboring field values in x-direction at (z + dz)
    vec2 E_left = rgba2vec(texture2D(texture, uv - offsetx + offsetz ));
    vec2 E_right = rgba2vec(texture2D(texture, uv + offsetx + offsetz ));

    // Compute second derivative in x (Laplacian)
    vec2 d2E_dx2 = (E_left - 2.0 * E_prev + E_right) / (dx * dx);

    // Compute change in field along z using Equation (5)
    float delta_n = n_i * n_i - n0 * n0;
    vec2 dE_dz = (d2E_dx2 + k0 * k0 * delta_n * E_prev) * (dz / (2.0 * k0 * n0));

    // Apply j multiplication: swap real and imaginary parts
    vec2 j_dE_dz = vec2(-dE_dz.y, dE_dz.x);

    // Forward propagation step: E(z) = E(z + dz) + j * dE/dz
    vec2 E_new = E_prev + j_dE_dz;

    // Store the updated field at the current z position
    gl_FragColor = vec2rgba(E_new);
}
