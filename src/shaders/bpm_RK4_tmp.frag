// RK4 BPM, not working, offset calculation is wrong
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
    vec2 offsetz = vec2(0.0, 1.0 / float(N) * dz / dx);
    vec2 offsetx = vec2(1.0 / float(N), 0.0);
    
    vec2 E_prev = rgba2vec(texture2D(texture, uv + offsetz));

    // Fetch the refractive index at (x, z)
    float n_i_encoded = texture2D(refractiveIndexTex, uv).a;
    float n_i = n_i_encoded * 5.0; // Decode refractive index

    // Fetch neighboring field values in x-direction at (z + dz)
    vec2 E_left = rgba2vec(texture2D(texture, uv - offsetx + offsetz));
    vec2 E_right = rgba2vec(texture2D(texture, uv + offsetx + offsetz));

    // Compute second derivative in x (Laplacian)
    vec2 d2E_dx2 = (E_left - 2.0 * E_prev + E_right) / (dx * dx);

    // Compute change in field along z using BPM equation
    // float delta_n = n_i * n_i - n0 * n0;
    float delta_n = n0 * n0 - n0 * n0;


    // Compute k1
    vec2 k1 = (d2E_dx2 + k0 * k0 * delta_n * E_prev) * (dz / (2.0 * k0 * n0));
    k1 = vec2(k1.y, -k1.x);  // Apply j multiplication

    // Compute E_mid1 for k2 step
    vec2 E_mid1 = E_prev + 0.5 * k1;

    // Fetch neighboring field values for E_mid1
    vec2 E_left_mid1 = rgba2vec(texture2D(texture, uv - offsetx + offsetz));
    vec2 E_right_mid1 = rgba2vec(texture2D(texture, uv + offsetx + offsetz));
    vec2 d2E_dx2_mid1 = (E_left_mid1 - 2.0 * E_mid1 + E_right_mid1) / (dx * dx);

    // Compute k2
    vec2 k2 = (d2E_dx2_mid1 + k0 * k0 * delta_n * E_mid1) * (dz / (2.0 * k0 * n0));
    k2 = vec2(k2.y, -k2.x);  // Apply j multiplication

    // Compute E_mid2 for k3 step
    vec2 E_mid2 = E_prev + 0.5 * k2;

    // Fetch neighboring field values for E_mid2
    vec2 E_left_mid2 = rgba2vec(texture2D(texture, uv - offsetx + offsetz));
    vec2 E_right_mid2 = rgba2vec(texture2D(texture, uv + offsetx + offsetz));
    vec2 d2E_dx2_mid2 = (E_left_mid2 - 2.0 * E_mid2 + E_right_mid2) / (dx * dx);

    // Compute k3
    vec2 k3 = (d2E_dx2_mid2 + k0 * k0 * delta_n * E_mid2) * (dz / (2.0 * k0 * n0));
    k3 = vec2(k3.y, -k3.x);  // Apply j multiplication

    // Compute E_end for k4 step
    vec2 E_end = E_prev + k3;

    // Fetch neighboring field values for E_end
    vec2 E_left_end = rgba2vec(texture2D(texture, uv - offsetx + offsetz));
    vec2 E_right_end = rgba2vec(texture2D(texture, uv + offsetx + offsetz));
    vec2 d2E_dx2_end = (E_left_end - 2.0 * E_end + E_right_end) / (dx * dx);

    // Compute k4
    vec2 k4 = (d2E_dx2_end + k0 * k0 * delta_n * E_end) * (dz / (2.0 * k0 * n0));
    k4 = vec2(k4.y, -k4.x);  // Apply j multiplication

    // Final RK4 update step
    vec2 E_new = E_prev + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    // Store the updated field at the current z position
    gl_FragColor = vec2rgba(E_new);
}
