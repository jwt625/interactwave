precision mediump float;
varying vec2 uv;

uniform int N;
uniform sampler2D texture;
uniform float k0;
uniform float dx;
uniform float dz;

#define PI 3.14159265358979
// Slab parameters (all normalized to the domain)
uniform float slab_angle;
uniform float slab_thickness;
uniform float slab_length;
uniform float slab_x_center;
uniform float slab_z_center;
uniform float slab_refractive_index;

#include ./utils/pack.frag

void main() {
    int n = int(uv.x * float(N));
    vec2 offset = vec2(0, 1.0 / float(N) * dz / dx);
    vec2 x = rgba2vec(texture2D(texture, uv + offset));

    float kx = (mod(uv.x + 0.5, 1.0) - 0.5) / dx * 2.0 * PI;
    float kz = sqrt(k0 * k0 - kx * kx); // Ensure non-negative sqrt

    // // Real-space z-coordinate (since uv.y corresponds to z in real space)
    // float z_world = uv.y; // Normalized domain (0 to 1)

    // // Check if the wave is currently inside the slab (along z only)
    // float z_min = slab_z_center - slab_thickness / 2.0;
    // float z_max = slab_z_center + slab_thickness / 2.0;
    // bool inside_slab = (z_world >= z_min) && (z_world <= z_max);

    // // Modify propagation speed inside the slab
    // if (inside_slab) {
    //     float k_slab = k0 * slab_refractive_index;
    //     kz = sqrt(max(k_slab * k_slab - kx * kx, 0.0)); // Ensure non-negative value
    // }

    float angle = (kz * dz);
    vec2 W = vec2(cos(angle), sin(angle));
    x = vec2(x.x * W.x - x.y * W.y, x.x * W.y + x.y * W.x);

    gl_FragColor = vec2rgba(x);
}
