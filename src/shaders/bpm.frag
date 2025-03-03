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
    // Fetch the current field at this location
    vec2 E = rgba2vec(texture2D(texture, uv));

    // Fetch the refractive index at this point
    float n_i = texture2D(refractiveIndexTex, uv).r;

    // Compute finite difference in x-direction
    vec2 E_left = rgba2vec(texture2D(texture, uv - vec2(1.0 / float(N), 0.0)));
    vec2 E_right = rgba2vec(texture2D(texture, uv + vec2(1.0 / float(N), 0.0)));

    // Apply second derivative (Laplacian) in x
    vec2 d2E_dx2 = (E_left - 2.0 * E + E_right) / (dx * dx);

    // Compute update term
    float delta_n = n_i * n_i - n0 * n0;
    vec2 dE_dz = (d2E_dx2 + k0 * k0 * delta_n * E) * (dz / (2.0 * k0 * n0));

    // Update field
    E += dE_dz;

    gl_FragColor = vec2rgba(E);
}
