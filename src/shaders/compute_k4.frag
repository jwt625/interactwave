precision mediump float;
varying vec2 uv;

uniform sampler2D E_prev_texture;
uniform sampler2D k3_texture;
uniform sampler2D refractiveIndexTex;
uniform float k0;
uniform float dx;
uniform float dz;
uniform float n0;
uniform int N;

#include ./utils/pack.frag

void main() {
    vec2 offsetz = vec2(0.0, 1.0 / float(N) * dz / dx);
    vec2 offsetx = vec2(1.0 / float(N), 0.0);

    vec2 E_prev = rgba2vec(texture2D(E_prev_texture, uv ));
    vec2 k3 = rgba2vec(texture2D(k3_texture, uv));

    vec2 E_end = E_prev + k3;

    vec2 k3_left = rgba2vec(texture2D(k3_texture, uv - offsetx));
    vec2 k3_right = rgba2vec(texture2D(k3_texture, uv + offsetx));
    vec2 E_left = rgba2vec(texture2D(E_prev_texture, uv - offsetx ));
    vec2 E_right = rgba2vec(texture2D(E_prev_texture, uv + offsetx ));

    vec2 E_left_end = E_left + k3_left;
    vec2 E_right_end = E_right + k3_right;

    float n_i = texture2D(refractiveIndexTex, uv).a * 5.0;
    float delta_n = n_i * n_i - n0 * n0;

    vec2 k4 = compute_dE_dz(E_end, E_left_end, E_right_end, delta_n, k0, n0, dx, dz);

    gl_FragColor = vec2rgba(k4);
}
