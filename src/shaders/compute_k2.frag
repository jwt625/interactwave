precision mediump float;
varying vec2 uv;

uniform sampler2D E_prev_texture;
uniform sampler2D k1_texture;
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
    vec2 k1 = rgba2vec(texture2D(k1_texture, uv));

    vec2 E_mid1 = E_prev + 0.5 * k1;

    vec2 k1_left = rgba2vec(texture2D(k1_texture, uv - offsetx));
    vec2 k1_right = rgba2vec(texture2D(k1_texture, uv + offsetx));
    vec2 E_left = rgba2vec(texture2D(E_prev_texture, uv - offsetx ));
    vec2 E_right = rgba2vec(texture2D(E_prev_texture, uv + offsetx ));

    vec2 E_left_mid1 = E_left + 0.5 * k1_left;
    vec2 E_right_mid1 = E_right + 0.5 * k1_right;

    float n_i = texture2D(refractiveIndexTex, uv).a * 5.0;
    float delta_n = n_i * n_i - n0 * n0;

    vec2 k2 = compute_dE_dz(E_mid1, E_left_mid1, E_right_mid1, delta_n, k0, n0, dx, dz);

    gl_FragColor = vec2rgba(k2);
}
