precision mediump float;
varying vec2 uv;

uniform sampler2D E_prev_texture;
uniform sampler2D k1_texture;
uniform sampler2D k2_texture;
uniform sampler2D k3_texture;
uniform sampler2D k4_texture;
uniform float dx;
uniform float dz;
uniform int N;

#include ./utils/pack.frag

void main() {
    vec2 offsetz = vec2(0.0, 1.0 / float(N) * dz / dx);
    vec2 offsetx = vec2(1.0 / float(N), 0.0);

    vec2 E_prev = rgba2vec(texture2D(E_prev_texture, uv + offsetz));
    vec2 k1 = rgba2vec(texture2D(k1_texture, uv + offsetz));
    vec2 k2 = rgba2vec(texture2D(k2_texture, uv + offsetz));
    vec2 k3 = rgba2vec(texture2D(k3_texture, uv + offsetz));
    vec2 k4 = rgba2vec(texture2D(k4_texture, uv + offsetz));

    vec2 E_new = E_prev + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    gl_FragColor = vec2rgba(E_new);
}
