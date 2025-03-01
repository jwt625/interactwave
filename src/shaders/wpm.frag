precision mediump float;
varying vec2 uv;

uniform int N;
uniform sampler2D texture;
uniform float k0;
uniform float dx;
uniform float dz;
uniform float lens_z;  // The depth at which the lens is located
uniform float lens_radius;
uniform float lens_refractive_index;

#define PI 3.14159265358979

#include ./utils/pack.frag


void main() {
    int n = int(uv.x * float(N));
    vec2 offset = vec2(0, 1.0 / float(N) * dz / dx);
    vec2 x = rgba2vec(texture2D(texture, uv + offset));

    float kx = (mod(uv.x + 0.5, 1.0) - 0.5) / dx * 2.0 * PI;
    float kz = sqrt(k0 * k0 - kx * kx);
    float angle = (kz * dz);

    vec2 W = vec2(cos(angle), sin(angle));
    x = vec2(x.x * W.x - x.y * W.y, x.x * W.y + x.y * W.x);

    // Apply lens phase shift when z reaches lens_z
    if (uv.y >= lens_z - dz && uv.y <= lens_z + dz) {
        float dx = uv.x - 0.5;  // Centered coordinate
        float dy = 0.0;  // Assuming a 1D lens effect (for simplicity)
        float r2 = dx * dx + dy * dy;

        // Lens thickness based on parabolic shape
        if (lens_radius * lens_radius - r2 < 0.0) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Debug: Output red if invalid
            return;
        }

        float thickness = lens_radius - sqrt(max(lens_radius * lens_radius - r2, 0.0));


        // Phase shift due to lens
        float phase_shift = k0 * (lens_refractive_index - 1.0) * thickness;
        vec2 lens_phase = vec2(cos(phase_shift), sin(phase_shift));

        // Apply phase shift
        float amplitude = length(x);
        x = vec2(amplitude * lens_phase.x, amplitude * lens_phase.y);

    }

    gl_FragColor = vec2rgba(x);
}
