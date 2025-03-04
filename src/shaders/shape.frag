precision lowp float;
uniform vec4 color;
varying vec2 uv;

uniform float width;
uniform float power;
uniform sampler2D texture;
uniform vec2 mouse;
uniform int N;
uniform float k0;
uniform float lens_z;
uniform float lens_radius;
uniform float lens_refractive_index;

#include ./utils/pack.frag

void main(){
    float uvx = uv.x - 0.5 / float(N);

    // old:
    // float A = smoothstep(0.0, 4.0 / float(N), width * 0.5 - abs(uvx - 0.5)) / sqrt(width) * 0.025;
    // Apply Gaussian function for smoother beam edges
    float A = exp(-pow((uvx - 0.5) / (width * 0.3), 2.0)) / sqrt(width) * 0.025;

    // Apply curvature (phase shift for beam focus)
    float cs2 = pow((uvx - 0.5), 2.0) * power;
    float angle = k0 * cs2 / (1.0 + sqrt(1.0 - cs2 * power));

    vec2 u = vec2(cos(angle), sin(angle)) * A;

    vec2 x = rgba2vec(texture2D(texture, uv));

    // Apply mouse blocking (cursor blocks the wave)
    float r = distance(uv, mouse);
    x *= float(smoothstep(0.0, 0.06, r));

    // Inject the source at the bottom
    x = mix(x, u, step(1.0 - 10.0 / float(N), uv.y));

    // Apply lens phase shift at lens_z
    if (abs(uv.y - lens_z) < 1.0 / float(N)) {
        float dx = uv.x - 0.5; // Centered x-coordinate
        float r2 = dx * dx;
        
        // Compute focal length
        float focal_length = lens_radius * lens_radius / (lens_refractive_index - 1.0);

        // Compute quadratic phase shift for a thin lens
        float lens_phase_shift = -0.5 * k0 / focal_length * r2;
        vec2 lens_phase = vec2(cos(lens_phase_shift), sin(lens_phase_shift));

        // Apply phase shift to the existing wave field
        x = vec2(x.x * lens_phase.x - x.y * lens_phase.y, 
                 x.x * lens_phase.y + x.y * lens_phase.x);
    }

    gl_FragColor = vec2rgba(x);
}
