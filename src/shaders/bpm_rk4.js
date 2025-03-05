import { regl, rect } from '../render';

// Import RK4 shaders
const compute_k1 = regl({
    frag: (await import('./compute_k1.frag')).default,
    vert: (await import('./utils/default.vert')).default,
    attributes: { position: rect },
    uniforms: {
        E_prev_texture: regl.prop('E_prev_texture'),
        refractiveIndexTex: regl.prop('refractiveIndexTex'),
        dx: regl.prop('dx'),
        dz: regl.prop('dz'),
        k0: regl.prop('k0'),
        n0: regl.prop('n0'),
        N: regl.prop('N'),
    },
    count: 6,
});

const compute_k2 = regl({
    frag: (await import('./compute_k2.frag')).default,
    vert: (await import('./utils/default.vert')).default,
    attributes: { position: rect },
    uniforms: {
        E_prev_texture: regl.prop('E_prev_texture'),
        k1_texture: regl.prop('k1_texture'),
        refractiveIndexTex: regl.prop('refractiveIndexTex'),
        dx: regl.prop('dx'),
        dz: regl.prop('dz'),
        k0: regl.prop('k0'),
        n0: regl.prop('n0'),
        N: regl.prop('N'),
    },
    count: 6,
});

const compute_k3 = regl({
    frag: (await import('./compute_k3.frag')).default,
    vert: (await import('./utils/default.vert')).default,
    attributes: { position: rect },
    uniforms: {
        E_prev_texture: regl.prop('E_prev_texture'),
        k2_texture: regl.prop('k2_texture'),
        refractiveIndexTex: regl.prop('refractiveIndexTex'),
        dx: regl.prop('dx'),
        dz: regl.prop('dz'),
        k0: regl.prop('k0'),
        n0: regl.prop('n0'),
        N: regl.prop('N'),
    },
    count: 6,
});

const compute_k4 = regl({
    frag: (await import('./compute_k4.frag')).default,
    vert: (await import('./utils/default.vert')).default,
    attributes: { position: rect },
    uniforms: {
        E_prev_texture: regl.prop('E_prev_texture'),
        k3_texture: regl.prop('k3_texture'),
        refractiveIndexTex: regl.prop('refractiveIndexTex'),
        dx: regl.prop('dx'),
        dz: regl.prop('dz'),
        k0: regl.prop('k0'),
        n0: regl.prop('n0'),
        N: regl.prop('N'),
    },
    count: 6,
});

const rk4_update = regl({
    frag: (await import('./rk4_update.frag')).default,
    vert: (await import('./utils/default.vert')).default,
    attributes: { position: rect },
    uniforms: {
        E_prev_texture: regl.prop('E_prev_texture'),
        k1_texture: regl.prop('k1_texture'),
        k2_texture: regl.prop('k2_texture'),
        k3_texture: regl.prop('k3_texture'),
        k4_texture: regl.prop('k4_texture'),
        dx: regl.prop('dx'),
        dz: regl.prop('dz'),
        N: regl.prop('N'),
    },
    count: 6,
});

// BPM function using RK4
function BPM_RK4(fbo_prev, fbo_final, fbo_k1, fbo_k2, fbo_k3, fbo_k4,
    N, k0, dz, dx, n0, refractiveIndexTex) {
    // Compute k1
    fbo_k1.use(() => {
        compute_k1({
            E_prev_texture: fbo_prev.color[0],
            refractiveIndexTex,
            dx, dz, k0, n0, N,
        });
    });

    // Compute k2
    fbo_k2.use(() => {
        compute_k2({
            E_prev_texture: fbo_prev.color[0],
            k1_texture: fbo_k1.color[0],
            refractiveIndexTex,
            dx, dz, k0, n0, N,
        });
    });

    // Compute k3
    fbo_k3.use(() => {
        compute_k3({
            E_prev_texture: fbo_prev.color[0],
            k2_texture: fbo_k2.color[0],
            refractiveIndexTex,
            dx, dz, k0, n0, N,
        });
    });

    // Compute k4
    fbo_k4.use(() => {
        compute_k4({
            E_prev_texture: fbo_prev.color[0],
            k3_texture: fbo_k3.color[0],
            refractiveIndexTex,
            dx, dz, k0, n0, N,
        });
    });

    // Final RK4 update
    fbo_final.use(() => {
        rk4_update({
            E_prev_texture: fbo_prev.color[0],
            k1_texture: fbo_k1.color[0],
            k2_texture: fbo_k2.color[0],
            k3_texture: fbo_k3.color[0],
            k4_texture: fbo_k4.color[0],
            dx, dz, N,
        });
    });

    // Swap buffers
    let temp = fbo_prev;
    fbo_prev = fbo_final;
    fbo_final = temp;

    return [fbo_prev, fbo_final, fbo_k1, fbo_k2, fbo_k3, fbo_k4];
}

export { BPM_RK4 };
