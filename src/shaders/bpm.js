import { regl, rect } from '../render';

// BPM shader
const bpm = regl({
    frag: (await import('./BPM.frag')).default,
    vert: (await import('./utils/default.vert')).default,
    attributes: {
        position: rect,
    },
    uniforms: {
        N: regl.prop('N'),
        texture: regl.prop('texture'),
        dx: regl.prop('dx'),
        dz: regl.prop('dz'),
        k0: regl.prop('k0'),
        n0: regl.prop('n0'),
        refractiveIndexTex: regl.prop('refractiveIndexTex'),
    },
    count: 6,
});

function BPM(fbo, fbo_temp, N, k0, dz, dx, n0, refractiveIndexTex) {
    fbo_temp.use(function () {
        regl.clear({
            color: [0, 0, 0, 0],
            depth: 1,
        });
        bpm({
            N: N,
            texture: fbo.color[0],
            dz: dz,
            dx: dx,
            k0: k0,
            n0: n0,
            refractiveIndexTex: refractiveIndexTex,
        });
    });

    // Swap buffers
    const temp = fbo_temp;
    fbo_temp = fbo;
    fbo = temp;
    
    return [fbo, fbo_temp];
}

export { BPM };
