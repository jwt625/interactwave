import {regl, rect} from '../render'



// wpm shader
const wpm = regl({
    frag: (await import('./wpm.frag')).default,
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
        slab_angle: regl.prop('slab_angle'),
        slab_thickness: regl.prop('slab_thickness'),
        slab_length: regl.prop('slab_length'),
        slab_x_center: regl.prop('slab_x_center'),
        slab_z_center: regl.prop('slab_z_center'),
        slab_refractive_index: regl.prop('slab_refractive_index'),
    },
    count: 6,
});



function WPM(fbo, fbo_temp, N, k0, dz, dx, slabParams){
    fbo_temp.use(function(){{
        regl.clear({
            color: [0, 0, 0, 0],
            depth: 1,
        })
        wpm({
            N: N,
            texture: fbo.color[0],
            dz: dz,
            dx: dx,
            k0: k0,
            slab_angle: slabParams.angle,
            slab_thickness: slabParams.thickness,
            slab_length: slabParams.length,
            slab_x_center: slabParams.x_center,
            slab_z_center: slabParams.z_center,
            slab_refractive_index: slabParams.refractiveIndex,
        })
    }})
    const temp = fbo_temp
    fbo_temp = fbo
    fbo = temp
    return [fbo, fbo_temp]
}

export {WPM}