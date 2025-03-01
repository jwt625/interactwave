import REGL from 'regl'

// create REGL
let regl = REGL({
    // extensions: ['OES_texture_float'],
    container: document.querySelector('#canvas'),
})

// default rectangle mesh
const rect = regl.buffer([
    [-1, -1],
    [1, -1],
    [1, 1],
    [1, 1],
    [-1, 1],
    [-1, -1],
])


export {
    regl, rect
}