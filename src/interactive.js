import { regl } from './render'
import { draw } from './shaders/draw'
import { FFT } from './shaders/fft'
import { WPM } from './shaders/wpm'
import { BPM } from './shaders/bpm'
import { SHAPE } from './shaders/shape'
import { sample } from './shaders/sample'
import { SmoothVar } from './smoothvar'
import { update_overlay } from './overlay'
// FFT domain
const levels = 7
const N = 2 ** levels
// const NH = Math.round(N / 4)
const NH = Math.round(N)

// resize 
regl._gl.canvas.width = N
regl._gl.canvas.height = N


const domain_size = 30  // original
// const domain_size = 3000  //beam prop works better at larger domain...
const dx = domain_size / N
const dz = domain_size / NH

const lensParams = {
    z: 0.5,  // Lens is placed at the center (normalized between 0 and 1)
    radius: 0.9,  // Lens curvature
    refractiveIndex: 1.5,  // Default glass
};

// beam launch parameters
const parameters = {
    width: new SmoothVar(0.7, domain_size/N*2, 0.7),
    power: new SmoothVar(-0.0, -1/5, 1/5),
    colormode: new SmoothVar(1, 0, 1)
}


let temp_fbo = regl.framebuffer({
    color: [
        // regl.texture({ type: "float", width: N, height: NH, format: "rgba"}),
        regl.texture({ width: N, height: NH, format: "rgba", mag:"nearest", min:"nearest"}),
    ]
})


const rgb_fbos = ['r', 'g', 'b'].map(x => regl.framebuffer({
    color: [
        // regl.texture({ type: "float", width: N, height: NH, format: "rgba"}),
        regl.texture({ width: N, height: NH, format: "rgba", mag:"nearest", min:"nearest" }),
    ]
}))

const rgb_fbos_mag = ['r', 'g', 'b'].map(x => regl.framebuffer({
    color: [
        // regl.texture({ type: "float", width: N, height: NH, format: "rgba"}),
        // regl.texture({ width: N, height: NH, format: "rgba", mag: "linear" }),
        regl.texture({ width: N, height: NH, format: "rgba", mag:"linear", min:"linear" }),
    ]
}))


// Main loop
rgb_fbos.map(x => x.use(function () {
    regl.clear({
        color: [0, 0, 0, 0],
        depth: 1,
    })
}))


let mx = 0
let my = 0
let m_down = false
let mdx = 0
let mdy = 0
let mpx = 0
let mpy = 0

let phase = 0
let block_light = false
let N_color = 3

// fps tracking
const SIMULATION_STEPS_PER_FRAME = 10; // Run 4 simulation steps per rendered frame
let lastRenderTime = performance.now();
let frameCount = 0;
let fps = 0;


// Slab parameters
const slabParams = {
    x_center: 0.5, // Center of the domain
    z_center: 0.25, // Middle between source and lens
    thickness: 0.05,
    length: 0.7,
    angle: 20,
    refractiveIndex: 1.5,
};


function stepSimulation() {
    for (let i = 0; i < 3; i++) {
        const wavelength = [0.63, 0.532, 0.47][i];
        const k0 = Math.PI * 2 / wavelength;

        let output = SHAPE(rgb_fbos[i], temp_fbo, NH, k0 * domain_size,
            parameters.width.value, parameters.power.value * domain_size,
            block_light ? mx : -1, block_light ? my : -1,
            lensParams.z, lensParams.radius, lensParams.refractiveIndex);
        rgb_fbos[i] = output[0];
        temp_fbo = output[1];

        // Wave propagation method
        // output = FFT(rgb_fbos[i], temp_fbo, levels, N, 1);
        // rgb_fbos[i] = output[0];
        // temp_fbo = output[1];

        // output = WPM(rgb_fbos[i], temp_fbo, N, k0, dz, dx, slabParams);
        // rgb_fbos[i] = output[0];
        // temp_fbo = output[1];

        // output = FFT(rgb_fbos[i], temp_fbo, levels, N, -1);
        // rgb_fbos[i] = output[0];
        // temp_fbo = output[1];


        // Beam propagation method
        output = BPM(rgb_fbos[i], temp_fbo, N, k0 , dz, dx, 1.0, refractiveIndexTex);
        rgb_fbos[i] = output[0];
        temp_fbo = output[1];

        rgb_fbos_mag[i].use(function () {
            regl.clear({ depth: 1 });
            sample({ texture: rgb_fbos[i] });
        });
    }

    phase = (phase - Math.PI / 200) % (Math.PI * 2);   // step the phase. Fake. Just for plotting
}

function createRefractiveIndexTexture(regl, N) {
    const data = new Uint8Array(N * N);
    const center = 0.55;
    const width = 0.05; // Transition width for smoothing

    for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {            
            let x = i / N;
            let transition = (Math.abs(x - center) - width) / width;
            let n = 1.0 + (1.01 - 1.0) * Math.exp(-transition * transition * 4.0); // Gaussian smoothing inside waveguide

            // Encode refractive index into 8-bit range (assuming max n = 5.0)
            let index = j * N + i;
            data[index] = Math.round((n / 5.0) * 255);
        }
    }

    return regl.texture({
        data,
        width: N,
        height: N,
        format: 'alpha',
        type: 'uint8',
        wrapS: 'clamp',
        wrapT: 'clamp',
        min: 'nearest',
        mag: 'nearest'
    });
}


// if (!regl.hasExtension('OES_texture_float')) {
//     throw new Error('Floating point textures are not supported on this device!');
// }


// Create refractive index texture
const refractiveIndexTex = createRefractiveIndexTexture(regl, N);

let lastMouseX = 0;
let lastMouseY = 0;

function update() {
    // Run the simulation multiple times before rendering
    for (let i = 0; i < SIMULATION_STEPS_PER_FRAME; i++) {
        stepSimulation();
    }

    // Apply user interaction updates **only once per render**
    if (m_down) {
        let dx = mx - lastMouseX;
        let dy = my - lastMouseY;
        parameters.width.add(dx * 0.7);
        parameters.power.add(dy * -0.4);
    }

    for (let i in parameters) {
        parameters[i].update();
    }

    lastMouseX = mx;
    lastMouseY = my;

    // Render only at monitor refresh rate (60 FPS)
    let now = performance.now();
    frameCount++;

    if (now - lastRenderTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastRenderTime = now;
        document.getElementById("fpsCounter").textContent = `FPS: ${fps}`;
    }

    draw({
        textureR: rgb_fbos_mag[0].color[0],
        textureG: rgb_fbos_mag[1].color[0],
        textureB: rgb_fbos_mag[2].color[0],
        phase: phase,
        colormode: parameters.colormode.value
    });

    update_overlay(parameters.width.value, parameters.power.value, 
        domain_size, lensParams, slabParams);
}


window.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect();
    const insideCanvas =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;

    if (!insideCanvas) return; // Do not update if outside canvas

    block_light = true;
    mx = (event.clientX - rect.left) / rect.width;
    my = 1 - (event.clientY - rect.top) / rect.height;
});

window.addEventListener('mousedown', (event) => {
    const rect = regl._gl.canvas.getBoundingClientRect();
    const insideCanvas =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;

    if (!insideCanvas) return; // Ignore clicks outside the canvas

    m_down = true;
    mpx = mx = (event.clientX - rect.left) / rect.width;
    mpy = my = 1 - (event.clientY - rect.top) / rect.height;
});


window.addEventListener('mouseup', () => {
    m_down = false
})

// Touch screen
const canvas = regl._gl.canvas
canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
canvas.addEventListener('touchend', handleTouchEnd, { passive: false });

function handleTouchStart(e) {
    e.preventDefault();
    m_down = true;
    block_light = true;
    const rect = canvas.getBoundingClientRect();
    mpx=mx = (e.touches[0].clientX - rect.left) / rect.width;
    mpy=my = 1 - (e.touches[0].clientY - rect.top) / rect.height;
}

function handleTouchMove(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    mx = (e.touches[0].clientX - rect.left) / rect.width;
    my = 1 - (e.touches[0].clientY - rect.top) / rect.height;
}

function handleTouchEnd(e) {
    e.preventDefault();
    m_down = false;
    block_light = false
}


window.regl = regl
window.parameters = parameters

document.getElementById('toggleButton').addEventListener('click', function(){
    parameters.colormode.set(1-parameters.colormode.target)
})


// Example: Update lens properties on user interaction
document.getElementById("lensControl").addEventListener("input", (event) => {
    lensParams.radius = parseFloat(event.target.value);
    // updateLens();
});


export { update }