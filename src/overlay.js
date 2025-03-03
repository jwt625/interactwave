let svg = document.querySelector('svg')

// console.log(svg)


function draw_arc(z, w, c, type){
    let sweep = 0
    if (c<0) sweep = 1

    if(w>Math.abs(1/c)){
        w = Math.abs(1/c)*0.99
    }
    
    let arc = document.createElementNS("http://www.w3.org/2000/svg", "path");
    let startX = 50 - w;
    let endX = 50 + w ;
    
    z = z - c*w**2 / (1+Math.sqrt(1-c**2*w**2))
    c = Math.abs(c)
    let d = `M ${startX},${z} A ${1/c} ${1/c} 0 0 ${sweep} ${endX} ${z}`;

    let o = Math.asin(w*c)/c
    if(c==0) {
        o = w
        d = `M ${startX},${z} L ${endX} ${z}`;
    }

    arc.setAttribute("d", d);
    arc.setAttribute("fill", "none");
    arc.setAttribute("stroke", "white");
    arc.setAttribute("stroke-opacity", 0.7)
    arc.setAttribute("stroke-width", 0.7)
    arc.setAttribute("stroke-linecap", "round")
    if(type=='dash'){
        arc.setAttribute("stroke-dasharray", "0.5 3")
        arc.setAttribute("stroke-dashoffset", -o+0.25)
        arc.setAttribute("stroke-opacity", 0.4)
    }
    svg.appendChild(arc);
}


function drawLensOverlay(lensParams) {
    let lens = document.createElementNS("http://www.w3.org/2000/svg", "ellipse");
    lens.setAttribute("cx", "50%");
    lens.setAttribute("cy", "50%");
    lens.setAttribute("rx", lensParams.radius * 50);
    lens.setAttribute("ry", "5");
    lens.setAttribute("stroke", "white");
    lens.setAttribute("fill", "none");
    lens.setAttribute("stroke-width", "0.2");

    svg.appendChild(lens);
}

function drawSlabOverlay(slabParams) {
    let slab = document.createElementNS("http://www.w3.org/2000/svg", "rect");

    // Convert slab length and thickness to overlay scale
    let slabWidth = slabParams.length * 100;  // Length in x-direction
    let slabHeight = slabParams.thickness * 100;  // Thickness in z-direction

    // Compute rotation center correctly before translation
    let angleRad = (Math.PI / 180) * slabParams.angle;
    let xCenter = slabParams.x_center * 100;
    let zCenter = slabParams.z_center * 100;
    

    // Set slab attributes
    slab.setAttribute("x", `${xCenter - slabWidth / 2}`);
    slab.setAttribute("y", `${zCenter - slabHeight / 2}`);
    slab.setAttribute("width", `${slabWidth}`);
    slab.setAttribute("height", `${slabHeight}`);
    slab.setAttribute("fill", "rgba(255, 255, 255, 0.3)");
    slab.setAttribute("transform", `rotate(${slabParams.angle}, ${xCenter}, ${zCenter})`);

    svg.appendChild(slab);
}

function update_overlay(width, power, domain_size, lensParams, slabParams){


    // Gaussian beam approximation?
    
    // zr = pi * w0**2 / wavelength
    // w0 = sqrt( zr * wl / pi )

    // w(z) = w0 sqrt( 1 + (z/zr)**2 )
    // w(z) = sqrt( (zr**2 * wl + z**2) / (zr * pi) )
    // w(z)**2 = (zr**2 * wl + z**2) / (zr * pi) 
    
    // R(z) = z ( 1 + (zr/z)**2 )
    // c(z) = 1/R(z) = z / (z**2 + zr**2)

    svg.innerHTML = ""

    const w1 = width*25
    const c1 = power
    const wl = 0.532
    const zri = wl/Math.PI/w1
    const zi = c1
    const X = zri**2 + zi**2
    const zr = zri/X
    const z = zi/X

    const s = 1/domain_size*100


    let z2 = z+1
    let w2 = Math.sqrt(wl/Math.PI) * Math.sqrt((zr**2+z2**2)/zr)
    let c2 = z2/(z2**2+zr**2)
    draw_arc((z2-z)*s, Math.min(Math.max(1, w2), domain_size/2-1)*s, c2/s, 'dash')

    z2 = z+domain_size-1
    w2 = Math.sqrt(wl/Math.PI) * Math.sqrt((zr**2+z2**2)/zr)
    c2 = z2/(z2**2+zr**2)
    draw_arc((z2-z)*s, Math.min(Math.max(1, w2), domain_size/2-1)*s, c2/s, 'dash')

    let z2_1 = z+domain_size-2
    let z2_2 = z+2

    let f = 1/(1+Math.exp(40*power+8))
    // console.log(power)
    // f = 1
    z2 = (1-f)*z2_1

    f = 1/(1+Math.exp(40*power+4))
    z2 = (1-f)*z2 + f*z2_2
    w2 = Math.sqrt(wl/Math.PI) * Math.sqrt((zr**2+z2**2)/zr)
    c2 = z2/(z2**2+zr**2)
    draw_arc((z2-z)*s, Math.min(Math.max(1, w2), domain_size/2-2)*s, c2/s)
    // console.log((z2-z)*s)

    // add lens
    drawLensOverlay(lensParams)

    // add slab
    // drawSlabOverlay(slabParams);
}


export {update_overlay}