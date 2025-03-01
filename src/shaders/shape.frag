precision mediump float;
uniform vec4 color;
varying vec2 uv;

uniform float width;
uniform float power;
uniform sampler2D texture;
uniform vec2 mouse;
uniform int N;

#include ./utils/pack.frag

void main(){
    float A = step(0.0, width*0.5-abs(uv.x-0.5))/sqrt(width)*0.02;
    // float A = exp(-pow((uv.x-0.5)/width*1.0, 2.0))/sqrt(width)*0.01;
    
    float angle = pow((uv.x-0.5), 2.0)*power;

    int n = int(uv.x * float(N));
    vec2 u = vec2(
        cos(angle),
        sin(angle)
    )*A;

    vec2 x = rgba2vec(texture2D(texture, uv));


    // vec2 u = vec2(step(0.0, -abs(uv.y-uv.x)))*0.3;
    x = mix(
        x*0.999,
        u,
        step(1.0-1.0/float(N), uv.y)
    );

    float r = distance(uv, mouse);
    x *= float(r>0.03);

    // border
    float border = clamp(1.0-min(float(n), float(N-n-1))/20.0, 0.0, 1.0);
    x = x*exp(-border*0.1);

    

    gl_FragColor =vec2rgba(x);
}