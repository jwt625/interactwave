var n=`precision mediump float;
varying vec2 uv;

uniform int N;
uniform sampler2D texture;
uniform float k0;
uniform float dx;
uniform float dz;

#define PI 3.14159265358979

vec2 rgba2vec(vec4 rgba){
    float mag = rgba.r + rgba.g / 255.0;
    vec2 d = normalize(rgba.ba - 0.5);
    return mag * d;
}

vec4 vec2rgba(vec2 vec){

    float mag = length(vec)*255.0;
    float scale = max(abs(vec.x), abs(vec.y));
    float dx = vec.x / scale / 2.0;
    float dy = vec.y / scale / 2.0;

    float mag0 = floor(mag);

    return vec4(
        mag0 / 255.0,
        mag - mag0,
        dx+0.5,
        dy+0.5
    );
}

void main(){
    int n = int(uv.x * float(N));
    vec2 offset = vec2(0, 1.0/float(N)*dz/dx);
    vec2 x = rgba2vec(texture2D(texture, uv + offset));
    
    

    float kx = (mod(float(n)/float(N)+0.5, 1.0)-0.5)/dx*2.0*PI;
    float kz = sqrt(k0*k0-kx*kx);
    
    float angle = (kz * dz);

    

    vec2 W = vec2(
        cos(angle),
        sin(angle)
    );

    x = vec2(
        x.x*W.x - x.y*W.y,
        x.x*W.y + x.y*W.x
    ); 

    gl_FragColor = vec2rgba(x);
    
}`;export{n as default};
