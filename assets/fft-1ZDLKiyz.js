var n=`precision mediump float;
varying vec2 uv;

uniform int N;
uniform int NP;
uniform sampler2D texture;
uniform float mul;
uniform float conj;

#define PI 3.14159265358979

int modI(int a, int b) {
    float m=float(a)-floor((float(a)+0.5)/float(b))*float(b);
    return int(floor(m+0.5));
}

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

    vec2 u = vec2(0);

    int n = int(uv.x * float(N));

    int n_A = modI(n, NP/2) + (n/NP) * NP / 2;
    int n_B = modI(n_A + N/2, N);

    vec2 x_A = rgba2vec(texture2D(texture, vec2(
        (float(n_A))/float(N),
        uv.y
    )));

    vec2 x_B = rgba2vec(texture2D(texture, vec2(
        (float(n_B))/float(N),
        uv.y
    )));

    float m = 1.0 - 2.0 * float(step(float(NP/2), float(modI(n, NP))));

    float angle = PI*2.0*fract(float(modI(n, NP/2))/float(NP));
    vec2 W = vec2(
        cos(angle),
        sin(angle) * conj
    );

    x_B = m*vec2(
        x_B.x*W.x - x_B.y*W.y,
        x_B.x*W.y + x_B.y*W.x
    );

    gl_FragColor = vec2rgba((x_A+x_B) * mul);
    
    
    
}`;export{n as default};
