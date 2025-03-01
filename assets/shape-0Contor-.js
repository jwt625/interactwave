var n=`precision mediump float;
uniform vec4 color;
varying vec2 uv;

uniform float width;
uniform float power;
uniform sampler2D texture;
uniform vec2 mouse;
uniform int N;
uniform float k0;

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
    float uvx = uv.x - 0.5/float(N);
    float A = step(0.0, width*0.5-abs(uvx-0.5))/sqrt(width)*0.02;
    
    
    float cs2 = pow((uvx-0.5), 2.0) * power;
    float angle = k0 * cs2 / (1.0 + sqrt(1.0 - cs2*power));

    int n = int(uv.x * float(N));
    vec2 u = vec2(
        cos(angle),
        sin(angle)
    )*A;

    vec2 x = rgba2vec(texture2D(texture, uv));

    
    x = mix(
        x*0.999,
        u,
        step(1.0-1.0/float(N), uv.y)
    );

    float r = distance(uv, mouse);
    x *= float(r>0.03);

    
    float border = clamp(1.0-min(float(n), float(N-n-1))/20.0, 0.0, 1.0);
    x = x*exp(-border*0.1);

    

    gl_FragColor =vec2rgba(x);
}`;export{n as default};
