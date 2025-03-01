var n=`precision mediump float;
varying vec2 uv;
uniform float phase;
uniform sampler2D textureR;
uniform sampler2D textureG;
uniform sampler2D textureB;
uniform float colormode;

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
vec4 colormap(float x) {
	vec3 color = mix(
		vec3(1, 0.1, 0),
		vec3(0, 0.1, 1),
		x+0.5
	) * abs(x)*2.0;
	color = mix(
		color,
		vec3(0.9),
		clamp((abs(x)-0.5)*2.0, 0.0, 1.0)
	);
	return vec4(color, 1.0);
}

#define gamma 1.0/2.2
void main(){

    vec2 u = vec2(0);

    

    vec2 x = rgba2vec(texture2D(textureG, uv));
    vec2 xr = rgba2vec(texture2D(textureR, uv));
    vec2 xb = rgba2vec(texture2D(textureB, uv));

    
    

    vec2 W = vec2(
        cos(phase),
        sin(phase)
    );

    x = vec2(
        x.x*W.x - x.y*W.y,
        x.x*W.y + x.y*W.x
    ); 

    vec4 color = colormap(x.x*10.0);

    vec3 rgb = vec3(
        pow(xr.x*xr.x+xr.y*xr.y, gamma), 
        pow(x.x*x.x+x.y*x.y, gamma), 
        pow(xb.x*xb.x+xb.y*xb.y, gamma)
    )*10.0;

    color = mix(color, vec4(rgb, 1), colormode);

    
    gl_FragColor = color;
}`;export{n as default};
