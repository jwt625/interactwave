vec4 colormap(float x) {
	vec4 color = mix(
		vec4(1, 0.1, 0, 1),
		vec4(0, 0.1, 1, 1),
		x+0.5
	) * abs(x)*2.0;
	color = mix(
		color,
		vec4(vec3(0.9),1),
		clamp((abs(x)-0.5)*2.0, 0.0, 1.0)
	);
	return color;
}