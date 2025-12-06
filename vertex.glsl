#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

// flat qualifier ensures the value is not interpolated across fragments,
// so each triangle face will keep the same normal value in the fragment shader -> flat shading.
flat out vec3 fsNormal;
out vec3 fsPos; // world-space position

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;

void main() {
    vec3 worldPos = vec3(uModel * vec4(aPos, 1.0));
    fsPos = worldPos;
    fsNormal = normalize(uNormalMat * aNormal); // transformed face normal
    gl_Position = uMVP * vec4(aPos, 1.0);
}
