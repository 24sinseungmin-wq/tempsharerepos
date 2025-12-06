#version 330 core
out vec4 FragColor;

flat in vec3 fsNormal;
in vec3 fsPos;

uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform vec3 uColor;
uniform int render_type;

void main() {
    vec3 color;
    switch (render_type)
    {
        case 1:
        {
            color=uColor;
            break;
        }
        default:
        {
        // simple Phong with flat normal (so lighting constant across each face)
            vec3 N = normalize(fsNormal);
            vec3 L = normalize(-uLightPos);
            vec3 V = normalize(uViewPos - fsPos);
            vec3 R = reflect(-L, N);

            float ambientStrength = 0.05;
            vec3 ambient = ambientStrength * uColor;

            float diff = max(dot(N, L), 0.0);
            vec3 diffuse = diff * uColor;

            float specularStrength = 0.1;   //1.0
            float shininess = 32.0; //32.0
            float spec = 0.0;
            spec = pow(max(dot(R, V), 0.0), shininess);
            vec3 specular = specularStrength * spec * vec3(1.0);

            color = ambient + diffuse + specular;
            break;
        }
    }

    FragColor = vec4(color, 1.0);
}
