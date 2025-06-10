#version 330 core
uniform float time;
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;

uniform sampler2D tex_water; // water

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

uniform float minHeight;
uniform float maxHeight;

void main() {
    vec2 texCoord = FragPos.xy * 20.0;
    vec2 waveOffset = vec2(sin(time + texCoord.x) * 0.2, cos(time + texCoord.y) * 0.2);
    vec2 waterTexCoord = texCoord + vec2(time * 0.05, time * 0.05) + waveOffset;
    vec4 final_color;

    // Renderowanie wody
    final_color = texture(tex_water, waterTexCoord);
    float specularStrength = 0.9;
    float shininess = 128.0;
        
    // Oświetlenie Phong
    vec3 ambient = 0.3 * lightColor;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * final_color.rgb;
    FragColor = vec4(result, 0.75);
}