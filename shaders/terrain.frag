#version 330 core
uniform float time;
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;

uniform sampler2D tex0;  // sand
uniform sampler2D tex1;  // grass
uniform sampler2D tex2;  // rock

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

uniform float minHeight;
uniform float maxHeight;

void main() {
    vec2 texCoord = FragPos.xy * 20.0;
    float specularStrength = 0.3;
    float shininess = 4.0;
    vec4 final_color;

    // Renderowanie lądu
    float height = FragPos.z;
    float normHeight = clamp((height - minHeight) / (maxHeight - minHeight), 0.0, 1.0);
    float slope = 1.0 - dot(normalize(Normal), vec3(0.0, 0.0, 1.0));
    vec4 color_sand = texture(tex0, texCoord);
    vec4 color_grass = texture(tex1, texCoord);
    vec4 color_rock = texture(tex2, texCoord);
    float w_sand = clamp((1.0 - normHeight * 10.0) * (1.0 - slope * 0.14), 0.0, 1.0);
    float w_grass = clamp((1.0 - abs(normHeight - 0.3) * 5.0) * (1.0 - slope * 0.14), 0.0, 1.0);
    float w_rock = clamp((normHeight - 0.3) * 5.0 * (0.7 + slope * 0.14), 0.0, 1.0);
    float sum = w_sand + w_grass + w_rock;
    if (sum < 0.001) sum = 1.0;
    w_sand /= sum;
    w_grass /= sum;
    w_rock /= sum;
    final_color = color_sand * w_sand + color_grass * w_grass + color_rock * w_rock;

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
    FragColor = vec4(result, 1.0);
}