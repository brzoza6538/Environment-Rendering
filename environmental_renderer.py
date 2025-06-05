import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import glm
from utils.map_loader import load_map
from PIL import Image

WIDTH = 1920
HEIGHT = 1080
SENSITIVITY = 0.05 # do rotacji

def load_texture(path, texture_unit):
    img = Image.open(path).convert('RGB')
    img_data = img.tobytes()
    width, height = img.size

    glActiveTexture(GL_TEXTURE0 + texture_unit)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex

def get_phong_shaders():
    vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 normal;

    out vec3 FragPos;
    out vec3 Normal;

    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;

    void main() {
        FragPos = vec3(model * vec4(position, 1.0));
        Normal = mat3(transpose(inverse(model))) * normal;
        gl_Position = projection * view * model * vec4(position, 1.0);
    }
    """

    fragment_shader = """
    #version 330 core
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
    // Normalizowanie wysokosci
    float height = FragPos.z;
    float normHeight = clamp((height - minHeight) / (maxHeight - minHeight), 0.0, 1.0);

    // Nachylenie (dla teksturowania)
    float slope = 1.0 - dot(normalize(Normal), vec3(0.0, 0.0, 1.0));

    // Skalowanie tekstur
    vec2 texCoord = FragPos.xy * 20.0;

    // Tekstury
    vec4 color_sand = texture(tex0, texCoord);
    vec4 color_grass = texture(tex1, texCoord);
    vec4 color_rock = texture(tex2, texCoord);

    // Wagi tekstur
    float w_sand = clamp((1.0 - normHeight * 10.0) * (1.0 - slope * 0.14), 0.0, 1.0);
    float w_grass = clamp((1.0 - abs(normHeight - 0.3) * 5.0) * (1.0 - slope * 0.14), 0.0, 1.0);
    float w_rock = clamp((normHeight - 0.3) * 5.0 * (0.7 + slope * 0.14), 0.0, 1.0);
          
    // Normalizacja wag
    float sum = w_sand + w_grass + w_rock;
    if (sum < 0.001) sum = 1.0;

    w_sand /= sum;
    w_grass /= sum;
    w_rock /= sum;

    // Blendowanie tekstur
    vec4 blended_color = color_sand * w_sand + color_grass * w_grass + color_rock * w_rock;

    // Oświetlenie Phong
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.3;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 4.0);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * blended_color.rgb;
    FragColor = vec4(result, 1.0);
}

    """
    return vertex_shader, fragment_shader

def main():
    vertices, indices = load_map()
    if vertices is None:
        return

    # Scale to normalized OpenGL range
    vertices = vertices / 100.0

    if not glfw.init():
        print("GLFW initialization failed")
        return

    window = glfw.create_window(WIDTH, HEIGHT, "Environmental Renderer", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window creation failed")
        return

    glfw.make_context_current(window)
    glViewport(0, 0, WIDTH, HEIGHT)
    glEnable(GL_DEPTH_TEST)

    texture_sand = load_texture("textures/sand.jpg", 0)
    texture_grass = load_texture("textures/grass.jpg", 1)
    texture_rock = load_texture("textures/rock.jpg", 2)

    normals = np.zeros_like(vertices)
    for i in range(0, len(indices), 3):
        v1 = vertices[indices[i]]
        v2 = vertices[indices[i + 1]]
        v3 = vertices[indices[i + 2]]

        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else normal

        normals[indices[i]] += normal
        normals[indices[i + 1]] += normal
        normals[indices[i + 2]] += normal

    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    NBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, NBO)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * normals.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    VERTEX_SHADER, FRAGMENT_SHADER = get_phong_shaders()

    def compile_shader(src, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode())
        return shader

    shader = glCreateProgram()
    vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    glAttachShader(shader, vs)
    glAttachShader(shader, fs)
    glLinkProgram(shader)
    glDeleteShader(vs)
    glDeleteShader(fs)

    minHeight = np.min(vertices[:, 2])
    maxHeight = np.max(vertices[:, 2])

    glUseProgram(shader)
    min_loc = glGetUniformLocation(shader, "minHeight")
    max_loc = glGetUniformLocation(shader, "maxHeight")

    # Wysokości do teksturowania (I może poziomu wody?)
    glUniform1f(min_loc, float(minHeight))
    glUniform1f(max_loc, float(maxHeight))

    aspect_ratio =  WIDTH / HEIGHT
    projection = glm.perspective(glm.radians(45.0), aspect_ratio, 0.1, 100.0)

    center = np.mean(vertices, axis=0)
    center = glm.vec3(center[0], center[1], center[2])
    radius = 3.0  # zmniejszony
    x_rotation = 0.0
    z_rotation = 0.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            x_rotation += 1.0*SENSITIVITY
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            x_rotation -= 1.0*SENSITIVITY
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            z_rotation += 1.0*SENSITIVITY
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            z_rotation -= 1.0*SENSITIVITY

        phi = glm.radians(x_rotation)
        theta = glm.radians(z_rotation)

        cam_x = center.x + radius * glm.cos(phi) * glm.cos(theta)
        cam_y = center.y + radius * glm.cos(phi) * glm.sin(theta)
        cam_z = center.z + radius * glm.sin(phi)

        camera_position = glm.vec3(cam_x, cam_y, cam_z)
        up = glm.vec3(0.0, 0.0, 1.0)

        view = glm.lookAt(camera_position, center, up)
        model = glm.mat4(1.0)

        glClearColor(0.1, 0.1, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        # Tekstury
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_sand)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, texture_grass)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, texture_rock)

        glUniform1i(glGetUniformLocation(shader, "tex0"), 0)
        glUniform1i(glGetUniformLocation(shader, "tex1"), 1)
        glUniform1i(glGetUniformLocation(shader, "tex2"), 2)


        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, np.array(projection.to_list(), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, np.array(view.to_list(), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, np.array(model.to_list(), dtype=np.float32))

        glUniform3f(glGetUniformLocation(shader, "lightPos"), center.x, center.y, center.z + 2.0)
        glUniform3f(glGetUniformLocation(shader, "viewPos"), camera_position.x, camera_position.y, camera_position.z)
        glUniform3f(glGetUniformLocation(shader, "lightColor"), 1.0, 1.0, 1.0)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteBuffers(1, [NBO])
    glfw.terminate()

def get_colors(vertices):
    color_keypoints = np.array([
    [1, 0, 1], 
    [0, 0, 1],  
    [0, 1, 0], 
    [1, 1, 0],
    [1, 0, 0],
    ], dtype=np.float32)

    z = vertices[:, 2]
    z_min, z_max = z.min(), z.max()
    normalized = (z - z_min) / (z_max - z_min)

    colors_matrix = np.zeros_like(vertices)
    segments = len(color_keypoints) - 1

    for i, t in enumerate(normalized):
        segment = min(int(t * segments), segments - 1)
        local_t = (t * segments) - segment

        c0 = color_keypoints[segment]
        c1 = color_keypoints[segment + 1]
        colors_matrix[i] = (1 - local_t) * c0 + local_t * c1
    
    return colors_matrix

if __name__ == "__main__":
    main()