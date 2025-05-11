import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import glm
from utils.map_loader import load_map

WIDTH = 1920
HEIGHT = 1080
SENSITIVITY = 0.05 # do rotacji

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

    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    void main() {
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;

        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;

        vec3 result = (ambient + diffuse + specular) * objectColor;
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
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, np.array(projection.to_list(), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, np.array(view.to_list(), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, np.array(model.to_list(), dtype=np.float32))

        glUniform3f(glGetUniformLocation(shader, "lightPos"), center.x, center.y, center.z + 2.0)
        glUniform3f(glGetUniformLocation(shader, "viewPos"), camera_position.x, camera_position.y, camera_position.z)
        glUniform3f(glGetUniformLocation(shader, "lightColor"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(shader, "objectColor"), 1.0, 1.0, 1.0)  # Biały dla testu

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteBuffers(1, [NBO])
    glfw.terminate()

if __name__ == "__main__":
    main()