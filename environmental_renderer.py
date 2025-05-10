import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import glm  
from utils.map_loader import load_map

def main():
    vertices, indices = load_map()
    if vertices is None:
        return

    if not glfw.init():
        print("GLFW initialization failed")
        return

    window = glfw.create_window(1020, 980, "Environmental Renderer", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window creation failed")
        return

    glfw.make_context_current(window)
    glViewport(0, 0, 1020, 980)
    glEnable(GL_DEPTH_TEST)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)

    VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 position;

    uniform mat4 projection;
    uniform mat4 view;

    void main() {
        vec4 scaledPos = vec4(position.x / 100.0, position.y / 100.0, position.z / 100.0, 1.0);
        gl_Position = projection * view * scaledPos;
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(0.3, 0.7, 0.2, 1.0);  // Zielony kolor
    }
    """

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

    aspect_ratio = 1020.0 / 980.0
    projection = glm.perspective(glm.radians(45.0), aspect_ratio, 0.1, 100.0)

    center = np.mean(vertices, axis=0)
    center = glm.vec3(center[0] / 100.0, center[1] / 100.0, center[2] / 100.0)
    camera_position = center + glm.vec3(0.0, 0.0, 5.0)

    radius = 5.0
    x_rotation = 0.0
    z_rotation = 0.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            x_rotation += 1.0
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            x_rotation -= 1.0
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            z_rotation += 1.0
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            z_rotation -= 1.0

        phi = glm.radians(x_rotation)
        theta = glm.radians(z_rotation)

        cam_x = center.x + radius * glm.cos(phi) * glm.cos(theta)
        cam_y = center.y + radius * glm.cos(phi) * glm.sin(theta)
        cam_z = center.z + radius * glm.sin(phi)

        camera_position = glm.vec3(cam_x, cam_y, cam_z)

        up = glm.vec3(0.0, 0.0, 1.0)

        view = glm.lookAt(camera_position, center, up)


        glClearColor(0.1, 0.1, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)
        projection_loc = glGetUniformLocation(shader, "projection")
        view_loc = glGetUniformLocation(shader, "view")

        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, np.array(projection.to_list(), dtype=np.float32))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, np.array(view.to_list(), dtype=np.float32))

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glfw.terminate()

if __name__ == "__main__":
    main()
