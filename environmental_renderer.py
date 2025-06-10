import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import glm
from utils.map_loader import load_map
from PIL import Image


WIDTH = 1920
HEIGHT = 1080
SPAWN_HEIGHT = 0.5
SENSITIVITY = 0.05 # do rotacji
SPEED = 0.005


lastX = WIDTH / 2
lastY = HEIGHT / 2
yaw = 0.0
pitch = 0.0
first_mouse = True


def load_shaders(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def mouse_callback(window, xpos, ypos):
    global lastX, lastY, yaw, pitch, first_mouse, camera_front

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = lastX - xpos
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    xoffset *= SENSITIVITY
    yoffset *= SENSITIVITY

    yaw += xoffset
    pitch += yoffset

    pitch = max(-89.0, min(89.0, pitch))

    direction = glm.vec3()
    direction.x = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
    direction.y = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    direction.z = np.sin(np.radians(pitch))
    camera_front = glm.normalize(direction)


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


def main():
    global camera_position, camera_front, camera_up

    if not glfw.init():
        print("GLFW initialization failed")
        return
    start_time = glfw.get_time()

    vertices, indices, water_height = load_map()
    if vertices is None:
        return

    vertices = vertices / 100.0

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


    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])

    quad_vertices = np.array([
        min_x, min_y, water_height, 0.0, 0.0,
        max_x, min_y, water_height, 1.0, 0.0,
        min_x, max_y, water_height, 0.0, 1.0,
        max_x, max_y, water_height, 1.0, 1.0,
    ], dtype=np.float32)

    quad_indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)

    quad_normals = np.array([
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
    ], dtype=np.float32)


    center = glm.vec3(np.mean(vertices[:,0]), np.mean(vertices[:,1]), np.mean(vertices[:,2]))
    camera_position = glm.vec3(center.x, center.y, center.z + SPAWN_HEIGHT)  # 20 jednostek ponad środek
    camera_front = glm.normalize(center - camera_position)
    camera_up = glm.vec3(0.0, 0.0, 1.0)


    window = glfw.create_window(WIDTH, HEIGHT, "Environmental Renderer", None, None)
    if not window:
        glfw.terminate()
        print("GLFW window creation failed")
        return

    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glViewport(0, 0, WIDTH, HEIGHT)
    glEnable(GL_DEPTH_TEST)


    texture_sand = load_texture("textures/sand.jpg", 0)
    texture_grass = load_texture("textures/grass.jpg", 1)
    texture_rock = load_texture("textures/rock.jpg", 2)
    texture_water = load_texture("textures/water.jpg", 3)


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

    water_VAO = glGenVertexArrays(1)
    water_VBO = glGenBuffers(1)
    water_EBO = glGenBuffers(1)
    water_NBO = glGenBuffers(1)

    glBindVertexArray(water_VAO)


    glBindBuffer(GL_ARRAY_BUFFER, water_VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, water_NBO)
    glBufferData(GL_ARRAY_BUFFER, quad_normals.nbytes, quad_normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, water_EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    glBindVertexArray(0)


    VERTEX_SHADER, FRAGMENT_SHADER = load_shaders("shaders/terrain.vert"), load_shaders("shaders/terrain.frag")
    WATER_VERTEX_SHADER, WATER_FRAGMENT_SHADER = load_shaders("shaders/water.vert"), load_shaders("shaders/water.frag")


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


    water_shader = glCreateProgram()
    wvs = compile_shader(WATER_VERTEX_SHADER, GL_VERTEX_SHADER)
    wfs = compile_shader(WATER_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    glAttachShader(water_shader, wvs)
    glAttachShader(water_shader, wfs)
    glLinkProgram(water_shader)
    glDeleteShader(wvs)
    glDeleteShader(wfs)


    minHeight = np.min(vertices[:, 2])
    maxHeight = np.max(vertices[:, 2])

    glUseProgram(shader)
    min_loc = glGetUniformLocation(shader, "minHeight")
    max_loc = glGetUniformLocation(shader, "maxHeight")

    # Wysokości do teksturowania
    glUniform1f(min_loc, float(minHeight))
    glUniform1f(max_loc, float(maxHeight))

    aspect_ratio =  WIDTH / HEIGHT
    projection = glm.perspective(glm.radians(45.0), aspect_ratio, 0.1, 100.0)


    while not glfw.window_should_close(window):
        glfw.poll_events()

        current_time = glfw.get_time() - start_time
        glUniform1f(glGetUniformLocation(shader, "time"), current_time)

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            camera_position += SPEED * camera_front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            camera_position -= SPEED * camera_front
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            camera_position -= SPEED * glm.normalize(glm.cross(camera_front, camera_up))
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            camera_position += SPEED * glm.normalize(glm.cross(camera_front, camera_up))
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            camera_position += SPEED * camera_up
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            camera_position -= SPEED * camera_up
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        view = glm.lookAt(camera_position, camera_position + camera_front, camera_up)
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

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


        glUseProgram(water_shader)
        glUniform1f(glGetUniformLocation(water_shader, "time"), current_time)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_water)
        glUniform1i(glGetUniformLocation(water_shader, "tex_water"), 0)

        glUniformMatrix4fv(glGetUniformLocation(water_shader, "projection"), 1, GL_FALSE, np.array(projection.to_list(), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(water_shader, "view"), 1, GL_FALSE, np.array(view.to_list(), dtype=np.float32))
        glUniformMatrix4fv(glGetUniformLocation(water_shader, "model"), 1, GL_FALSE, np.array(glm.mat4(1.0).to_list(), dtype=np.float32))

        glUniform3f(glGetUniformLocation(water_shader, "lightPos"), center.x, center.y, center.z + 2.0)
        glUniform3f(glGetUniformLocation(water_shader, "viewPos"), camera_position.x, camera_position.y, camera_position.z)
        glUniform3f(glGetUniformLocation(water_shader, "lightColor"), 1.0, 1.0, 1.0)

        glBindVertexArray(water_VAO)
        glDrawElements(GL_TRIANGLES, len(quad_indices), GL_UNSIGNED_INT, None)


        glfw.swap_buffers(window)


    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteBuffers(1, [NBO])

    glDeleteVertexArrays(1, [water_VAO])
    glDeleteBuffers(1, [water_VBO])
    glDeleteBuffers(1, [water_EBO])


    glfw.terminate()


if __name__ == "__main__":
    main()
