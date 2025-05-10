import glfw
from OpenGL.GL import *
import numpy as np
import ctypes

glfw.init()
window = glfw.create_window(800, 600, "environment renderer", None, None)
glfw.make_context_current(window)

shader = glCreateProgram()
vs = glCreateShader(GL_VERTEX_SHADER)
fs = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(vs, "#version 330 core\nlayout (location = 0) in vec3 pos;\nvoid main() { gl_Position = vec4(pos, 1.0); }")
glShaderSource(fs, "#version 330 core\nout vec4 FragColor;\nvoid main() { FragColor = vec4(1, 0.5, 0.2, 1); }")
glCompileShader(vs)
glCompileShader(fs)
glAttachShader(shader, vs)
glAttachShader(shader, fs)
glLinkProgram(shader)
glDeleteShader(vs)
glDeleteShader(fs)

vertices = np.array([
     0.5,  0.5, 0.0,
     0.5, -0.5, 0.0,
    -0.5, -0.5, 0.0,
    -0.5,  0.5, 0.0
], dtype=np.float32)

indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.uint32)

VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
EBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
