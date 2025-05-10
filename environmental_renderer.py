import glfw
from OpenGL.GL import*
from utils.map_loader import load_map

def main():
    points, distance, height = load_map()
    if points is None:
        return
    else:
        # print(f"points dimensions: {points.shape}")
        # print(f"distance: {points[1][2] - points[0][2]}")
        # print(f"max_height: {max([point[1] for point in points])}")
        if not glfw.init():
            print("Glfw library wasn't initialized")
            return
        window = glfw.create_window(1020, 980, "Environmental Renderer Window", None, None)
        if not window:
            glfw.terminate()
            print("Glfw window can't be created")
            exit()
        glfw.set_window_pos(window, 0, 40) 
        glfw.make_context_current(window)
        while not glfw.window_should_close(window):
            glfw.poll_events()
            glfw.swap_buffers(window)
        glfw.terminate()
        return

if __name__ == "__main__":
    main()
