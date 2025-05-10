import imageio
import numpy as np
from utils.argument_parser import parse_arguments

def load_map():
    arguments = parse_arguments()

    map_filepath = arguments.map
    distance = arguments.distance
    height = arguments.height

    try:
        image = imageio.imread(map_filepath, mode='F')
        z_grid = image / 255 * height  # Zamiana z_grid na wysokość (height)
        # z_grid = np.ones_like(image, dtype=np.float32)

        rows, cols = image.shape

        x = np.arange(0, rows * distance, distance)
        y = np.arange(0, cols * distance, distance)  # y staje się położeniem w poziomie

        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')  # zamieniamy y_grid z z_grid

        # Zmieniamy sposób układania punktów 3D, aby pasowały do zamiany y i z
        points_3d = np.stack([x_grid, y_grid, z_grid], axis=2).astype(np.float32)  # (x, z, y)

        indices = []
        print(cols)
        print(rows)
        for i in range(rows):
            for j in range(cols):
                centre = i * cols + j
                
                # Indeksy sąsiadów
                left = centre - 1  # Punkt po lewej stronie
                top = centre - cols  # Punkt powyżej
                bottom = centre + cols  # Punkt poniżej
                right = centre + 1  # Punkt po prawej stronie

                # Tworzenie trójkątów, sprawdzamy czy sąsiedzi istnieją
                if j > 0 and i > 0:  # dla punktu [a-1, b], [a, b-1], [a, b]
                    indices.extend([left, top, centre])  # pierwszy trójkąt

                if j < cols - 1 and i < rows - 1:  # dla punktu [a+1, b], [a, b+1], [a, b]
                    indices.extend([right, bottom, centre])  # drugi trójkąt

        return points_3d.reshape(-1, 3), np.array(indices, dtype=np.uint32)

    except Exception as e:
        print(f"Error loading the map file. {e}")
        return None, None
