import imageio
import numpy as np
from utils.argument_parser import parse_arguments

def load_map():
    arguments = parse_arguments()

    map_filepath = arguments.map
    distance = arguments.distance
    height = arguments.height
    water_height = arguments.wheight

    try:
        image = imageio.imread(map_filepath, mode='F')
        z_grid = image / 255 * height

        rows, cols = image.shape

        x = np.arange(0, rows * distance, distance)
        y = np.arange(0, cols * distance, distance)

        x_grid, y_grid = np.meshgrid(x, y, indexing='ij') 

        points_3d = np.stack([x_grid, y_grid, z_grid], axis=2).astype(np.float32)

        indices = []

        for i in range(rows):
            for j in range(cols):
                centre = i * cols + j
                
                left = centre - 1
                top = centre - cols
                bottom = centre + cols
                right = centre + 1

                if j > 0 and i > 0:
                    indices.extend([left, top, centre])

                if j < cols - 1 and i < rows - 1:
                    indices.extend([right, bottom, centre])

        return points_3d.reshape(-1, 3), np.array(indices, dtype=np.uint32), water_height

    except Exception as e:
        print(f"Error loading the map file. {e}")
        return None, None
