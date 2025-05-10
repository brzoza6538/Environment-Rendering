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

        y_grid = image / 255 * height
        
        rows, cols = image.shape

        x = np.arange(0, rows * distance, distance)
        z = np.arange(0, cols * distance, distance)
        
        x_grid, z_grid = np.meshgrid(x, z, indexing='ij')
        
        points_3d = np.stack([x_grid, y_grid, z_grid], axis=2).astype(np.float32)
        
        # return points_3d
        return points_3d.reshape(-1, 3), distance, height

    except Exception as e:
        print(f"Error loading the map file. {e}")
        return None, None, None
