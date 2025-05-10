import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(prog="Environmental Renderer", description="Reads .bmp or .png height map, distance between points, map height from console arguments and renders environment in openGL window.", epilog="Program prepared by Michał Brzeziński, Michał Krasnodębski, Stanisław Sieniawski and Aleksandra Szymańska for Computer Graphics course at Warsaw University of Technology.")

    parser.add_argument("-M", "--map", type=str, default='./maps/example.bmp', help="height map filepath .bmp or .png format")
    parser.add_argument("-D", "--distance", type=float, default=1, help="x or z distance between points")
    parser.add_argument("-H", "--height", type=float, default=255, help="map height")

    return parser.parse_args()
