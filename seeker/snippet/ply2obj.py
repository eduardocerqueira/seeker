#date: 2023-08-08T17:05:40Z
#url: https://api.github.com/gists/e68564d2ded295d6933a3822c98d04ae
#owner: https://api.github.com/users/jotix16

"""
Simple script to convert ply to obj models
"""
import pymeshlab
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_dir', help='Path to ycb dataset models directory containing .ply files and .png textures.')
    parser.add_argument('output_dir', help='Path to output directory where .obj files will be saved.')

    args = parser.parse_args()
    return args.input_dir, args.output_dir


def ply_path_to_obj_path(ply_path):
    """
    Replaces the .ply extension with .obj extension
    """
    return os.path.splitext(ply_path)[0] + '.obj'


def convert(ply_path, obj_path):
    """
    Converts the given .ply file to an .obj file
    """

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(ply_path)
    ms.save_current_mesh(obj_path)

def main():
    input_dir, output_dir = parse_args()

    for filename in os.listdir(input_dir):
        if filename.endswith('.ply'):
            ply_path = os.path.join(input_dir, filename)
            obj_path = os.path.join(output_dir, ply_path_to_obj_path(filename))
            print(f"Converting {ply_path} to {obj_path}")
            convert(ply_path, obj_path)

if __name__ == '__main__':
    # python dumb.py local_data/bop_datasets/ycbv/models isyhand_properties/objects/meshes
    main()