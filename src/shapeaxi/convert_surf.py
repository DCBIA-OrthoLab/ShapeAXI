import os
import argparse

import utils

def main(args):

    surf = utils.ReadSurf(args.surf)
    utils.WriteSurf(surf, args.out)
    

def get_argparse():
    # Function to parse the arguments
    parser = argparse.ArgumentParser(description='Teeth challenge convert OBJ files to VTK. It adds the label information to each VTK file')
    parser.add_argument('--surf', type=str, help='Input surface {.obj, .off, .stl, .vtp, .vtk}', required=True)
    parser.add_argument('--out', type=str, help='Output filename {.vtk, .stl}', default="out.vtk")
    return parser

if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)
