import vtk
import numpy as np 
import os
import pandas as pd
import argparse
import json
from tqdm import tqdm

from . import utils

# This file reads a CSV file, process its data to compute the minimum magnitude/scaling factor for all shapes, and save the results back to a CSV file

def main(args):
    surf_scales = []
    df = pd.read_csv(args.csv)
    pbar = tqdm(df.iterrows(), total=len(df))

    for idx, row in pbar:
        surf = utils.ReadSurf(row[args.surf_column])

        unit_surf, surf_mean, surf_scale = utils.ScaleSurf(surf)
        surf_scales.append(surf_scale)

        pbar.set_description('surf_scale={surf_scale}'.format(surf_scale=surf_scale))

    surf_scales = np.array(surf_scales)

    df['surf_scale'] = surf_scales

    min_scale = np.min(surf_scales)

    if args.out:
        df.to_csv(args.out, index=False)

    print("MinScale:", min_scale)
    return min_scale

def get_argparse():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Computes the minimum magnitude/scaling factor for all shapes after centering each at 0.')
    parser.add_argument('--csv', type=str, help='CSV file with column surf', required=True)
    parser.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    parser.add_argument('--out', type=str, default=None, help='Output json filename')
    return parser

if __name__ == "__main__":
    
    parser = get_argparse()
    args = parser.parse_args()

    main(args)
