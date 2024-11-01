import pandas as pd
import os
import vtk
import json
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tqdm import tqdm
from pathlib import Path
import sys
import argparse

from shapeaxi import utils

#This files contains a function to convert OBJ files to VTK and add the label information to each VTK file

def main(args):
    csv_path = args.csv
    output_dir = args.out
    vtk_output_dir = os.path.join(output_dir, "teeth-grand_challenge_vtk")  # Folder for VTK files
    os.makedirs(vtk_output_dir, exist_ok=True)  # Create the output folder if it doesn't exist
    df = pd.read_csv(csv_path, dtype=str)

    LUT = np.array([33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 9, 10, 11, 12, 13, 14, 15,
                    16, 0, 0, 24, 23, 22, 21, 20, 19, 18, 17, 0, 0, 25, 26, 27, 28, 29, 30, 31, 32])

    pbar = tqdm(range(len(df)),desc='Converting...', total=len(df))
    all_data = []  # List to store all data

    for idx in pbar:
        model = df.iloc[idx]
        surf_path = model['surf']
        pbar.set_description(f'{surf_path}')
        label_path = model['label']
        split = model['split']
        reader = vtk.vtkOBJReader()
        reader.SetFileName(surf_path)
        reader.Update()
        surf = reader.GetOutput()

        verts = vtk_to_numpy(surf.GetPoints().GetData())
        faces = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        with open(label_path) as f:
            json_data = json.load(f)
        vertex_labels_FDI = np.array(json_data['labels'])
        vertex_labels = LUT[vertex_labels_FDI]

        vertex_instances = np.array(json_data['instances'])

        vertex_instances_vtk = numpy_to_vtk(vertex_instances)
        vertex_instances_vtk.SetName("instances")
        
        vertex_labels_vtk = numpy_to_vtk(vertex_labels)
        vertex_labels_vtk.SetName("UniversalID")

        surf.GetPointData().AddArray(vertex_labels_vtk)
        surf.GetPointData().AddArray(vertex_instances_vtk)

        file_basename = Path(surf_path).stem
        out_path = os.path.join(vtk_output_dir, f'{file_basename}.vtk') # Updated path

        # Add the data to the list
        all_data.append({'surf': out_path, 'label': label_path, 'split': split})

        utils.Write(surf, out_path, print_out=False)

    # Create a single CSV file with just the paths to the VTK objects
    vtk_paths = [data['surf'] for data in all_data]
    vtk_paths_df = pd.DataFrame({'surf': vtk_paths})
    vtk_paths_df.to_csv(os.path.join(output_dir, f'{os.path.splitext(args.csv)[0]}_vtk.csv'), index=False)

def get_argparse():
    # Function to parse the arguments
    parser = argparse.ArgumentParser(description='Teeth challenge convert OBJ files to VTK. It adds the label information to each VTK file')
    parser.add_argument('--csv', type=str, help='CSV with columns surf,label,split', required=True)
    parser.add_argument('--out', type=str, help='Output directory', default="./")
    return parser

if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)
