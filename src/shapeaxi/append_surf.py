import argparse
import pandas as pd
import vtk
import utils
import os

def main():
    parser = argparse.ArgumentParser(description='Merge VTK surface files listed in a CSV into a single mesh.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with columns surf_fn')
    parser.add_argument('--groupby', type=str, default=None, help='group column name in csv.')
    parser.add_argument('--mount_point', type=str, default=None, help='Use a mount point to prepend to the surf_fn column in the csv.')
    parser.add_argument('--out', type=str, default='out.stl', help='Output file name or directory if using groupby (default: out.stl)')
    parser.add_argument('--out_ext', type=str, default='.stl', help='Output file name extension if using groupby (default: .stl)')
    args = parser.parse_args()


    # Read the CSV file to get filenames    
    df = pd.read_csv(args.csv)

    if args.groupby:        

        if not os.path.exists(args.out) or not os.path.isdir(args.out):
            os.makedirs(args.out)

        for name, group in df.groupby(args.groupby):
            # Create an append filter
            append_filter = vtk.vtkAppendPolyData()
            for idx, row in group.iterrows():
                if args.mount_point:
                    surf = utils.ReadSurf(os.path.join(args.mount_point, row['surf_fn']))
                else:
                    surf = utils.ReadSurf(row['surf_fn'])
                append_filter.AddInputData(surf)
            # Execute the append filter to combine the polydata
            append_filter.Update()

            # Write the result to a file
            utils.WriteSurf(append_filter.GetOutput(), os.path.join(args.out, name + args.out_ext))
            
    else:
        # Create an append filter
        append_filter = vtk.vtkAppendPolyData()
        for idx, row in df.iterrows():
            if args.mount_point:
                surf = utils.ReadSurf(os.path.join(args.mount_point, row['surf_fn']))
            else:
                surf = utils.ReadSurf(row['surf_fn'])
            append_filter.AddInputData(surf)

        # Execute the append filter to combine the polydata
        append_filter.Update()

        # Write the result to a file
        utils.WriteSurf(append_filter.GetOutput(), args.out)
        

if __name__ == "__main__":
    main()
