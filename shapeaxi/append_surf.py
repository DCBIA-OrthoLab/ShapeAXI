import argparse
import pandas as pd
import vtk
import utils

def main():
    parser = argparse.ArgumentParser(description='Merge VTK surface files listed in a CSV into a single mesh.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file with columns surf_fn, group.')
    parser.add_argument('--out', type=str, default='out.stl', help='Output file name (default: out.stl)')
    args = parser.parse_args()


    # Create an append filter
    append_filter = vtk.vtkAppendPolyData()

    # Read the CSV file to get filenames    
    df = pd.read_csv(args.csv)
    for idx, row in df.iterrows():
        surf = utils.ReadSurf(row['surf_fn'])
        append_filter.AddInputData(surf)

    # Execute the append filter to combine the polydata
    append_filter.Update()

    # Write the result to a file
    utils.WriteSurf(append_filter.GetOutput(), args.out)

    print(f"Combined mesh written to {args.output_file}")

if __name__ == "__main__":
    main()
