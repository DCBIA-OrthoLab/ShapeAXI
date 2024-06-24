import vtk
import argparse 

def convert_stl_to_vtk(stl_filename, vtk_filename):
    # Create a reader for the STL file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_filename)

    # Create a writer for the VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_filename)
    
    # Connect the output of the reader directly to the input of the writer
    writer.SetInputConnection(reader.GetOutputPort())
    
    # Write the VTK file
    writer.Write()

def get_argparse():
    # Function to parse the arguments
    parser = argparse.ArgumentParser(description='Converts a STL file to a VTK file')
    parser.add_argument('--surf', type=str, help='Surface stl file', required=True)
    parser.add_argument('--out', type=str, help='Output', default="out.vtk")
    return parser

if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    convert_stl_to_vtk(args.surf, args.out)