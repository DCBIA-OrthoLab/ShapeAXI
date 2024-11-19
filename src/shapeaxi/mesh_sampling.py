from pytorch3d.ops import (sample_points_from_meshes,
                           knn_points, 
                           knn_gather,
                           sample_farthest_points)

from pytorch3d.structures import Meshes
import utils
import numpy as np
import argparse
import pandas as pd
import os
import torch 

from saxi_transforms import EvalTransform

def sample_points(x, Ns):
    """
    Samples Ns points from each batch in a tensor of shape (Bs, N, F).

    Args:
        x (torch.Tensor): Input tensor of shape (Bs, N, F).
        Ns (int): Number of points to sample from each batch.

    Returns:
        torch.Tensor: Output tensor of shape (Bs, Ns, F).
    """
    Bs, N, F = x.shape

    # Generate random indices for sampling
    indices = torch.randint(low=0, high=N, size=(Bs, Ns), device=x.device).unsqueeze(-1)

    # Gather the sampled points
    x = knn_gather(x, indices).squeeze(-2).contiguous()

    return x, indices

def main(args):
    # Load the CSV file
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: The CSV file '{args.csv}' does not exist.")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Ensure the 'surf' column exists
    if 'surf' not in df.columns:
        print("Error: The CSV file must contain a column named 'surf'.")
        return

    idx = None
    concat_shapes = []
    for i, row in df.iterrows():
        # Construct the full file path
        file_path = os.path.join(args.mount_point, row['surf'])
        if not os.path.isfile(file_path):
            print(f"Warning: File '{file_path}' does not exist. Skipping.")
            continue

        # Read the VTK file
        try:
            surf = utils.ReadSurf(file_path)
            
            V, F = utils.PolyDataToTensors_v_f(surf)

            if args.idx_sampling > 0:
                if args.idx_fn is not None:
                    idx = np.load(args.idx_fn)
                    idx = torch.tensor(idx, dtype=torch.int64, device=V.device)

                if idx is None:
                    P, idx = sample_points(V.unsqueeze(0), args.num_samples)
                else:
                    P = knn_gather(V.unsqueeze(0), idx).squeeze(-2).squeeze(0).contiguous()
            else:
                mesh = Meshes(verts=V.unsqueeze(0), faces=F.unsqueeze(0))
                P = sample_points_from_meshes(mesh, args.num_samples*2)
                
                P, _ = sample_farthest_points(P, K=args.num_samples)
            
            if args.normalize:
                P = EvalTransform()(P)

            if args.scale != 1.0:
                P = P * args.scale

            if not args.concat:
                out_path = os.path.join(args.out, row['surf'].replace('.vtk', '.npy'))
                out_dir = os.path.dirname(out_path)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                np.save(out_path, P.squeeze(0).cpu().numpy())
            else:
                concat_shapes.append(P.unsqueeze(0).cpu())

        except Exception as e:
            print(f"Error reading VTK file '{file_path}': {e}")

    if args.concat:
        concat_shapes = torch.cat(concat_shapes)
        print(concat_shapes.shape)
        np.save(args.out, concat_shapes.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample surfaces")
    parser.add_argument("--csv", type=str, help="Path to the CSV file.", required=True)
    parser.add_argument("--num_samples", type=int, help="Number of samples", required=True)
    parser.add_argument("--idx_sampling", type=int, help="Sample using the first shape idx for the rest", default=0)
    parser.add_argument("--idx_fn", type=str, help="Sample using a precomputed indices", default=None)
    parser.add_argument("--mount_point", type=str, help="Mount point to prepend to file paths.", default="./")
    parser.add_argument("--factor", type=int, default=4, help="num_samples*factor for initial sampling, output is done with sample_farthest_points to ensure coverage")
    parser.add_argument("--normalize", type=int, default=0, help="Normalize the shapes")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the shapes")
    parser.add_argument("--concat", type=int, default=0, help="Concat the output in a single file")
    parser.add_argument("--out", type=str, default="./out", help="Output directory or filename if using --concat")

    args = parser.parse_args()

    main(args)