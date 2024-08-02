import os
import torch

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
import sys


import utils
import pandas as pd
from tqdm import tqdm
import argparse

def main(args):

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")    
    
    target = utils.ReadSurf(args.target)
    target, target_mean_bb, target_scale_factor = utils.ScaleSurf(target)
    target_v, target_f, target_e = utils.PolyDataToTensors(target, device=device)
    target_mesh = Meshes(verts=[target_v], faces=[target_f])

    source = utils.IcoSphere(args.subdivision_level)
    source_v, source_f, source_e = utils.PolyDataToTensors(source, device=device)
    source_mesh = Meshes(verts=[source_v], faces=[source_f])
    
    deform_verts = torch.full(source_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    NPoints = source_v.shape[0]
    # Number of optimization steps
    Niter = args.n_iter
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 
    # Plot period for the losses
    plot_period = 250
    loop = tqdm(range(Niter))

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_source_mesh = source_mesh.offset_verts(deform_verts)
        
        # We sample 5k points from the surface of each mesh 
        sample_target = sample_points_from_meshes(target_mesh, NPoints)
        # sample_source = sample_points_from_meshes(new_source_mesh, NPoints)
        
        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_target, new_source_mesh.verts_packed().unsqueeze(0))
        
        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_source_mesh)
        
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_source_mesh)
        
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_source_mesh, method="uniform")
        
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
        
        # Print the losses
        loop.set_description('total_loss = %.6f' % loss)
        
        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
            
        # Optimization step
        loss.backward()
        optimizer.step()


    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    df = pd.DataFrame({
        "chamfer": chamfer_losses,
        "edge": edge_losses,
        "normal": normal_losses,
        "laplacian": laplacian_losses,
    })    

    df.to_csv(os.path.splitext(args.output)[0] + "_losses.csv", index=False)

    new_source_mesh_v = (new_source_mesh.verts_packed().detach().cpu())/target_scale_factor + target_mean_bb

    new_source_mesh_surf = utils.TensorToPolyData(new_source_mesh_v, new_source_mesh.faces_packed().detach().cpu())

    if args.smooth:
        new_source_mesh_surf = utils.SmoothPolyData(new_source_mesh_surf, args.smooth_iter, args.smooth_relaxation_factor)
        
    utils.WriteSurf(new_source_mesh_surf, args.output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--subdivision_level", type=int, help="subdivision level", default=6)
    parser.add_argument("--n_iter", type=int, help="Number of iteration steps", default=20000)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1.5)
    parser.add_argument("--target", type=str, help="target mesh")
    parser.add_argument("--output", type=str, help="output mesh")
    parser.add_argument("--smooth", type=int, default=0, help="smooth the output mesh")
    parser.add_argument("--smooth_iter", type=int, help="Number of smoothing iterations", default=15)
    parser.add_argument("--smooth_relaxation_factor", type=float, help="Relaxation factor for smoothing", default=0.1)
    args = parser.parse_args()
    main(args)



