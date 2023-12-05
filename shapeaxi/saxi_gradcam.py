import argparse
import math
import os
import pandas as pd
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, HiResCAM #, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import pickle
from tqdm import tqdm
import monai
from monai.transforms import (    
    ScaleIntensityRange
)
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from . import saxi_nets, post_process as psp
from .saxi_dataset import SaxiDataset
from .saxi_transforms import TrainTransform, EvalTransform

# Loops over the folds to generate a visualization to explain what is happening in the network after the evaluation part of the training is done.
# Especially identify the parts of the picture which is the most important for the network to make a decision.


## Gradcam function for Regression and Classification model
def Classification_Regression_gradcam(args):
    fname = os.path.basename(args.csv_test)    
    ext = os.path.splitext(fname)[1]

    # Read of the test data from a CSV or Parquet file
    if ext == ".csv":
        df_test = pd.read_csv(args.csv_test)
    else:
        df_test = pd.read_parquet(args.csv_test)

    # The dataset and corresponding data loader are initialized for evaluation purposes.
    test_ds = SaxiDataset(df_test, transform=EvalTransform(), **vars(args))
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    
    model.ico_sphere(radius=args.radius, subdivision_level=args.subdivision_level)

    device = torch.device('cuda')
    model = model.to(device)

    model.eval()

    target_layer = getattr(model.F.module, args.target_layer)

    target_layers = None    
    if isinstance(target_layer, nn.Sequential):
        target_layer = target_layer[-1]

        target_layers = [target_layer]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = None
    if not args.target_class is None:
        targets = [ClassifierOutputTarget(args.target_class)]

    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    out_dir = os.path.join(os.path.dirname(args.csv_test), "grad_cam", str(args.target_class))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
        V = V.cuda(non_blocking=True)
        F = F.cuda(non_blocking=True)
        CN = CN.cuda(non_blocking=True)

        X, PF = model.render(V, F, CN)
        gcam_np = cam(input_tensor=X, targets=targets)

        GCAM = torch.tensor(gcam_np).to(device)

        P_faces = torch.zeros(1, F.shape[1]).to(device)
        V_gcam = -1*torch.ones(V.shape[1], dtype=torch.float32).to(device)

        for pf, gc in zip(PF.squeeze(), GCAM):
            P_faces[:, pf] = torch.maximum(P_faces[:, pf], gc)

        faces_pid0 = F[0,:,0].to(torch.int64)
        V_gcam[faces_pid0] = P_faces

        surf = test_ds.getSurf(idx)

        V_gcam = numpy_to_vtk(V_gcam.cpu().numpy())
        if not args.target_class is None:
            array_name = "grad_cam_target_class_{target_class}".format(target_class=args.target_class)
        else:
            array_name = "grad_cam_max"
        V_gcam.SetName(array_name)
        surf.GetPointData().AddArray(V_gcam)

        # Median filtering is applied to smooth the CAM on the surface
        psp.MedianFilter(surf, V_gcam)

        surf_path = df_test.loc[idx][args.surf_column]
        ext = os.path.splitext(surf_path)[1]

        if ext == '':
            ext = ".vtk"
            surf_path += ext

        out_surf_path = os.path.join(out_dir, surf_path)

        if not os.path.exists(os.path.dirname(out_surf_path)):
            os.makedirs(os.path.dirname(out_surf_path))

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(out_surf_path)
        writer.SetInputData(surf)
        writer.Write()


        X = (X*(PF>=0)).cpu().numpy()        
        vid_np = scale_intensity(X).permute(0,1,3,4,2).squeeze().cpu().numpy().squeeze().astype(np.uint8)        
        gcam_np = scale_intensity(gcam_np).squeeze().numpy().astype(np.uint8)

        
        # out_vid_path = surf_path.replace(ext, '.mp4')
        out_vid_path = surf_path.replace(ext, '.avi')
        
        out_vid_path = os.path.join(out_dir, out_vid_path)

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # The video is generated with a specified frames-per-second rate
        out = cv2.VideoWriter(out_vid_path, fourcc, args.fps, (256, 256))

        for v, g in zip(vid_np, gcam_np):
            c = cv2.applyColorMap(g, cv2.COLORMAP_JET)            
            b = cv2.addWeighted(v[:,:,0:3], 0.5, c, 0.5, 0)
            out.write(b)

        out.release()


def main(args):
    if args.nn == 'SaxiClassification' or args.nn == 'SaxiRegression':
        Classification_Regression_gradcam(args)


def get_argparse():
    # The arguments are defined for the script 
    parser = argparse.ArgumentParser(description='Saxi GradCam')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)   
    input_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    input_group.add_argument('--class_column', help='Class column name', type=str, default="class")
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")

    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', help='Model for prediction', type=str, required=True)    
    model_group.add_argument('--target_layer', help='Target layer for GradCam. For example in ResNet, the target layer is the last conv layer which is layer4', type=str, default='layer4')
    model_group.add_argument('--target_class', help='Target class', type=int, default=None)
    model_group.add_argument('--nn', help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification', type=str, default='SaxiClassification')


    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    hyper_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=2)
  
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--fps', help='Frames per second', type=int, default=24)    

    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    main(args)

