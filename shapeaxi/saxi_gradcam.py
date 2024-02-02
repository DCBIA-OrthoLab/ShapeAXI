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

import saxi_nets, post_process as psp
from saxi_dataset import SaxiDataset, BrainIBISDataset
from saxi_transforms import TrainTransform, EvalTransform, UnitSurfTransform, RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform


# Loops over the folds to generate a visualization to explain what is happening in the network after the evaluation part of the training is done.
# Especially identify the parts of the picture which is the most important for the network to make a decision.


## Gradcam function for Regression and Classification model
def SaxiClassification_Regression_gradcam(args):
    fname = os.path.basename(args.csv_test)    
    ext = os.path.splitext(fname)[1]

    # Read of the test data from a CSV or Parquet file
    if ext == ".csv":
        df_test = pd.read_csv(args.csv_test)
    else:
        df_test = pd.read_parquet(args.csv_test)
    
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    model.ico_sphere(radius=args.radius, subdivision_level=args.subdivision_level)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # The dataset and corresponding data loader are initialized for evaluation purposes.
    test_ds = SaxiDataset(df_test, transform=EvalTransform(), **vars(args))

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

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


#################################################################################### ICOCONV PART ##############################################################################################################

class Classification_for_left_path(nn.Module):
    def __init__(self,classification_layer,xR,demographic):
        super().__init__()
        self.classification_layer = classification_layer
        self.xR = xR
        self.demographic = demographic

    def forward(self,xL):
        l = [xL,self.xR,self.demographic]
        x = torch.cat(l,dim=1)
        x = self.classification_layer(x)
        return x

class Classification_for_right_path(nn.Module):
    def __init__(self,classification_layer,xL,demographic):
        super().__init__()
        self.classification_layer = classification_layer
        self.xL = xL
        self.demographic = demographic

    def forward(self,xR):
        l = [self.xL,xR,self.demographic]
        x = torch.cat(l,dim=1)
        x = self.classification_layer(x)     
        return x 


def SaxiIcoClassification_gradcam(args):     

    fname = os.path.basename(args.csv_test)    
    ext = os.path.splitext(fname)[1]

    # Read of the test data from a CSV or Parquet file
    if ext == ".csv":
        df_test = pd.read_csv(args.csv_test)
    else:
        df_test = pd.read_parquet(args.csv_test) 
    
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR
    list_path_ico = [args.path_ico_left,args.path_ico_right]

    test_ds = BrainIBISDataset(df_test,list_demographic,list_path_ico,transform=UnitSurfTransform())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    hemisphere = 'left'#'left','right'

    ico_lvl = 2
    if ico_lvl == 1:
        radius = 1.76 
    elif ico_lvl == 2:
        radius = 1

    list_path_ico = [args.path_ico_left,args.path_ico_right]

    ###Demographics
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR
    #List of used demographics 

    ###Transformation
    list_train_transform = []    
    list_train_transform.append(CenterTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())
    list_train_transform.append(GaussianNoisePointTransform(args.mean,args.std))
    list_train_transform.append(NormalizePointTransform())
    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterTransform())
    list_val_and_test_transform.append(NormalizePointTransform())
    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)


    list_nb_verts_ico = [12,42]
    nb_images = list_nb_verts_ico[ico_lvl-1]


    targets = None
    if not args.target_class is None:
        targets = [ClassifierOutputTarget(args.target_class)]

    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

        VL, FL, VFL, FFL, VR, FR, VFR, FFR, D, Y = batch 
        VL = VL.cuda(non_blocking=True)
        FL = FL.cuda(non_blocking=True)
        VFL = VFL.cuda(non_blocking=True)
        FFL = FFL.cuda(non_blocking=True)
        VR = VR.cuda(non_blocking=True)
        FR = FR.cuda(non_blocking=True)
        VFR = VFR.cuda(non_blocking=True)
        FFR = FFR.cuda(non_blocking=True)
        D = D.cuda(non_blocking=True)

        xL, PF = model.render(VL,FL,VFL,FFL)
        xR, PF = model.render(VR,FR,VFR,FFR)

        classification_layer = model.Classification

        if hemisphere == 'left':
            input_tensor_cam = xL
            xR = model.poolingR(model.IcosahedronConv2dR(model.TimeDistributedR(xR))) 
            classifier = Classification_for_left_path(classification_layer,xR,D)
            model_cam = nn.Sequential(model.TimeDistributedL, model.IcosahedronConv2dL, model.poolingL,classifier)
        else:
            input_tensor_cam = xR
            xL = model.poolingL(model.IcosahedronConv2dL(model.TimeDistributedL(xL))) 
            classifier = Classification_for_right_path(classification_layer,xL,D)
            model_cam = nn.Sequential(model.TimeDistributedR, model.IcosahedronConv2dR, model.poolingR,classifier)

        target_layers = [model_cam[0].module.layer4[-1]]
        cam = GradCAM(model=model_cam, target_layers=target_layers)


    grayscale_cam = torch.Tensor(cam(input_tensor=input_tensor_cam, targets=targets))

    name_save = 'grad_cam.pt'
    torch.save(grayscale_cam, args.out+"/"+name_save)




def main(args):
    if args.nn == 'SaxiClassification' or args.nn == 'SaxiRegression':
        SaxiClassification_Regression_gradcam(args)
    elif args.nn == 'SaxiIcoClassification':
        SaxiIcoClassification_gradcam(args) 



def get_argparse():
    # The arguments are defined for the script 
    parser = argparse.ArgumentParser(description='Saxi GradCam')

    ##Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_valid',  type=str, help='CSV with column surf')
    input_group.add_argument('--csv_train',  type=str, help='CSV with column surf')
    input_group.add_argument('--csv_test',  type=str, help='CSV with column surf', required=True)   
    input_group.add_argument('--surf_column',  type=str, help='Surface column name', default="surf")
    input_group.add_argument('--class_column',  type=str, help='Class column name', default="class")
    input_group.add_argument('--num_workers',  type=int, help='Number of workers for loading', default=4)
    input_group.add_argument('--mount_point',  type=str, help='Dataset mount directory', default="./")
    input_group.add_argument('--path_ico_left', type=str, help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)', default='./3DObject/sphere_f327680_v163842.vtk')
    input_group.add_argument('--path_ico_right', type=str, help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)', default='./3DObject/sphere_f327680_v163842.vtk')
    input_group.add_argument('--layer', type=str, help="Layer, choose between 'Att','IcoConv2D','IcoConv1D','IcoLinear' (default: IcoConv2D)", default='IcoConv2D')

    ##Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.35)    
    hyper_group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
    
    ##Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', type=str, help='Model for prediction', required=True)    
    model_group.add_argument('--target_layer', type=str, help='Target layer for GradCam. For example in ResNet, the target layer is the last conv layer which is layer4', default='layer4')
    model_group.add_argument('--target_class', type=int, help='Target class', default=None)
    model_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification', default='SaxiClassification')

    ##Gaussian Filter
    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, default=0, help='Mean (default: 0)')
    gaussian_group.add_argument('--std', type=float, default=0.005, help='Standard deviation (default: 0.005)')
    
    ##Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--fps', type=int, help='Frames per second', default=24)  
    output_group.add_argument('--out', type=str, help='Output directory', default='./grad_cam')

    return parser


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)

