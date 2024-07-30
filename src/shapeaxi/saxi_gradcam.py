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
import subprocess

from shapeaxi import saxi_nets, post_process as psp, utils
from shapeaxi.saxi_dataset import SaxiDataset, SaxiIcoDataset, SaxiFreesurferDataset
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform, UnitSurfTransform, RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform

# Loops over the folds to generate a visualization to explain what is happening in the network after the evaluation part of the training is done.
# Especially identify the parts of the picture which is the most important for the network to make a decision.


def gradcam_process(args, grayscale_cam, F, PF, V, device):
    '''
    Function to process the GradCAM values and add them to the input surface mesh (surf) as a point data array

    Args :
        args : arguments
        grayscale_cam : GradCAM values
        F : Faces
        PF : Point faces
        V : Vertices
        device : device (cuda or cpu)
    '''
    GCAM = torch.tensor(grayscale_cam).to(device)

    P_faces = torch.zeros(1, F.shape[1]).to(device)
    V_gcam = -1*torch.ones(V.shape[1], dtype=torch.float32).to(device)

    for pf, gc in zip(PF.squeeze(), GCAM):
        P_faces[:, pf] = torch.maximum(P_faces[:, pf], gc)

    faces_pid0 = F[0,:,0].to(torch.int64)
    V_gcam[faces_pid0] = P_faces
    V_gcam = numpy_to_vtk(V_gcam.cpu().numpy())

    if not args.target_class is None:
        array_name = "grad_cam_target_class_{target_class}".format(target_class=args.target_class)
    else:
        array_name = "grad_cam_max"

    V_gcam.SetName(array_name)

    return V_gcam


def gradcam_save(args, out_dir, V_gcam, surf_path, surf):
    '''
    Function to save the GradCAM on the surface

    Args : 
        gradcam_path : path to save the GradCAM
        V_gcam : GradCAM values
        surf_path : path to the surface
        surf : surface read by utils.ReadSurf
        hemisphere : hemisphere (lh or rh)
    '''

    gradcam_path = os.path.join(out_dir, os.path.dirname(surf_path))

    if not os.path.exists(gradcam_path):
        os.makedirs(gradcam_path)
    
    if args.fs_path is not None:
        surf_path = os.path.join(args.fs_path, surf_path)

    out_surf_path = os.path.join(gradcam_path, os.path.basename(surf_path))

    subprocess.call(["cp", surf_path, out_surf_path])

    surf = utils.ReadSurf(out_surf_path)

    surf.GetPointData().AddArray(V_gcam)

    # Median filtering is applied to smooth the CAM on the surface
    psp.MedianFilter(surf, V_gcam)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_surf_path)
    writer.SetInputData(surf)
    writer.Write()
    

def SaxiClassification_Regression_gradcam(args, df_test, model, device):
    '''
    Function called to generate the GradCAM for the classification and regression models

    Args :
        args : arguments
        df_test : test dataframe
        model : model loaded from checkpoint
        device : device (cuda or cpu)
    '''
    model.ico_sphere(radius=args.radius, subdivision_level=args.subdivision_level)
    model = model.to(device)
    model.eval()
    test_ds = SaxiDataset(df_test, transform=EvalTransform(), **vars(args))
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    target_layer = getattr(model.F.module, args.target_layer)
    target_layers = None 

    if isinstance(target_layer, nn.Sequential):
        target_layer = target_layer[-1]
        target_layers = [target_layer]

    # Construct the CAM object
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = None
    if not args.target_class is None:
        targets = [ClassifierOutputTarget(args.target_class)]

    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    out_dir = os.path.join(os.path.dirname(args.csv_test), "grad_cam", str(args.target_class))

    for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
        V = V.cuda(non_blocking=True)
        F = F.cuda(non_blocking=True)
        CN = CN.cuda(non_blocking=True)

        X, PF = model.render(V, F, CN)
        gcam_np = cam(input_tensor=X, targets=targets)

        Vcam = gradcam_process(args, gcam_np, F, PF, V, device)

        surf = test_ds.getSurf(idx)
        surf.GetPointData().AddArray(V_gcam)

        # Median filtering is applied to smooth the CAM on the surface
        psp.MedianFilter(surf, V_gcam)

        surf_path = os.path.basename(df_test.loc[idx][args.surf_column])

        gradcam_save(args, out_dir, V_gcam, surf_path, surf)



#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                    IcoConv Freesurfer                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class PoolingAttentionLayer(nn.Module):
    # class to return only the context vector of each pooling layer
    def __init__(self, attention_layer):
        super().__init__()
        self.attention_layer = attention_layer

    def forward(self, x):
        x, score = self.attention_layer(x)
        return x


class SelfAttentionLayer(nn.Module):
    # class to return only the context vector of the Attention layer
    def __init__(self, attention_layer, values_layer):
        super().__init__()
        self.attention_layer = attention_layer
        self.values_layer = values_layer

    def forward(self, x):
        values = self.values_layer(x)  # Pass input through the values_layer
        x, score = self.attention_layer(x, values)
        return x



def SaxiRing_gradcam(args, df_test, model, device):
    '''
    Function called to generate the GradCAM for the Rings models

    Args :
        args : arguments
        df_test : test dataframe
        model : model loaded from checkpoint
        device : device (cuda or cpu)
    '''
    model.to(device)
    model.eval()
    test_ds = SaxiFreesurferDataset(df_test, transform=UnitSurfTransform(), name_class=args.class_column, freesurfer_path=args.fs_path)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    targets = None
    if args.target_class is not None:
        targets = [ClassifierOutputTarget(args.target_class)]

    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    out_dir = os.path.join(os.path.dirname(args.csv_test), "grad_cam", str(args.target_class))

    model_camL = nn.Sequential(
        model.TimeDistributedL,
        PoolingAttentionLayer(model.down1),
        PoolingAttentionLayer(model.down2),
        SelfAttentionLayer(model.Att, model.W),
    )

    model_camR = nn.Sequential(
        model.TimeDistributedR,
        PoolingAttentionLayer(model.down1),
        PoolingAttentionLayer(model.down2),
        SelfAttentionLayer(model.Att, model.W),
    )
    
    target_layers_l = [model_camL[0].module.layer4[-1]]
    camL = GradCAM(model=model_camL, target_layers=target_layers_l)

    target_layers_r = [model_camR[0].module.layer4[-1]]
    camR = GradCAM(model=model_camR, target_layers=target_layers_r)


    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = batch
        VL = VL.cuda(non_blocking=True)
        FL = FL.cuda(non_blocking=True)
        VFL = VFL.cuda(non_blocking=True)
        FFL = FFL.cuda(non_blocking=True)
        VR = VR.cuda(non_blocking=True)
        FR = FR.cuda(non_blocking=True)
        VFR = VFR.cuda(non_blocking=True)
        FFR = FFR.cuda(non_blocking=True)
        FFL = FFL.squeeze(0)
        FFR = FFR.squeeze(0)

        XL, PFL = model.render(VL, FL, VFL, FFL)
        XR, PFR = model.render(VR, FR, VFR, FFR)

        grayscale_camL = camL(input_tensor=XL, targets=targets)
        grayscale_camR = camR(input_tensor=XR, targets=targets)
        
        VcamL = gradcam_process(args, grayscale_camL, FL, PFL, VL, device)
        VcamR = gradcam_process(args, grayscale_camR, FR, PFR, VR, device)

        surfL, surfR, surfL_path, surfR_path = test_ds.getSurf(idx)
        
        gradcam_save(args, out_dir, VcamL, surfL_path, surfL)
        gradcam_save(args, out_dir, VcamR, surfR_path, surfR)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                         IcoConv                                                                                   #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


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


def SaxiIcoClassification_gradcam(args, df_test, model, device):     
    
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR
    list_path_ico = [args.path_ico_left,args.path_ico_right]

    test_ds = SaxiIcoDataset(df_test,list_demographic,list_path_ico,transform=UnitSurfTransform())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

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

        if args.hemisphere == 'left':
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


#####################################################################################################################################################################################

def main(args):
    fname = os.path.basename(args.csv_test)    
    ext = os.path.splitext(fname)[1]

    if ext == ".csv":
        df_test = pd.read_csv(args.csv_test)
    else:
        df_test = pd.read_parquet(args.csv_test)

    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    print(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gradcam_functions = {
        "SaxiClassification": SaxiClassification_Regression_gradcam,
        "SaxiRegression": SaxiClassification_Regression_gradcam,
        "SaxiRingClassification": SaxiClassification_Regression_gradcam,
        "SaxiRing": SaxiRing_gradcam,
        "SaxiMHA": SaxiRing_gradcam,
        "SaxiIcoClassification": SaxiIcoClassification_gradcam
    }

    # Train the model
    if args.nn in gradcam_functions:
        gradcam_functions[args.nn](
            args,
            df_test,
            model,
            device,
        )
    else:
        raise ValueError(f"Unknown neural network name: {args.nn}")



def get_argparse():
    # The arguments are defined for the script 
    parser = argparse.ArgumentParser(description='Saxi GradCam')

    ##Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_test',  type=str, help='CSV with column surf', required=True)   
    input_group.add_argument('--surf_column',  type=str, help='Surface column name', default="surf")
    input_group.add_argument('--class_column',  type=str, help='Class column name', default="class")
    input_group.add_argument('--num_workers',  type=int, help='Number of workers for loading', default=4)
    input_group.add_argument('--mount_point',  type=str, help='Dataset mount directory', default="./")

    ##Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model', type=str, help='Model for prediction', required=True)    
    model_group.add_argument('--target_layer', type=str, help='Target layer for GradCam. For example in ResNet, the target layer is the last conv layer which is layer4', default='layer4')
    model_group.add_argument('--target_class', type=int, help='Target class', default=1)
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
    initial_args, unknownargs = parser.parse_known_args()
    model_args = getattr(saxi_nets, initial_args.nn)
    model_args.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)

