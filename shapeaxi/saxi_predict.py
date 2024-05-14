import argparse
import math
import os
import pandas as pd
import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import nrrd
import monai

from shapeaxi import saxi_nets, utils
from shapeaxi.saxi_dataset import SaxiDataset, SaxiIcoDataset, SaxiIcoDataset_fs 
from shapeaxi.saxi_transforms import EvalTransform, UnitSurfTransform, TrainTransform, RandomRemoveTeethTransform, RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform
from shapeaxi.post_process import RemoveIslands, DilateLabel, ErodeLabel, Threshold
from shapeaxi.dental_model_seg import segmentation_crown, post_processing
from shapeaxi.colors import bcolors
torch.backends.cudnn.benchmark = False

# This file proposes a prediction with the test data. It calls SaxiDataset which is a custom class from PyTorch that inherits from torch.utils.data.Datset.
# It calls also EvalTransform 


def SaxiSegmentation_predict(args, mount_point, df, fname, ext):
    # The dataset and corresponding data loader are initialized for evaluation purposes.
    class_weights = None
    out_channels = 34
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    ds = SaxiDataset(df, mount_point = args.mount_point, transform=UnitSurfTransform(), surf_column="surf")
    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    softmax = torch.nn.Softmax(dim=2)

    predictions = {"surf": [], "pred": []}

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
            V, F, CN = batch
            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True).to(torch.float32)
            x, X, PF = model((V, F, CN))
            x = softmax(x*(PF>=0))
            P_faces = torch.zeros(out_channels, F.shape[1]).to(device)
            V_labels_prediction = torch.zeros(V.shape[1]).to(device).to(torch.int64)
            PF = PF.squeeze()
            x = x.squeeze()

            for pf, pred in zip(PF, x):
                P_faces[:, pf] += pred

            P_faces = torch.argmax(P_faces, dim=0)
            faces_pid0 = F[0,:,0]
            V_labels_prediction[faces_pid0] = P_faces
            surf = ds.getSurf(idx)
            V_labels_prediction = numpy_to_vtk(V_labels_prediction.cpu().numpy())
            V_labels_prediction.SetName(args.array_name)
            surf.GetPointData().AddArray(V_labels_prediction)
            output_fn = os.path.join(args.out, df["surf"][idx])
            out_dir = os.path.dirname(output_fn.replace("./", ""))

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            post_processing(surf, V_labels_prediction, out_channels)

            utils.Write(surf , output_fn, print_out=False)

            #Save to the new CSV file
            predictions["surf"].append(df["surf"][idx])
            predictions["pred"].append(output_fn)

            if args.crown_segmentation:
                #Extraction of the vtk_file name to create a folder with the same name and store all the teeth files in this folder
                vtk_path = os.path.splitext(output_fn)[0]
                if not os.path.exists(vtk_path):
                    os.makedirs(vtk_path) 
                vtk_directory = os.path.normpath(vtk_path)
                vtk_filename = os.path.basename(vtk_directory)
                #Segmentation of each tooth in a specific vtk file
                segmentation_crown(surf, args, vtk_filename, vtk_directory)


    out_dir = os.path.join(args.out, os.path.basename(args.model))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Create a new DataFrame from the predictions dictionary
    predictions_df = pd.DataFrame(predictions)
    csv_filename = os.path.splitext(args.csv)[0]
    # Save the predictions DataFrame to a new CSV file
    predictions_csv_path = os.path.join(out_dir, f"{csv_filename}_prediction.csv")
    predictions_df.to_csv(predictions_csv_path, index=False) 
        
    print(bcolors.SUCCESS, f"Saving results to {predictions_csv_path}", bcolors.ENDC)


def SaxiClassification_predict(args, mount_point, df, fname, ext, test_loader, model):
    with torch.no_grad():
        # The prediction is performed on the test data
        probs = []
        predictions = []
        softmax = nn.Softmax(dim=1)

        for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True)
            X, PF = model.render(V, F, CN)
            x, x_s = model(X)
            x = softmax(x).detach()
            probs.append(x)
            predictions.append(torch.argmax(x, dim=1, keepdim=True))

        probs = torch.cat(probs).detach().cpu().numpy()
        predictions = torch.cat(predictions).cpu().numpy().squeeze()

        out_dir = os.path.join(args.out, os.path.basename(args.model))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_probs = os.path.join(out_dir, fname.replace(ext, "_probs.pickle"))
        pickle.dump(probs, open(out_probs, 'wb'))

        df['pred'] = predictions
        if ext == ".csv":
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
            df.to_csv(out_name, index=False)
        else:
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.parquet"))
            df.to_parquet(out_name, index=False)
        print(bcolors.SUCCESS, f"Saving results to {out_name}", bcolors.ENDC)


def SaxiRegression_predict(args, mount_point, df, fname, ext, test_loader, model):
    with torch.no_grad():
        predictions = []
        softmax = nn.Softmax(dim=1)

        for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True)
            X, PF = model.render(V, F, CN)
            x, x_s = model(X)
            predictions.append(x.detach())
        
        predictions = torch.cat(predictions).cpu().numpy().squeeze()
        out_dir = os.path.join(args.out, os.path.basename(args.model))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        df['pred'] = predictions
        if ext == ".csv":
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
            df.to_csv(out_name, index=False)
        else:
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.parquet"))
            df.to_parquet(out_name, index=False)
        print(bcolors.SUCCESS, f"Saving results to {out_name}", bcolors.ENDC)


def SaxiIcoClassification_predict(args, mount_point, df, fname, ext):
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    model.to(torch.device(args.device))
    model.eval()
    
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR
    list_path_ico = [args.path_ico_left,args.path_ico_right]

    test_ds = SaxiIcoDataset(df,list_demographic,list_path_ico,name_class = args.class_column,transform=UnitSurfTransform())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():
        # The prediction is performed on the test data
        probs = []
        predictions = []
        softmax = nn.Softmax(dim=1)

        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
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

            X = (VL, FL, VFL, FFL, VR, FR, VFR, FFR, D)
            x = model(X)

            x = softmax(x).detach()
            probs.append(x)
            predictions.append(torch.argmax(x, dim=1, keepdim=True))

        probs = torch.cat(probs).detach().cpu().numpy()
        predictions = torch.cat(predictions).cpu().numpy().squeeze()

        out_dir = os.path.join(args.out, os.path.basename(args.model))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_probs = os.path.join(out_dir, fname.replace(ext, "_probs.pickle"))
        pickle.dump(probs, open(out_probs, 'wb'))

        df['pred'] = predictions
        if ext == ".csv":
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
            df.to_csv(out_name, index=False)
        else:
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.parquet"))
            df.to_parquet(out_name, index=False)
        print(bcolors.SUCCESS, f"Saving results to {out_name}", bcolors.ENDC)


def SaxiIcoClassification_fs_predict(args, mount_point, df, fname, ext):
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)
    model.to(torch.device(args.device))
    model.eval()
    test_ds = SaxiIcoDataset_fs(df,transform=UnitSurfTransform(),name_class=args.class_column,freesurfer_path=args.fs_path)
        
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():
        # The prediction is performed on the test data
        probs = []
        predictions = []
        softmax = nn.Softmax(dim=1)

        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
            VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = batch 
            VL = VL.cuda(non_blocking=True,device=args.device)
            FL = FL.cuda(non_blocking=True,device=args.device)
            VFL = VFL.cuda(non_blocking=True,device=args.device)
            FFL = FFL.cuda(non_blocking=True,device=args.device)
            VR = VR.cuda(non_blocking=True,device=args.device)
            FR = FR.cuda(non_blocking=True,device=args.device)
            VFR = VFR.cuda(non_blocking=True,device=args.device)
            FFR = FFR.cuda(non_blocking=True,device=args.device)
            FFL = FFL.squeeze(0)
            FFR = FFR.squeeze(0)

            X = (VL, FL, VFL, FFL, VR, FR, VFR, FFR)
            x = model(X)

            x = softmax(x).detach()
            probs.append(x)
            predictions.append(torch.argmax(x, dim=1, keepdim=True))

        probs = torch.cat(probs).detach().cpu().numpy()
        predictions = torch.cat(predictions).cpu().numpy().squeeze()

        out_dir = os.path.join(args.out, os.path.basename(args.model))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_probs = os.path.join(out_dir, fname.replace(ext, "_probs.pickle"))
        pickle.dump(probs, open(out_probs, 'wb'))

        df['pred'] = predictions
        if ext == ".csv":
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
            df.to_csv(out_name, index=False)
        else:
            out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.parquet"))
            df.to_parquet(out_name, index=False)
        print(bcolors.SUCCESS, f"Saving results to {out_name}", bcolors.ENDC)


def main(args):
    # Read of the test data from a CSV or Parquet file
    mount_point = args.mount_point
    fname = os.path.basename(args.csv)    
    ext = os.path.splitext(fname)[1]
    path_to_csv = os.path.join(mount_point, args.csv)
    
    if ext == ".csv":
        df = pd.read_csv(path_to_csv)
    else:
        df = pd.read_parquet(path_to_csv)
    
    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
        # The dataset and corresponding data loader are initialized for evaluation purposes.
        SAXINETS = getattr(saxi_nets, args.nn)
        model = SAXINETS.load_from_checkpoint(args.model)
        model.ico_sphere(args.radius, args.subdivision_level)
        model.to(torch.device(args.device))
        model.eval()
        scale_factor = None
        if model.hparams.scale_factor:
            scale_factor = model.hparams.scale_factor
        test_ds = SaxiDataset(df, transform=EvalTransform(scale_factor), **vars(args))
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

        if args.nn == "SaxiRegression":
            SaxiRegression_predict(args, mount_point, df, fname, ext, test_loader, model)
        else:
            SaxiClassification_predict(args, mount_point, df, fname, ext, test_loader, model)
    
    elif args.nn == "SaxiRingClassification":
        SAXINETS = getattr(saxi_nets, args.nn)
        model = SAXINETS.load_from_checkpoint(args.model)
        model.to(torch.device(args.device))
        model.eval()
        scale_factor = None
        if model.hparams.scale_factor:
            scale_factor = model.hparams.scale_factor
        test_ds = SaxiDataset(df, transform=EvalTransform(scale_factor), **vars(args))
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)
        
        SaxiClassification_predict(args, mount_point, df, fname, ext, test_loader, model)

    elif args.nn == "SaxiSegmentation":
        SaxiSegmentation_predict(args, mount_point, df, fname, ext) 

    elif args.nn == "SaxiIcoClassification":
        SaxiIcoClassification_predict(args, mount_point, df, fname, ext)
    
    elif args.nn == "SaxiIcoClassification_fs" or args.nn == "SaxiRing":
        SaxiIcoClassification_fs_predict(args, mount_point, df, fname, ext)

    else:
        raise NotImplementedError(f"Neural network {args.nn} is not implemented")             


def get_argparse():
    # This function defines the arguments for the prediction
    parser = argparse.ArgumentParser(description='Saxi prediction')    

    ##Trained
    model_group = parser.add_argument_group('Trained')
    model_group.add_argument('--model', type=str, help='Model for prediction', required=True)
    model_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification, SaxiRing, SaxiRingClassification', required=True, choices=["SaxiClassification", "SaxiRegression", "SaxiSegmentation", "SaxiIcoClassification", "SaxiIcoClassification_fs", 'SaxiRing', 'SaxiRingClassification'])

    ##Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv', type=str, help='CSV with column surf', required=True)   
    input_group.add_argument('--surf_column', type=str, help='Surface column name', default='surf')
    input_group.add_argument('--class_column', type=str, help='Class column name', default='class')
    input_group.add_argument('--mount_point', type=str, help='Dataset mount directory', default='./')
    input_group.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    input_group.add_argument('--array_name', type=str, help = 'Predicted ID array name for output vtk', default='PredictedID')
    input_group.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    input_group.add_argument('--crown_segmentation', type=bool, help='Isolation of each different tooth in a specific vtk file', default=False)
    input_group.add_argument('--fdi', type=int, help='numbering system. 0: universal numbering; 1: FDI world dental Federation notation', default=0)
    input_group.add_argument('--path_ico_left', type=str, default='./3DObject/sphere_f327680_v163842.vtk', help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)')
    input_group.add_argument('--path_ico_right', type=str, default='./3DObject/sphere_f327680_v163842.vtk', help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)')
    input_group.add_argument('--fs_path', type=str, help='Path to freesurfer folder', default=None)
    input_group.add_argument('--device', type=str, help='Device for prediction', default='cuda:0')

    ##Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.35)    
    hyper_group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)

    ##Gaussian Filter
    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, help='Mean (default: 0)',default=0,)
    gaussian_group.add_argument('--std', type=float, help='Standard deviation (default: 0.005)', default=0.005)

    ##Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', type=str, help='Output directory', default="./")

    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    main(args)
