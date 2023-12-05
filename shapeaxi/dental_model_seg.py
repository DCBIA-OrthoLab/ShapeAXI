import argparse
import os
from argparse import Namespace
import torch
import pandas as pd
from torch.utils.data import DataLoader
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
from tqdm import tqdm
import vtk
import numpy as np

from . import saxi_eval, saxi_predict, saxi_nets, utils
from .post_process import RemoveIslands, DilateLabel, ErodeLabel, Threshold
from .saxi_folds import bcolors, get_argparse_dict
from .saxi_dataset import SaxiDataset
from .saxi_transforms import UnitSurfTransform


def main(args):
    print(bcolors.INFO, "Start evaluation of the model", bcolors.ENDC)
    
    if args.model is None:
        out_channels = 34
        model = saxi_nets.DentalModelSeg()
    else:
        model = args.model

    # Check if the input is a vtk file or a csv file
    if args.csv:
        # Loading of the data
        path_csv = os.path.join(args.mount_point,args.csv)
        df = pd.read_csv(path_csv)
        fname = os.path.basename(args.csv)
        ds, dataloader, device, model, softmax = load_data(df, args, model)
        
        # Creation of the dictionary to store the predictions in a csv file with the same format as the input csv file (input vtk path/prediction vtk path)
        predictions = {"surf": [], "pred": []}
        
        with torch.no_grad():
            # We go through the dataloader to get the prediction of the model
            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                
                # Prediction of the model on the input data
                surf, V_labels_prediction = prediction(model, batch, out_channels, device, softmax, ds, idx, args)
                # Post processing on the data (closing operation, remove islands, etc)
                post_processing(surf, V_labels_prediction, out_channels)

                if args.fdi == 1:
                    surf = ConvertFDI(surf)
                
                # Save the data in a vtk file
                output_fn = save_data(df, surf, fname, args, idx)
                output_fn = os.path.normpath(output_fn)

                # Add this prediciton in the predictions dictionary of the csv file
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
            
        
        # Create a dataframe with the predictions 
        predictions_df = pd.DataFrame(predictions)
        # Save the dataframe in a csv file
        csv_filename = os.path.splitext(os.path.basename(args.csv))[0]
        predictions_csv_path = os.path.join(args.out, f"{csv_filename}_pred.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)

        print(bcolors.SUCCESS,f"Saving results to {predictions_csv_path}", bcolors.ENDC)


    else:
        # If it is a vtk file, we create a dataframe with the path of the vtk file and the output directory
        fname = os.path.basename(args.vtk)
        df = pd.DataFrame([{"surf": args.vtk, "out": args.out}])
        # Loading of the data
        ds, dataloader, device, model, softmax = load_data(df, args, model)

        with torch.no_grad():
            # We go through the dataloader to get the prediction of the model
            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                
                # Prediction of the model on the input data
                surf, V_labels_prediction = prediction(model, batch, out_channels, device, softmax, ds, idx, args)    
                # Post processing on the data (closing operation, remove islands, etc)
                post_processing(surf, V_labels_prediction, out_channels)  

                if args.fdi == 1:
                    surf = ConvertFDI(surf)
                
                # Save the data in a vtk file
                output_fn = save_data(df, surf, fname, args, idx)
                output_fn = os.path.normpath(output_fn)

                if args.crown_segmentation:
                    #Extraction of the vtk_file name to create a folder with the same name and store all the teeth files in this folder
                    vtk_path = os.path.splitext(output_fn)[0]
                    if not os.path.exists(vtk_path):
                        os.makedirs(vtk_path) 
                    vtk_directory = os.path.normpath(vtk_path)
                    vtk_filename = os.path.basename(vtk_directory)
                    #Segmentation of each tooth in a specific vtk file
                    segmentation_crown(surf, args, vtk_filename, vtk_directory)


# Load the data
def load_data(df, args, model):
    ds = SaxiDataset(df,args.mount_point, transform=UnitSurfTransform(), surf_column="surf")
    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    softmax = torch.nn.Softmax(dim=2)
    return ds, dataloader, device, model, softmax


# Prediction using the model
def prediction( model, batch, out_channels, device, softmax, ds, idx, args):
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

    return surf, V_labels_prediction


# Post processing on the data
def post_processing(surf, V_labels_prediction, out_channels):
    # Start with gum
    RemoveIslands(surf, V_labels_prediction, 33, 500,ignore_neg1 = True) 

    for label in tqdm(range(out_channels),desc = 'Removing islands'):
        RemoveIslands(surf, V_labels_prediction, label, 200,ignore_neg1 = True) 

    # CLOSING OPERATION
    # One tooth at a time
    for label in tqdm(range(out_channels),desc = 'Closing operation'):
        DilateLabel(surf, V_labels_prediction, label, iterations=2, dilateOverTarget=False, target=None) 
        ErodeLabel(surf, V_labels_prediction, label, iterations=2, target=None)       


# Save the data in a vtk file or in a csv file
def save_data(df, surf, fname, args, idx): 
    if args.vtk:
        if args.overwrite: 
            # If the overwrite argument is true, the original file is overwritten by the prediction file
            os.remove(os.path.join(args.mount_point,args.vtk))
            output_fn = os.path.join(args.mount_point, args.vtk)
            utils.Write(surf, output_fn, print_out=False)
            print(bcolors.SUCCESS,f"Saving results to {output_fn}", bcolors.ENDC)
        else:
            # If the overwrite argument is false, the prediction file is saved in the output directory with the same name as the original file + _pred.vtk
            if not os.path.exists(args.out):
                os.makedirs(args.out) 
            output_fn = os.path.join(args.out, f"{os.path.splitext(fname)[0]}_pred.vtk")
            utils.Write(surf , output_fn, print_out=False)
            print(bcolors.SUCCESS,f"Saving results to {output_fn}", bcolors.ENDC)
    else:
        # If the input is a csv file, the prediction vtk files are saved in the output directory with the same name as the original files + _pred.vtk
        if not os.path.exists(args.out):
            os.makedirs(args.out) 
        filename = os.path.splitext(df["surf"][idx])[0]
        new_filename = filename + "_pred.vtk"
        output_fn = os.path.join(args.out, new_filename)
        utils.Write(surf , output_fn, print_out=False)

    return output_fn


# Isolate each label in a specific vtk file
def segmentation_crown(surf, args, fname, directory):
    if not os.path.exists(args.out):
        os.makedirs(args.out) 

    surf_point_data = surf.GetPointData().GetScalars(args.array_name) 
    labels = np.unique(surf_point_data)
    
    for label in tqdm(labels, desc = 'Isolating labels'):
        thresh_label = Threshold(surf, args.array_name ,label-0.5,label+0.5)
        if (args.fdi==0 and label != 33) or(args.fdi==1 and label !=0):
            output_teeth = os.path.join(directory, f'{fname}_id_{label}.vtk')
            utils.Write(thresh_label,output_teeth,print_out=False) 
        else:
        # gum
            output_teeth = os.path.join(directory, f'{fname}_gum.vtk')
            utils.Write(thresh_label,output_teeth,print_out=False) 
    # all teeth 
    no_gum = Threshold(surf,args.array_name ,33-0.5,33+0.5,invert=True)
    output_teeth = os.path.join(directory, f'{fname}_all_teeth.vtk')
    utils.Write(no_gum,output_teeth,print_out=False)
    print(bcolors.SUCCESS,"Each teeth are saved", bcolors.ENDC)


# Convert the universal numbering system to the FDI world dental Federation notation
def ConvertFDI(surf):
  print('Converting to FDI...')
  LUT = np.array([0,18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                  38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48,0])
  # extract UniversalID array
  labels = vtk_to_numpy(surf.GetPointData().GetScalars(args.scal))
  # convert to their numbering system
  labels = LUT[labels]
  vtk_id = numpy_to_vtk(labels)
  vtk_id.SetName(args.scal)
  surf.GetPointData().AddArray(vtk_id)
  return surf



def get_argparse():
    parser = argparse.ArgumentParser(description='Evaluate classification result')
    
    # Create a mutually exclusive group for --vtk and --csv
    vtk_csv_group = parser.add_mutually_exclusive_group(required=True)
    vtk_csv_group.add_argument('--vtk', type=str, help='Path to your vtk file', default=None)
    vtk_csv_group.add_argument('--csv', type=str, help='Path to your csv file', default=None)

    parser.add_argument('--model', type=str, help='Path to the model', default=None)
    parser.add_argument('--out', type=str, help='Output directory', default="./predictions")
    parser.add_argument('--mount_point', type=str, help='Mount point for the dataset', default="./")
    parser.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    parser.add_argument('--crown_segmentation', help='Isolation of each different tooth in a specific vtk file', default=False)
    parser.add_argument('--array_name', type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--fdi', type=int, help = 'numbering system. 0: universal numbering; 1: FDI world dental Federation notation', default=0)
    parser.add_argument('--overwrite', help='Overwrite the input vtk file', default=False)

    return parser

def cml():
    parser = get_argparse()
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cml()