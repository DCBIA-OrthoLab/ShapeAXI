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

from shapeaxi import saxi_nets, utils
from shapeaxi.colors import bcolors
from shapeaxi.post_process import RemoveIslands, DilateLabel, ErodeLabel, Threshold
from shapeaxi.saxi_dataset import SaxiDataset
from shapeaxi.saxi_transforms import UnitSurfTransform


def main(args):
    print(bcolors.INFO, "Start prediction of your data", bcolors.ENDC)

    out_channels = 34
    device = args.device

    model = saxi_nets.DentalModelSeg(custom_model=args.model, device=device)

    # Check if the input is a vtk file or a csv file
    if args.csv is not None:
        # Loading of the data
        path_csv = os.path.join(args.csv)
        df = pd.read_csv(path_csv)
        fname = os.path.basename(args.csv)
        predictions = {"surf": [], "pred": []}

    else:
        fname = os.path.basename(args.surf)
        df = pd.DataFrame([{"surf": args.surf, "out": args.out}])
    
    ds, dataloader, model, softmax = load_data(df, args, model, device)
    # Creation of the dictionary to store the predictions in a csv file with the same format as the input csv file (input vtk path/prediction vtk path)
        
    with torch.no_grad():
        # We go through the dataloader to get the prediction of the model
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            # Prediction of the model on the input data
            surf, V_labels_prediction = prediction(model, batch, out_channels, device, softmax, ds, idx, args)
            # Post processing on the data (closing operation, remove islands, etc)
            post_processing(surf, V_labels_prediction, out_channels)

            if args.fdi == 1:
                surf = ConvertFDI(surf, args)
            

            # Save the data in a vtk file
            if args.csv is not None:
                output_fn = save_data_vtk_from_csv(df, surf, fname, args, idx)
            else:
                output_fn = save_data_vtk(df, surf, fname, args, idx)
            output_fn = os.path.normpath(output_fn)

            if args.csv is not None:
                # Add this prediciton in the predictions dictionary of the csv file
                predictions["surf"].append(df["surf"][idx])
                predictions["pred"].append(output_fn) 

            if args.crown_segmentation:
                #Extraction of the vtk_file name to create a folder with the same name and store all the teeth files in this folder
                path_data = os.path.splitext(output_fn)[0]
                if not os.path.exists(path_data):
                    os.makedirs(path_data) 
                path_data = os.path.normpath(path_data)
                data_filename = os.path.basename(path_data)
                #Segmentation of each tooth in a specific vtk file
                segmentation_crown(surf, args, data_filename, path_data)
        
    if args.csv is not None:
        save_csv(predictions, args)

# Load the data
def load_data(df, args, model, device):
    ds = SaxiDataset(df, transform=UnitSurfTransform(), surf_column="surf")
    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    model.to(device)
    model.eval()
    softmax = torch.nn.Softmax(dim=2)
    return ds, dataloader, model, softmax

# Prediction using the model
def prediction( model, batch, out_channels, device, softmax, ds, idx, args):
    V, F, CN = batch
    V = V.to(device)
    F = F.to(device)
    CN = CN.to(device).to(torch.float32)
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
def save_data_vtk(df, surf, fname, args, idx): 
    os.makedirs(args.out, exist_ok=True)
    ext = os.path.splitext(fname)[1]

    if args.overwrite: 
        # If the overwrite argument is true, the original file is overwritten by the prediction file
        if ext == ".stl":
            os.remove(args.stl)
        utils.Write(surf, args.surf, print_out=False)
        output_fn = args.surf
        print(bcolors.SUCCESS,f"Saving results to {output_fn}", bcolors.ENDC)

    else:
        # If the overwrite argument is false, the prediction file is saved in the output directory with the same name as the original file + _pred.vtk
        output_fn = os.path.join(args.out, f"{os.path.splitext(fname)[0]}_{args.suffix}.vtk")
        utils.Write(surf , output_fn, print_out=False)
        print(bcolors.SUCCESS,f"Saving results to {output_fn}", bcolors.ENDC)

    return output_fn


def save_data_vtk_from_csv(df, surf, fname, args, idx):
    os.makedirs(args.out, exist_ok=True)
    # If the input is a csv file, the prediction vtk files are saved in the output directory with the same name as the original files + _pred.vtk
    filename = os.path.splitext(df["surf"][idx])[0]
    ext = os.path.splitext(df["surf"][idx])[1]

    # Same thing as above but with csv file
    if args.overwrite:
        if ext == ".stl":
            os.remove(df["surf"][idx])
        new_filename = filename + ".vtk"
        output_fn = new_filename
        utils.Write(surf, output_fn, print_out=False)
        print(bcolors.SUCCESS,f"Saving results to {output_fn}", bcolors.ENDC)

    else:
        # Creation of the output path for the vtk file without the path from the vtk_folder
        new_filename = filename + f"_{args.suffix}.vtk"
        true_vtk_path = new_filename.replace(args.vtk_folder, "")

        csv_filename = os.path.splitext(os.path.basename(args.csv))[0]
        predictions_csv_name = f"{csv_filename}_{args.suffix}"
        # Creation of the directory to store the prediction vtk files
        output_fn = os.path.normpath(args.out + "/" + predictions_csv_name + "/" + true_vtk_path)
        directory = os.path.dirname(output_fn)
        os.makedirs(directory, exist_ok=True)
        print(bcolors.SUCCESS,f"Saving results to {output_fn}", bcolors.ENDC)
        utils.Write(surf , output_fn, print_out=False)
    
    return output_fn


# Isolate each label in a specific vtk file
def segmentation_crown(surf, args, fname, directory):
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


def save_csv(predictions, args):
    # Create a dataframe with the predictions 
    predictions_df = pd.DataFrame(predictions)
    csv_filename = os.path.splitext(os.path.basename(args.csv))[0]
    
    if args.overwrite:
        predictions_csv_path = args.csv
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(bcolors.SUCCESS,f"Saving results to {predictions_csv_path}", bcolors.ENDC)

    else:
        # Save the dataframe in a csv file
        predictions_csv_path = os.path.join(args.out, f"{csv_filename}_{args.suffix}.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(bcolors.SUCCESS,f"Saving results to {predictions_csv_path}", bcolors.ENDC)


# Convert the universal numbering system to the FDI world dental Federation notation
def ConvertFDI(surf, args):
    print('Converting to FDI...')
    LUT = np.array([0,18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                    38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48,0])
    # extract UniversalID array
    labels = vtk_to_numpy(surf.GetPointData().GetScalars(args.array_name))
    # convert to their numbering system
    labels = LUT[labels]
    vtk_id = numpy_to_vtk(labels)
    vtk_id.SetName(args.array_name)
    surf.GetPointData().AddArray(vtk_id)
    return surf



def get_argparse():
    parser = argparse.ArgumentParser(description='Evaluate classification result')
    
    # Create a mutually exclusive group for --vtk and --csv
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--surf', type=str, help='Path to your vtk file',default=None)
    input_group.add_argument('--csv', type=str, help='Path to your csv file',default=None)

    parser.add_argument('--model', type=str, help='Path to the model', default=None)
    parser.add_argument('--suffix', type=str, help='Suffix of the prediction', default='pred')
    parser.add_argument('--out', type=str, help='Output directory', default='./predictions')
    parser.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    parser.add_argument('--crown_segmentation', type=int, help='Isolation of each different tooth in a specific vtk file', default=None)
    parser.add_argument('--array_name', type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")
    parser.add_argument('--fdi', type=int, help = 'Numbering system. 0: Universal numbering; 1: FDI world dental Federation notation', default=0)
    parser.add_argument('--overwrite', type=int, help='Overwrite the input vtk file', default=None)
    parser.add_argument('--device', type=str, help='Device to use for inference', default='cuda:0',choices=['cpu', 'cuda:0'])
    parser.add_argument('--vtk_folder', type=str, help='Path to tronquate your input path', default='')

    return parser


def cml():
    parser = get_argparse()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cml()