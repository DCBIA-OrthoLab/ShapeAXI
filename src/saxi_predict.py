import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader

import src.saxi_nets as saxi_nets
from src.saxi_dataset import SaxiDataset, RandomRemoveTeethTransform, UnitSurfTransform
from src.saxi_transforms import EvalTransform

import pickle
from tqdm import tqdm

import src.utils as utils
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import nrrd



def main(args):

    mount_point = args.mount_point

    fname = os.path.basename(args.csv)    
    ext = os.path.splitext(fname)[1]

    if ext == ".csv":
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)
    
    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
        
        SAXINETS = getattr(saxi_nets, args.nn)
        model = SAXINETS.load_from_checkpoint(args.model)

        model.ico_sphere(args.radius, args.subdivision_level)

        model.to(torch.device('cuda:0'))
        model.eval()

        scale_factor = None

        if model.hparams.scale_factor:
            scale_factor = model.hparams.scale_factor

        test_ds = SaxiDataset(df, transform=EvalTransform(scale_factor), **vars(args))
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    elif args.nn == "SaxiSegmentation":

        class_weights = None
        out_channels = 34

        # model = saxi_nets.MonaiUNet(args, out_channels = out_channels, class_weights=class_weights, image_size=320, subdivision_level=2)
        # model.model.module.load_state_dict(torch.load(args.model)
        MONAI = getattr(saxi_nets, args.nn)
        model = MONAI.load_from_checkpoint(args.model)


        ds = SaxiDataset(df, mount_point = args.mount_point, transform=UnitSurfTransform(), surf_column="surf")

        dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
        

        device = torch.device('cuda')
        model.to(device)
        model.eval()

        softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        if args.nn == "SaxiClassification":

            probs = []
            predictions = []
            softmax = nn.Softmax(dim=1)

            for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
                
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


        elif args.nn == "SaxiRegression":        
            
            predictions = []
            softmax = nn.Softmax(dim=1)

            for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
                
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
    

        elif args.nn == "SaxiSegmentation":

            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

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
                output_fn = output_fn.replace("./", "")
                
                out_dir = os.path.dirname(output_fn)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                utils.Write(surf , output_fn, print_out=False)

            out_dir = os.path.join(args.out, os.path.basename(args.model))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            data = {
                "surf": df["surf"],
                "pred": [os.path.join(args.out, df["surf"][idx]) for idx in range(len(df))]
            }
            
            result_df = pd.DataFrame(data)

            output_fn = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
            print(f"Saving results to {output_fn}")
            result_df.to_csv(output_fn, index=False)              


def get_argparse():
    parser = argparse.ArgumentParser(description='Saxi prediction')    

    model_group = parser.add_argument_group('Trained')
    model_group.add_argument('--model', help='Model for prediction', type=str, default = "/work/leclercq/data/07-21-22_val-loss0.169.pth")
    model_group.add_argument('--nn', help='Neural network name', type=str, default='SaxiClassification')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv', help='CSV with column surf', type=str, required=True)   
    input_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    input_group.add_argument('--class_column', help='Class column name', type=str, default="class")
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")

    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    hyper_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=2)


    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    return parser


if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    main(args)
