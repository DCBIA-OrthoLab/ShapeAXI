import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader

import src.saxi_nets as saxi_nets
from saxi_dataset import SaxiDataset
from saxi_transforms import EvalTransform

import pickle
from tqdm import tqdm

def main(args):
    
    fname = os.path.basename(args.csv_test)    
    ext = os.path.splitext(fname)[1]

    if ext == ".csv":
        df_test = pd.read_csv(args.csv_test)
    else:
        df_test = pd.read_parquet(args.csv_test)

    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS.load_from_checkpoint(args.model)

    model.ico_sphere(args.radius, args.subdivision_level)

    model.to(torch.device('cuda:0'))
    model.eval()

    scale_factor = None

    if model.hparams.scale_factor:
        scale_factor = model.hparams.scale_factor

    test_ds = SaxiDataset(df_test, transform=EvalTransform(scale_factor), **vars(args))
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

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

            df_test['pred'] = predictions
            if ext == ".csv":
                out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
                df_test.to_csv(out_name, index=False)
            else:
                out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.parquet"))
                df_test.to_parquet(out_name, index=False)
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

            df_test['pred'] = predictions
            if ext == ".csv":
                out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.csv"))
                df_test.to_csv(out_name, index=False)
            else:
                out_name = os.path.join(out_dir, fname.replace(ext, "_prediction.parquet"))
                df_test.to_parquet(out_name, index=False)


def get_argparse():
    parser = argparse.ArgumentParser(description='Saxi prediction')    

    model_group = parser.add_argument_group('Trained')
    model_group.add_argument('--model', help='Model for prediction', type=str, required=True)
    model_group.add_argument('--nn', help='Neural network name', type=str, default='SaxiClassification')

    input_group = parser.add_argument_group('Input')

    input_group.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)   
    input_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    input_group.add_argument('--class_column', help='Class column name', type=str, default="class")
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)

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

