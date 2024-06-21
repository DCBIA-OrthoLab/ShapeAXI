import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shapeaxi.saxi_dataset import SaxiDataset
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform
from shapeaxi import saxi_nets
from shapeaxi.colors import bcolors

import lightning as L

import pickle
from tqdm import tqdm

def main(args):
    
    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)

    NN = getattr(saxi_nets, args.nn)    
    model = NN.load_from_checkpoint(args.model)
    model.eval()
    model.cuda()


    scale_factor = None
    if hasattr(model.hparams, 'scale_factor'):
        scale_factor = model.hparams.scale_factor

    test_ds = SaxiDataset(df, transform=EvalTransform(scale_factor), CN=True, **vars(args))
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    fname = args.csv
    ext = os.path.splitext(fname)[1]

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
            
            X_mesh = model.create_mesh(V, F, CN)
            x = model(X_mesh)
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


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Shape Analysis Explainaiblity and Interpretability predict', conflict_handler='resolve')

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--csv', type=str, help='CSV with column surf', required=True)   
    input_group.add_argument('--nn', help='Type of neural network', type=str, required=True)
    input_group.add_argument('--model', help='Model for prediction', type=str, required=True)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--surf_column', type=str, default='surf_path', help='Column name for the surface data')  
    input_group.add_argument('--class_column', type=str, default=None, help='Column name for the class column')
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    args, unknownargs = parser.parse_known_args()

    NN = getattr(saxi_nets, args.nn)    
    NN.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
