import argparse
import subprocess


import math
import os
import sys
import pandas as pd
import numpy as np 

import torch

from src.saxi_dataset import SaxiDataModule, RandomRemoveTeethTransform, UnitSurfTransform, SaxiDataset
from src.saxi_transforms import TrainTransform, EvalTransform
import src.saxi_nets as saxi_nets
from src.saxi_nets import MonaiUNet
from src.saxi_logger import SaxiImageLogger, TeethNetImageLogger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight
from pl_bolts.models.self_supervised import Moco_v2


def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    mount_point = args.mount_point  
    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))

    saxi_args = vars(args)

    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
        callbacks = [early_stop_callback, checkpoint_callback]

        saxi_data = SaxiDataModule(df_train, df_val, df_test,
                            mount_point = mount_point,
                            batch_size = args.batch_size,
                            num_workers = args.num_workers,
                            model = args.nn,
                            surf_column = args.surf_column,
                            class_column = args.class_column,
                            train_transform = TrainTransform(scale_factor=args.scale_factor),
                            valid_transform = EvalTransform(scale_factor=args.scale_factor),
                            test_transform = EvalTransform(scale_factor=args.scale_factor))
        logger=None
        if args.tb_dir:
            logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
            callbacks.append(SaxiImageLogger())

        elif args.neptune_project:
            logger = NeptuneLogger(
                project=args.neptune_project,
                tags=args.neptune_tags,
                api_key=os.environ['NEPTUNE_API_TOKEN']
            )
            image_logger = SaxiImageLoggerNeptune(num_images=args.num_images)
        
        if args.nn == "SaxiClassification":
            unique_classes = np.sort(np.unique(df_train[args.class_column]))
            unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
            class_weights = unique_class_weights
            saxi_args['class_weights'] = class_weights
            saxi_args['out_classes'] = len(class_weights)

        elif args.nn =="SaxiRegression":
            saxi_args['out_features'] = 1

        SAXINETS = getattr(saxi_nets, args.nn)
        model = SAXINETS(**saxi_args)

        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            log_every_n_steps=args.log_every_n_steps,
            callbacks=callbacks,
            devices=torch.cuda.device_count(), 
            accelerator="gpu", 
            strategy=DDPStrategy(find_unused_parameters=False),
            num_sanity_val_steps=0,
            profiler=args.profiler
        )

        trainer.fit(model, datamodule=saxi_data, ckpt_path=args.model)


    elif args.nn == "SaxiSegmentation":
        class_weights = None

        saxi_data = SaxiDataModule(df_train, df_val, df_test,
                            mount_point = mount_point,
                            batch_size = args.batch_size,
                            num_workers = args.num_workers,
                            model = args.nn,
                            surf_column = 'surf',
                            surf_property = 'UniversalID',
                            #train_transform = RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True),
                            train_transform = UnitSurfTransform(),
                            valid_transform = UnitSurfTransform(),
                            test_transform = UnitSurfTransform())

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
        
        logger=None
        
        if args.tb_dir:
            logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

        image_logger = TeethNetImageLogger()

        model = MonaiUNet(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples)

        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            log_every_n_steps=args.log_every_n_steps,
            callbacks=[early_stop_callback, checkpoint_callback, image_logger],
            devices=torch.cuda.device_count(), 
            accelerator="gpu", 
            strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
            num_sanity_val_steps=0,
            profiler=args.profiler
        )

        trainer.fit(model, datamodule=saxi_data, ckpt_path=args.model)
        trainer.test(datamodule=saxi_data)




def get_argparse():
    parser = argparse.ArgumentParser(description='Shape Analysis Explainability and Interpretability')

    in_group = parser.add_argument_group('Input')

    in_group.add_argument('--csv_train', help='CSV with column surf', type=str, required=True)    
    in_group.add_argument('--csv_valid', help='CSV with column surf', type=str)
    in_group.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)        
    in_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    in_group.add_argument('--class_column', help='Class column name', type=str, default=None)
    in_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    in_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)

    model_group = parser.add_argument_group('Input model')
    model_group.add_argument('--model', help='Model to continue training', type=str, default= None)

    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--nn', help='Neural network name', type=str, default='SaxiClassification')
    hyper_group.add_argument('--base_encoder', help='Base encoder for the feature extraction', type=str, default='resnet18')
    hyper_group.add_argument('--base_encoder_params', help='Base encoder parameters that are passed to build the feature extraction', type=str, default='pretrained=False,n_input_channels=4,spatial_dims=2,num_classes=512')
    hyper_group.add_argument('--hidden_dim', help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', type=int, default=512)
    hyper_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    hyper_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    hyper_group.add_argument('--image_size', help='Image resolution size', type=int, default=256)
    hyper_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hyper_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)   
    hyper_group.add_argument('--batch_size', help='Batch size', type=int, default=6)    
    hyper_group.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)  
    hyper_group.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    hyper_group.add_argument('--scale_factor', help='Scale factor to rescale the shapes', type=float, default=1.0) 

    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default=None)
    logger_group.add_argument('--neptune_project', help='Neptune project', type=str, default=None)
    logger_group.add_argument('--neptune_tags', help='Neptune tags', type=str, default=None)

    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--out', help='Output', type=str, default="./")

    debug_group = parser.add_argument_group('Debug')
    debug_group.add_argument('--profiler', help='Use a profiler', type=str, default=None)

    return parser



if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()
    main(args)