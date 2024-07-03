import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from shapeaxi.saxi_dataset import SaxiDataModule, SaxiDataModuleVF
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform
from shapeaxi import saxi_nets
from shapeaxi import saxi_logger

import lightning as L

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger



def main(args):
    
    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)

    NN = getattr(saxi_nets, args.nn)    
    model = NN(**vars(args))

    train_transform = TrainTransform(scale_factor=args.scale_factor)
    valid_transform = EvalTransform(scale_factor=args.scale_factor)
    lotus_data = SaxiDataModule(df_train, df_val, df_val, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, surf_column=args.surf_column, class_column=args.class_column, scalar_column=args.scalar_column, train_transform=train_transform, valid_transform=valid_transform, drop_last=False)
    # lotus_data = SaxiDataModuleVF(df_train, df_val, df_val, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, surf_column=args.surf_column, class_column=args.class_column, train_transform=train_transform, valid_transform=valid_transform, drop_last=False)

    # lotus_data.setup()
    # dl = lotus_data.train_dataloader()
    # for batch in dl:
    #     V, F = batch
    #     print(V.shape, F.shape) 

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True,
    )

    callbacks.append(checkpoint_callback)


    if args.monitor:
        checkpoint_callback_acc = ModelCheckpoint(
            dirpath=args.out,
            filename='{epoch}-{' + args.monitor + '}:.2f}',
            save_top_k=2,
            monitor=args.monitor,
            save_last=True,
            mode='max'
        )

        callbacks.append(checkpoint_callback_acc)

    
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=True, mode="min")
    callbacks.append(early_stop_callback)

    
    logger_neptune = None

    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/saxi',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )
        LOGGER = getattr(saxi_logger, args.logger)    
        image_logger = LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    
    trainer = Trainer(
        logger=logger_neptune,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        # strategy=DDPStrategy(),
        strategy='ddp'
    )
    
    trainer.fit(model, datamodule=lotus_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Shape Analysis Explainaiblity and Interpretability train', conflict_handler='resolve')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=2)
    hparams_group.add_argument('--monitor', help='which other variable to monitor to save checkpoints', type=str, default=None)

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="USAEReconstruction")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--surf_column', type=str, default='surf_path', help='Column name for the surface data')  
    input_group.add_argument('--class_column', type=str, default=None, help='Column name for the class column')  
    input_group.add_argument('--scalar_column', type=str, default=None, help='Column name for the scalar column')  
    input_group.add_argument('--scale_factor', type=float, default=None, help='Use a common scale factor')      
  
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=0)
    output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, default="USAEReconstructionNeptuneLogger")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=20)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(saxi_nets, args.nn)    
    NN.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
