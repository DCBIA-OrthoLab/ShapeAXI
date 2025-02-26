
import argparse

import math
import os
import pandas as pd
import numpy as np 

import monai
import torch

from sklearn.utils import class_weight

import lightning as L

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger

from shapeaxi import saxi_dataset 
from shapeaxi.saxi_transforms import *
from shapeaxi import saxi_nets_lightning
from shapeaxi import saxi_logger

def Saxi_train(args, callbacks):

    deterministic = None
    if args.seed_everything:
        seed_everything(args.seed_everything, workers=True)
        deterministic = True

    DATAMODULE = getattr(saxi_dataset, args.data_module)

    args_d = vars(args)

    data = DATAMODULE(**args_d)
    
    SAXINETS = getattr(saxi_nets_lightning, args.nn)
    model = SAXINETS(**args_d)
    
    logger_neptune = None
    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/saxi',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )
        LOGGER = getattr(saxi_logger, args.logger)    
        image_logger = LOGGER(**args_d)
        callbacks.append(image_logger)

    trainer = Trainer(logger=logger_neptune, 
        max_epochs=args.epochs, 
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=args.find_unused_parameters),
        gradient_clip_val=args.gradient_clip_val,
        deterministic=deterministic)
    trainer.fit(model, datamodule=data, ckpt_path=args.model)

def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
    
    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=args.patience, 
        verbose=True, 
        mode="min"
    )

    Saxi_train(args, [checkpoint_callback, early_stop_callback])


def get_argparse():
    parser = argparse.ArgumentParser(description='Shape Analysis Explainability and Interpretability training script.')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)
    hparams_group.add_argument('--gradient_clip_val', help='Gradient clipping for the trainer', type=float, default=None)
    hparams_group.add_argument('--seed_everything', help='Seed everything for training', type=int, default=None)
    hparams_group.add_argument('--find_unused_parameters', help='Find unused parameters', type=int, default=0)
    
    
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Neural network name', required=True, type=str, default=None)
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    
    input_group.add_argument('--data_module', help='Data module type', required=True, type=str, default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=0)
    output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    ##Logger
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--logger', type=str, help='Logger class name', default=None)
    logger_group.add_argument('--log_every_n_steps', type=int, help='Log every n steps during training', default=10)    
    
    logger_group.add_argument('--neptune_project', type=str, help='Neptune project', default=None)
    logger_group.add_argument('--neptune_tags', type=str, nargs='+', help='Neptune tags', default=None)
    logger_group.add_argument('--neptune_token', type=str, help='Neptune token', default=None)

    #Freesurfer
    # fs_group = parser.add_argument_group('Freesurfer')
    # fs_group.add_argument('--freesurfer', help='Use freesurfer data', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = get_argparse()
    initial_args, unknownargs = parser.parse_known_args()

    if initial_args.nn is not None:
        
        model_args = getattr(saxi_nets_lightning, initial_args.nn)
        parser = model_args.add_model_specific_args(parser)

    if initial_args.data_module is not None:
        data_module = getattr(saxi_dataset, initial_args.data_module)
        parser = data_module.add_data_specific_args(parser)

    if initial_args.logger is not None:
        logger = getattr(saxi_logger, initial_args.logger)
        parser = logger.add_logger_specific_args(parser)

    args = parser.parse_args()
    main(args)