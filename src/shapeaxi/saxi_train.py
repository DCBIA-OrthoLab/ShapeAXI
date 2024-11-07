
import argparse

import math
import os
import pandas as pd
import numpy as np 

import monai
import torch

from sklearn.utils import class_weight

import lightning as L
 
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger

from shapeaxi import saxi_dataset 
from shapeaxi.saxi_transforms import *
from shapeaxi import saxi_nets_lightning
from shapeaxi import saxi_logger


# def logger_neptune_tensorboard(args):
#     image_logger = None
#     logger = None

#     if args.tb_dir:
#         logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
#         image_logger = {
#             "SaxiSegmentation": SaxiImageLoggerTensorboardSegmentation,
#             "SaxiIcoClassification": SaxiImageLoggerTensorboardIco,
#             "SaxiIcoClassification_fs": SaxiImageLoggerTensorboardIco_fs,
#             "SaxiRing": SaxiImageLoggerTensorboardIco_fs,
#         }.get(args.nn, SaxiImageLoggerTensorboard)()

#     elif args.neptune_project:
#         logger = NeptuneLogger(
#             project=args.neptune_project,
#             tags=args.neptune_tags,
#             api_key=args.neptune_token,
#         )
#         image_logger = {
#             "SaxiIcoClassification_fs": SaxiImageLoggerNeptune_Ico_fs,
#             "SaxiRing": SaxiImageLoggerNeptune_Ico_fs,
#             "SaxiMHA": SaxiImageLoggerNeptune_Ico_fs,
#             "SaxiRingMT": SaxiImageLoggerNeptune_SaxiRingMT,
#         }.get(args.nn, SaxiImageLoggerNeptune)(num_images=args.num_images)

#     return logger, image_logger


def list_transforms(args):
    #Transformation
    list_train_transform = [] 
    list_train_transform.append(CenterTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())        
    list_train_transform.append(GaussianNoisePointTransform(args.mean,args.std)) #Do not use this transformation if your object is not a sphere
    list_train_transform.append(NormalizePointTransform()) #Do not use this transformation if your object is not a sphere
    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterTransform())
    list_val_and_test_transform.append(NormalizePointTransform())
    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    return train_transform, val_and_test_transform


def Saxi_train(args, callbacks):

    train_transform, val_and_test_transform = list_transforms(args)

    # if args.freesurfer != 0:
    #     print("Freesurfer")
    #     data = SaxiFreesurferDataModule(args.batch_size,train,val,test,train_transform=train_transform,val_and_test_transform=val_and_test_transform,num_workers=args.num_workers,name_class=args.class_column,freesurfer_path=args.fs_path)


    DATAMODULE = getattr(saxi_dataset, args.data_module)

    args_d = vars(args)


    train_transform = TrainTransform(scale_factor=args.scale_factor)
    valid_transform = EvalTransform(scale_factor=args.scale_factor)
    
    args_d['train_transform'] = train_transform
    args_d['valid_transform'] = valid_transform
    args_d['test_transform'] = valid_transform

    data = DATAMODULE(**args_d)
    
    # unique_classes = np.sort(np.unique(df_train[args.class_column]))
    # unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
    # class_weights = unique_class_weights
    # saxi_args['class_weights'] = class_weights
    # saxi_args['out_classes'] = len(class_weights)
    # saxi_args['out_size'] = 256

    # #Regression
    # saxi_args['out_features'] = 1
    
    # #IcoConv
    # if args.nn == "SaxiIcoClassification":
    #     list_path_ico = [args.path_ico_left,args.path_ico_right]
    #     #Demographics
    #     list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR
    #     train_transform, val_and_test_transform = list_transforms(args)
    #     #Get number of images
    #     list_nb_verts_ico = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
    #     nb_images = list_nb_verts_ico[args.ico_lvl-1]
    #     saxi_args['nbr_demographic'] = nbr_demographic
    #     saxi_args['weights'] = weights

    #     if args.ico_lvl == 1:
    #         args.radius = 1.76 
    #     elif args.ico_lvl == 2:
    #         args.radius = 1
    
    SAXINETS = getattr(saxi_nets_lightning, args.nn)
    model = SAXINETS(**args_d)

    # callbacks = [early_stop_callback, checkpoint_callback]
    # logger, image_logger = logger_neptune_tensorboard(args)
    
    logger_neptune = None
    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/saxinets',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )
        LOGGER = getattr(saxi_logger, args.logger)    
        image_logger = LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    trainer = Trainer(logger=logger_neptune,max_epochs=args.epochs, log_every_n_steps=args.log_every_n_steps,callbacks=callbacks,devices=torch.cuda.device_count(), accelerator="gpu", strategy=DDPStrategy(find_unused_parameters=False),num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data, ckpt_path=args.model)

def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=False
    )
    
    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=200, 
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

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Neural network name', required=True, type=str, 
                             choices=['SaxiClassification', 'SaxiRegression', 'SaxiSegmentation', 'SaxiIcoClassification',
                                       'SaxiIcoClassification_fs', 'SaxiRing', 'SaxiRingClassification', 'SaxiRingMT', 
                                       'SaxiMHA', 'SaxiMHAFBClassification', 'SaxiMHAFBRegression', 'SaxiOctree', 'SaxiMHAClassification', 
                                       'SaxiMHAFBRegression_V', 'SaxiPointTransformer', 'SaxiMHAFBClassification_V','SaxiIdxAE', 'SaxiDenoiseUnet', 'SaxiDDPMUnet'])

    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    
    input_group.add_argument('--data_module', help='Data module type', required=True, type=str, default=None)
    
    input_group.add_argument('--scale_factor', type=float, help='Scale factor for the shapes', default=1.0)

    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, help='Mean (default: 0)', default=0)
    gaussian_group.add_argument('--std', type=float, help='Standard deviation (default: 0.005)', default=0.005)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=0)
    output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    ##Logger
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--logger', type=str, help='Logger class name', default=None)
    logger_group.add_argument('--log_steps', type=int, help='Log steps for the callback (neptune)', default=10)    
    logger_group.add_argument('--log_every_n_steps', type=int, help='Log every n steps', default=10)    
    logger_group.add_argument('--tb_dir', type=str, help='Tensorboard output dir', default=None)
    logger_group.add_argument('--tb_name', type=str, help='Tensorboard experiment name', default="tensorboard")
    logger_group.add_argument('--neptune_project', type=str, help='Neptune project', default=None)
    logger_group.add_argument('--neptune_tags', type=str, nargs='+', help='Neptune tags', default=None)
    logger_group.add_argument('--neptune_token', type=str, help='Neptune token', default=None)
    logger_group.add_argument('--num_images', type=int, help='Number of images to log', default=12)
    

    #Freesurfer
    fs_group = parser.add_argument_group('Freesurfer')
    fs_group.add_argument('--freesurfer', help='Use freesurfer data', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = get_argparse()
    initial_args, unknownargs = parser.parse_known_args()
    model_args = getattr(saxi_nets_lightning, initial_args.nn)
    parser = model_args.add_model_specific_args(parser)

    data_module = getattr(saxi_dataset, initial_args.data_module)
    parser = data_module.add_data_specific_args(parser)

    args = parser.parse_args()
    main(args)