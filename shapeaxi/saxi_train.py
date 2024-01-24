import argparse
import subprocess
import math
import os
import sys
import pandas as pd
import numpy as np 
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import monai
import nibabel as nib
from sklearn.utils import class_weight
from pl_bolts.models.self_supervised import Moco_v2


from shapeaxi.saxi_dataset import SaxiDataModule, SaxiDataset, BrainIBISDataModule
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform, RandomRemoveTeethTransform, UnitSurfTransform, RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform
from shapeaxi import saxi_nets
from shapeaxi.saxi_nets import MonaiUNet, SaxiIcoClassification
from shapeaxi.saxi_logger import SaxiImageLogger, TeethNetImageLogger, ImageLogger

# Training machine learning models

def SaxiClassification_SaxiRegression_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test, early_stop_callback):
    #Initialize the dataset and corresponding data loader for training and validation
    callbacks = [early_stop_callback, checkpoint_callback]
    saxi_args = vars(args)

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



def SaxiSegmentation_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test, logger, early_stop_callback):
     # The dataset and corresponding data loader are initialized for training and validation
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

    image_logger = TeethNetImageLogger()
    model = MonaiUNet(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples)

    trainer = Trainer(logger=logger,max_epochs=args.epochs,log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
        num_sanity_val_steps=0,
        profiler=args.profiler
    )

    trainer.fit(model, datamodule=saxi_data, ckpt_path=args.model)
    # trainer.test(datamodule=saxi_data)



def SaxiIcoClassification_train(args, checkpoint_callback, mount_point, train, val, test, logger, early_stop_callback):

    list_path_ico = [args.path_ico_left,args.path_ico_right]

    #Demographics
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR

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

    #Get number of images
    list_nb_verts_ico = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
    nb_images = list_nb_verts_ico[args.ico_lvl-1]
    
    #Creation of Dataset
    brain_data = BrainIBISDataModule(args.batch_size,list_demographic,train,val,test,list_path_ico,
                                    train_transform = train_transform,
                                    val_and_test_transform=val_and_test_transform,
                                    num_workers=args.num_workers)#MLR

    nbr_features = brain_data.get_features()
    weights = brain_data.get_weigths()
    nbr_demographic = brain_data.get_nbr_demographic()

    if args.ico_lvl == 1:
        radius = 1.76 
    elif args.ico_lvl == 2:
        radius = 1

    saxi_args = vars(args)
    saxi_args['nbr_features'] = nbr_features
    saxi_args['nbr_demographic'] = nbr_demographic
    saxi_args['weights'] = weights
    saxi_args['radius'] = radius

    #Creation of our model
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    #Image Logger (Useful if we use Tensorboard)
    image_logger = ImageLogger(num_features = nbr_features,num_images = nb_images,mean = 0,std=args.noise_lvl)

    #Trainer
    trainer = Trainer(log_every_n_steps=10,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=args.epochs,callbacks=[early_stop_callback,checkpoint_callback,image_logger],accelerator="gpu") #,accelerator="gpu"
    trainer.fit(model,datamodule=brain_data)
    # trainer.test(model, datamodule=brain_data)


def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    # Mount the dataset
    mount_point = args.mount_point
    path_train = os.path.join(mount_point, args.csv_train)
    path_val = os.path.join(mount_point, args.csv_valid)
    path_test = os.path.join(mount_point, args.csv_test)
    
    # Load the data
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)

    #Create the logger for the training
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    
    else:
        logger = TensorBoardLogger(save_dir=mount_point, name=args.tb_name)  
    
    #Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    # Train the model depending on the neural network
    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
        SaxiClassification_SaxiRegression_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test, early_stop_callback)

    elif args.nn == "SaxiSegmentation":
        SaxiSegmentation_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test, logger, early_stop_callback)
    
    elif args.nn == "SaxiIcoClassification":
        SaxiIcoClassification_train(args, checkpoint_callback, mount_point, path_train, path_val, path_test, logger, early_stop_callback)
    
    else:
        raise ValueError ("Unknown neural network name: {}, choose between SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification".format(args.nn))




def get_argparse():
    #This function defines the arguments that can be passed to the script
    parser = argparse.ArgumentParser(description='Shape Analysis Explainability and Interpretability')

    ##Input
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_train', type=str, help='CSV with column surf', required=True)    
    input_group.add_argument('--csv_valid', type=str, help='CSV with column surf')
    input_group.add_argument('--csv_test', type=str, help='CSV with column surf', required=True)        
    input_group.add_argument('--surf_column', type=str, help='Surface column name', default="surf")
    input_group.add_argument('--class_column', type=str, help='Class column name', default=None)
    input_group.add_argument('--mount_point', type=str, help='Dataset mount directory', default="./")
    input_group.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    input_group.add_argument('--path_ico_left', type=str, help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)', default='./3DObject/sphere_f327680_v163842.vtk')
    input_group.add_argument('--path_ico_right', type=str, help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)', default='./3DObject/sphere_f327680_v163842.vtk')

    ##Model
    model_group = parser.add_argument_group('Input model')
    model_group.add_argument('--model', type=str, help='Model to continue training', default=None)

    ##Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification', required=True, choices=["SaxiClassification", "SaxiRegression", "SaxiSegmentation", "SaxiIcoClassification"])
    hyper_group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
    hyper_group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
    hyper_group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
    hyper_group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.35)    
    hyper_group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=1)
    hyper_group.add_argument('--image_size', type=int, help='Image resolution size', default=256)
    hyper_group.add_argument('--lr', type=float, help='Learning rate', default=1e-4,)
    hyper_group.add_argument('--epochs', type=int, help='Max number of epochs', default=200)   
    hyper_group.add_argument('--batch_size', type=int, help='Batch size', default=3)    
    hyper_group.add_argument('--train_sphere_samples', type=int, help='Number of training sphere samples or views used during training and validation', default=4)  
    hyper_group.add_argument('--patience', type=int, help='Patience for early stopping', default=30)
    hyper_group.add_argument('--scale_factor', type=float, help='Scale factor to rescale the shapes', default=1.0) 
    hyper_group.add_argument('--noise_lvl', type=float, help='Noise level (default: 0.01)', default=0.01)
    hyper_group.add_argument('--ico_lvl', type=int, help='Ico level, minimum level is 1 (default: 2)', default=2)
    hyper_group.add_argument('--pretrained',  type=bool, help='Pretrained (default: False)', default=False)
    hyper_group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)
    hyper_group.add_argument('--layer', type=str, help="Layer, choose between 'Att','IcoConv2D','IcoConv1D','IcoLinear' (default: IcoConv2D)", default='IcoConv2D')

    ##Gaussian Filter
    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, help='Mean (default: 0)', default=0)
    gaussian_group.add_argument('--std', type=float, help='Standard deviation (default: 0.005)', default=0.005)


    ##Logger
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', type=int, help='Log every n steps', default=10)    
    logger_group.add_argument('--tb_dir', type=str, help='Tensorboard output dir', default=None)
    logger_group.add_argument('--tb_name', type=str, help='Tensorboard experiment name', default="tensorboard")
    logger_group.add_argument('--neptune_project', type=str, help='Neptune project', default=None)
    logger_group.add_argument('--neptune_tags', type=str, help='Neptune tags', default=None)

    ##Output
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--out', type=str, help='Output', default="./")

    ##Debug
    debug_group = parser.add_argument_group('Debug')
    debug_group.add_argument('--profiler', type=str, help='Use a profiler', default=None)

    return parser


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)