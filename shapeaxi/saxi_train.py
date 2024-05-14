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
torch.set_float32_matmul_precision('high')


from shapeaxi.saxi_dataset import SaxiDataModule, SaxiDataset, SaxiIcoDataModule_fs, SaxiIcoDataModule, SaxiIcoDataset_fs, SaxiIcoDataModule_MT
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform, RandomRemoveTeethTransform, UnitSurfTransform, RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform
from shapeaxi import saxi_nets
from shapeaxi.saxi_nets import MonaiUNet, SaxiIcoClassification
from shapeaxi.saxi_logger import SaxiImageLoggerTensorboard, SaxiImageLoggerTensorboardSegmentation, SaxiImageLoggerTensorboardIco, SaxiImageLoggerTensorboardIco_fs, SaxiImageLoggerNeptune, SaxiImageLoggerNeptune_Ico_fs


def logger_neptune_tensorboard(args):
    image_logger = None
    logger = None

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
        if args.nn == "SaxiSegmentation":
            image_logger = SaxiImageLoggerTensorboardSegmentation()
        elif args.nn == "SaxiIcoClassification":
            image_logger = SaxiImageLoggerTensorboardIco()
        elif args.nn == "SaxiIcoClassification_fs" or args.nn == "SaxiRing":
            image_logger = SaxiImageLoggerTensorboardIco_fs()
        else:
            image_logger = SaxiImageLoggerTensorboard()

    elif args.neptune_project:
        logger = NeptuneLogger(
            project=args.neptune_project,
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )

        if args.nn == "SaxiIcoClassification_fs" or args.nn == "SaxiRing":
            image_logger = SaxiImageLoggerNeptune_Ico_fs(num_images=args.num_images)
        else:
            image_logger = SaxiImageLoggerNeptune(num_images=args.num_images)

    return logger, image_logger


def Saxi_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test, early_stop_callback):

    data = SaxiDataModule(df_train, df_val, df_test,mount_point = mount_point,batch_size = args.batch_size,num_workers = args.num_workers,model = args.nn,surf_column = args.surf_column,class_column = args.class_column, train_transform = TrainTransform(scale_factor=args.scale_factor),valid_transform = EvalTransform(scale_factor=args.scale_factor),test_transform = EvalTransform(scale_factor=args.scale_factor))

    saxi_args = vars(args)
    if args.nn == "SaxiClassification":
        unique_classes = np.sort(np.unique(df_train[args.class_column]))
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
        class_weights = unique_class_weights
        saxi_args['class_weights'] = class_weights
        saxi_args['out_classes'] = len(class_weights)

    elif args.nn =="SaxiRegression":
        saxi_args['out_features'] = 1
    
    else:
        print("Check for Segmentation the use of Monai of SaxiNets")
        # model = MonaiUNet(args, out_channels = 34, class_weights=None, image_size=320, train_sphere_samples=args.train_sphere_samples)

    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    callbacks = [early_stop_callback, checkpoint_callback]
    logger, image_logger = logger_neptune_tensorboard(args)

    if image_logger:
        callbacks.append(image_logger)

    trainer = Trainer(logger=logger,max_epochs=args.epochs,log_every_n_steps=args.log_every_n_steps,callbacks=callbacks,devices=torch.cuda.device_count(), accelerator="gpu", strategy=DDPStrategy(find_unused_parameters=False),num_sanity_val_steps=0,profiler=args.profiler)
    trainer.fit(model, datamodule=data, ckpt_path=args.model)


def SaxiIcoClassification_train(args, checkpoint_callback, mount_point, train, val, test, early_stop_callback):

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
    brain_data = SaxiIcoDataModule(args.batch_size,list_demographic,train,val,test,list_path_ico,train_transform = train_transform,val_and_test_transform=val_and_test_transform,num_workers=args.num_workers,name_class=args.class_column)#MLR
    weights = brain_data.get_weigths()
    nbr_demographic = brain_data.get_nbr_demographic()

    df_train = pd.read_csv(train)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    

    print('Number of classes:',len(unique_class_weights))

    if args.ico_lvl == 1:
        args.radius = 1.76 
    elif args.ico_lvl == 2:
        args.radius = 1

    saxi_args = vars(args)
    saxi_args['nbr_demographic'] = nbr_demographic
    saxi_args['weights'] = weights
    saxi_args['out_classes'] = 2
    saxi_args['out_size'] = 256

    #Creation of our model
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    callbacks = [early_stop_callback, checkpoint_callback]
    logger, image_logger = logger_neptune_tensorboard(args)

    if image_logger:
        callbacks.append(image_logger)
    #Trainer
    trainer = Trainer(log_every_n_steps=10,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=args.epochs,callbacks=callbacks,accelerator="gpu") #,accelerator="gpu"
    trainer.fit(model,datamodule=brain_data)


def SaxiIcoClassification_fs_train(args, checkpoint_callback, mount_point, train, val, test, early_stop_callback):
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
   
    #Creation of Dataset
    brain_data = SaxiIcoDataModule_fs(args.batch_size,train,val,test,train_transform=train_transform,val_and_test_transform=val_and_test_transform,num_workers=args.num_workers,name_class=args.class_column,freesurfer_path=args.fs_path,normalize_features=True)
    df_train = pd.read_csv(train)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    nb_classes = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    

    print('Number of classes:',len(nb_classes))

    saxi_args = vars(args)
    saxi_args['out_classes'] = len(nb_classes)
    saxi_args['out_size'] = 256

    #Creation of our model
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    callbacks = [early_stop_callback, checkpoint_callback]
    logger, image_logger = logger_neptune_tensorboard(args)

    if image_logger:
        callbacks.append(image_logger)

    trainer = Trainer(log_every_n_steps=args.log_every_n_steps,logger=logger,max_epochs=args.epochs,callbacks=callbacks,accelerator="gpu", devices=torch.cuda.device_count())
    trainer.fit(model,datamodule=brain_data)


def SaxiRing_train(args, checkpoint_callback, mount_point, train, val, test, early_stop_callback):
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

    saxi_args = vars(args)

    df_train = pd.read_csv(train)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    nb_classes = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column])) 
   
    if args.fs_path is None:
        df_val = pd.read_csv(val)
        df_test = pd.read_csv(test)
        data = SaxiDataModule(df_train, df_val, df_test,mount_point = mount_point,batch_size = args.batch_size,num_workers = args.num_workers,model = args.nn,surf_column = args.surf_column,class_column = args.class_column, train_transform = TrainTransform(scale_factor=args.scale_factor),valid_transform = EvalTransform(scale_factor=args.scale_factor),test_transform = EvalTransform(scale_factor=args.scale_factor))
        saxi_args['class_weights'] = nb_classes

    else:
        data = SaxiIcoDataModule_fs(args.batch_size,train,val,test,train_transform=train_transform,val_and_test_transform=val_and_test_transform,num_workers=args.num_workers,name_class=args.class_column,freesurfer_path=args.fs_path)
    
    saxi_args['out_classes'] = len(nb_classes)  
    saxi_args['out_size'] = 256

    print("Number of classes:",len(nb_classes))

    #Creation of our model
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    callbacks = [early_stop_callback, checkpoint_callback]
    logger, image_logger = logger_neptune_tensorboard(args)

    if image_logger:
        callbacks.append(image_logger)

    trainer = Trainer(log_every_n_steps=args.log_every_n_steps,logger=logger,max_epochs=args.epochs,callbacks=callbacks,accelerator="gpu", devices=torch.cuda.device_count())
    trainer.fit(model,datamodule=data)


def SaxiRingMT_train(args, checkpoint_callback, mount_point, train, val, test, early_stop_callback):
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

    saxi_args = vars(args)

    df_train = pd.read_csv(train)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    nb_classes = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column])) 
    
    data = SaxiIcoDataModule_MT(args.batch_size,train,val,test,train_transform=train_transform,val_and_test_transform=val_and_test_transform,num_workers=args.num_workers,name_class=args.class_column,freesurfer_path=args.fs_path)
    
    saxi_args['out_classes'] = len(nb_classes)  
    saxi_args['out_size'] = 256

    print("Number of classes:",len(nb_classes))

    #Creation of our model
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    callbacks = [early_stop_callback, checkpoint_callback]
    logger, image_logger = logger_neptune_tensorboard(args)

    if image_logger:
        callbacks.append(image_logger)

    trainer = Trainer(log_every_n_steps=args.log_every_n_steps,logger=logger,max_epochs=args.epochs,callbacks=callbacks,accelerator="gpu", devices=torch.cuda.device_count())
    trainer.fit(model,datamodule=data)


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
    
    #Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    # Train the model depending on the neural network
    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression" or args.nn == "SaxiSegmentation":
        Saxi_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test, early_stop_callback)
    
    elif args.nn == "SaxiIcoClassification":
        SaxiIcoClassification_train(args, checkpoint_callback, mount_point, path_train, path_val, path_test, early_stop_callback)

    elif args.nn == "SaxiIcoClassification_fs":
        SaxiIcoClassification_fs_train(args, checkpoint_callback, mount_point, path_train, path_val, path_test, early_stop_callback)
    
    elif args.nn == "SaxiRing" or args.nn == "SaxiRingClassification":
        SaxiRing_train(args, checkpoint_callback, mount_point, path_train, path_val, path_test, early_stop_callback)
    
    elif args.nn == "SaxiRingMT":
        SaxiRingMT_train(args, checkpoint_callback, mount_point, path_train, path_val, path_test, early_stop_callback)
    
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
    input_group.add_argument('--fs_path', type=str, help='Path to freesurfer folder', default=None)

    ##Model
    model_group = parser.add_argument_group('Input model')
    model_group.add_argument('--model', type=str, help='Model to continue training', default=None)

    ##Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification, SaxiIcoClassification_fs, SaxiRing, SaxiRingClassification', required=True, choices=["SaxiClassification", "SaxiRegression", "SaxiSegmentation", "SaxiIcoClassification", "SaxiIcoClassification_fs", "SaxiRing", "SaxiRingClassification", "SaxiRingMT"])
    hyper_group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
    hyper_group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
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
    logger_group.add_argument('--num_images', type=int, help='Number of images to log', default=12)

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