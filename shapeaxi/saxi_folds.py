import os
import sys
import glob 
import subprocess
import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import torch
torch.set_float32_matmul_precision('high')

from shapeaxi import compute_min_scale, split_train_eval, saxi_eval, saxi_predict, saxi_train, saxi_gradcam
from shapeaxi.colors import bcolors

def get_last_checkpoint(checkpoint_dir):
    # Get the last checkpoint
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=os.path.getctime, reverse=True) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

def get_val_loss(checkpoint_path):
    # Get the last checkpoint
    checkpoint_fn = os.path.basename(checkpoint_path)
    checkpoint_dict = eval('dict(%s)' % checkpoint_fn.replace('-',',').replace(' ','').replace('.ckpt', ''))
    return checkpoint_dict['val_loss']

def get_best_checkpoint(checkpoint_dir):
    # Get the last checkpoint
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=get_val_loss) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

def get_argparse_dict(parser):
    # Get the default arguments from the parser
    default = {}
    for action in parser._actions:
        if action.dest != "help":
            default[action.dest] = action.default
    return default

def replace_extension(fname, new_extension):
    return os.path.splitext(fname)[0] + new_extension

def get_output_filename(fname, suffix):
    return replace_extension(fname, f'_{suffix}')

def positive_float(value):
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")
    return fvalue

def positive_int(value):
    fvalue = int(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")
    return fvalue

def aggregate(ext, args, f, out_prediction_agg):
    if args.csv is not None:
        csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
    else:
        csv_test = args.csv_train.replace(ext, '_fold{f}_test.csv').format(f=f)

    saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
    best_model_path = get_best_checkpoint(saxi_train_args_out)

    fname = os.path.basename(csv_test)
    out_prediction_fn = os.path.join(args.out, 'test', 'fold{f}'.format(f=f), os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

    return out_prediction_fn

def main(args, arg_groups):

    #Main function
    create_folds = False
    scale_factor = None
    
    #Creation of the folder of output if it does not exist
    os.makedirs(args.out, exist_ok=True)
    
    if args.csv_train is None and args.csv_test is None:
        ext = os.path.splitext(args.csv)[1]
    else:
        ext = os.path.splitext(args.csv_train)[1]

    if ext != '.csv' and ext != '.parquet':
        raise ValueError(f'Invalid file extension {ext}')
    

#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                  Scale Factor                                                                                     #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


    if args.compute_scale_factor:
        print(bcolors.INFO, "Start computing the scale factor", bcolors.ENDC)
        # Compute the scale factor for the dataset
        compute_min_scale_args_out = os.path.basename(args.csv)   
        compute_min_scale_args_out = compute_min_scale_args_out.replace(ext, '_scale_factor' + ext)
        compute_min_scale_args_out = os.path.join(args.out, compute_min_scale_args_out)

        if os.path.exists(compute_min_scale_args_out):
            # If the scale factor has already been computed, load it
            df_scale = pd.read_csv(compute_min_scale_args_out)
            scale_factor = np.min(df_scale['surf_scale'])

        else:
            # Compute the scale factor
            compute_min_scale_args = get_argparse_dict(compute_min_scale.get_argparse())
            compute_min_scale_args['csv'] = args.csv
            compute_min_scale_args['mount_point'] = args.mount_point   
            compute_min_scale_args['surf_column'] = args.surf_column
            compute_min_scale_args['out'] = compute_min_scale_args_out
            compute_min_scale_args = Namespace(**compute_min_scale_args)
            scale_factor = compute_min_scale.main(compute_min_scale_args)

        print(bcolors.SUCCESS, "End computing the scale factor", bcolors.ENDC)
    
    elif args.compute_features:
        print(bcolors.INFO, "Start normalizing features", bcolors.ENDC)

        compute_features_args_out = get_argparse_dict(compute_features.get_argparse())
        compute_features_args_out['csv'] = args.csv
        compute_features_args_out['out'] = "/CMF/data/floda"   
        compute_features_args_out['fs_path'] = args.fs_path

        compute_features_args_out = Namespace(**compute_features_args_out)
        compute_features.main(compute_features_args_out)
        print(bcolors.SUCCESS, "End normalizing features", bcolors.ENDC)

    else:
        if args.csv is not None:
            #Check if the input file has a column with the scale factor
            df = pd.read_csv(os.path.join(args.mount_point, args.csv))
            if args.column_scale_factor in df.columns:
                scale_factor = df[args.column_scale_factor].min()
        else:
            df = pd.read_csv(os.path.join(args.mount_point, args.csv_train))
            if args.column_scale_factor in df.columns:
                scale_factor = df[args.column_scale_factor].min()


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                     Split                                                                                         #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


    create_folds = False
    for f in range(args.folds):
        if args.csv is not None:
            csv_train = get_output_filename(args.csv, f'train_fold{f}_train.csv')
            csv_valid = get_output_filename(args.csv, f'train_fold{f}_train_test.csv')
            csv_test = get_output_filename(args.csv, f'train_fold{f}_test.csv')
         
        else:
            csv_train = get_output_filename(args.csv_train, f'fold{f}_train_train.csv')
            csv_valid = get_output_filename(args.csv_train, f'fold{f}_train_test.csv')
            csv_test = get_output_filename(args.csv_train, f'fold{f}_test.csv')

        if not os.path.exists(csv_train) or not os.path.exists(csv_valid) or not os.path.exists(csv_test):
            create_folds = True
            break
    
    if create_folds:
        # Check if the user gives as input the first train and test dataset 
        if args.csv_train is None and args.csv_test is None:
            # First split of the data to use only the train set for the split into the different folds 
            print(bcolors.INFO, "Start spliting data", bcolors.ENDC)

            split_csv_args = get_argparse_dict(split_train_eval.get_argparse())
            split_csv_args['csv'] = args.csv
            split_csv_args['split'] = args.valid_split
            split_csv_args['group_by'] = args.group_by
            split_csv_args['mount_point'] = args.mount_point
            split_csv_args = Namespace(**split_csv_args)

            split_train_eval.first_split_data(split_csv_args)
            print(bcolors.SUCCESS, "End spliting data", bcolors.ENDC)


        # Creation of test and train dataset for each fold
        print(f"{bcolors.INFO}Start creating {args.folds} folds{bcolors.ENDC}")

        if args.csv is not None:
            csv_train = args.csv.replace(ext, '_train.csv')
        else:
            csv_train = args.csv_train

        split_csv_args = get_argparse_dict(split_train_eval.get_argparse())
        for k in arg_groups['Split']:
            split_csv_args[k] = arg_groups['Split'][k]
        split_csv_args['csv'] = csv_train
        split_csv_args['mount_point'] = args.mount_point
        split_csv_args = Namespace(**split_csv_args)

        split_train_eval.split_data_folds_test_train(split_csv_args)

        # Creation of the val dataset
        for f in range(args.folds):
            # Use of split_train_eval to split the train into train and validation
            if args.csv is not None:
                csv_train = get_output_filename(args.csv, f'train_fold{f}_train.csv')
            else:
                csv_train = get_output_filename(args.csv_train, f'fold{f}_train.csv')

            split_csv_args = get_argparse_dict(split_train_eval.get_argparse())

            for k in arg_groups['Split']:
                split_csv_args[k] = arg_groups['Split'][k]

            split_csv_args['csv'] = csv_train
            split_csv_args['mount_point'] = args.mount_point

            split_csv_args = Namespace(**split_csv_args)
            split_train_eval.split_data_folds_train_eval(split_csv_args)

        print(f"{bcolors.SUCCESS}End of creating the {args.folds} folds {bcolors.ENDC}")


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                     Train                                                                                         #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


    for f in range(0, args.folds):
        #Train the model for each fold
        print(bcolors.INFO, "Start training for fold {f}".format(f=f), bcolors.ENDC)
        if args.csv is not None:
            csv_train = args.csv.replace(ext, '_train_fold{f}_train_train.csv').format(f=f)
            csv_valid = args.csv.replace(ext, '_train_fold{f}_train_test.csv').format(f=f)
            csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
        else:
            csv_train = args.csv_train.replace(ext, '_fold{f}_train_train.csv').format(f=f)
            csv_valid = args.csv_train.replace(ext, '_fold{f}_train_test.csv').format(f=f)
            csv_test = args.csv_train.replace(ext, '_fold{f}_test.csv').format(f=f)

        saxi_train_args = get_argparse_dict(saxi_train.get_argparse())

        for k in arg_groups['Train']:
            if k in saxi_train_args:        
                saxi_train_args[k] = arg_groups['Train'][k]

        saxi_train_args['csv_train'] = csv_train
        saxi_train_args['csv_test'] = csv_test
        saxi_train_args['csv_valid'] = csv_valid
        saxi_train_args['scale_factor'] = scale_factor
        saxi_train_args['out'] = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
        last_checkpoint = get_last_checkpoint(saxi_train_args['out'])

        if last_checkpoint is None:
            command = [sys.executable, '-m', 'shapeaxi.saxi_train']

            for k in saxi_train_args:
                if saxi_train_args[k]:
                    command.append('--' + str(k))
                    command.append(str(saxi_train_args[k]))
            
            env = os.environ.copy()
            env['NEPTUNE_API_TOKEN'] = os.environ['NEPTUNE_API_TOKEN']
            subprocess.run(command, env=env)

        print(bcolors.SUCCESS, "End training for fold {f}".format(f=f), bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                      Test                                                                                         #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################

    # Initialize the best score and the best model fold to 0
    best_eval_metric = 0.0
    best_model_fold = ""

    for f in range(0, args.folds):
        #Test the model for each fold
        print(bcolors.INFO, "Start test for fold {f}".format(f=f), bcolors.ENDC)
        if args.csv is not None:
            csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
        else:
            csv_test = args.csv_train.replace(ext, '_fold{f}_test.csv').format(f=f)
        saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
        best_model_path = get_best_checkpoint(saxi_train_args_out)
        saxi_predict_args = get_argparse_dict(saxi_predict.get_argparse())
        
        saxi_predict_args['csv'] = csv_test
        saxi_predict_args['model'] = best_model_path
        saxi_predict_args['surf_column'] = args.surf_column
        saxi_predict_args['class_column'] = args.class_column
        saxi_predict_args['mount_point'] = args.mount_point
        saxi_predict_args['nn'] = args.nn
        saxi_predict_args['crown_segmentation'] = args.crown_segmentation
        saxi_predict_args['fdi'] = args.fdi
        saxi_predict_args['path_ico_right'] = args.path_ico_right
        saxi_predict_args['path_ico_left'] = args.path_ico_left
        saxi_predict_args['fs_path'] = args.fs_path
        saxi_predict_args['device'] = 'cuda:0'
        saxi_predict_args['out'] = os.path.join(args.out, 'test', 'fold{f}'.format(f=f))

        saxi_predict_args = Namespace(**saxi_predict_args)
        fname = os.path.basename(csv_test)
        out_prediction = os.path.join(saxi_predict_args.out, os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

        if not os.path.exists(out_prediction):
            saxi_predict.main(saxi_predict_args)

        print(bcolors.SUCCESS, "End test for fold {f}".format(f=f), bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   Evaluation                                                                                      #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################

        #Run the evaluation for the prediction
        print(bcolors.INFO, "Start evaluation for fold {f}".format(f=f), bcolors.ENDC)
        saxi_eval_args = get_argparse_dict(saxi_eval.get_argparse())

        saxi_eval_args['csv'] = out_prediction
        saxi_eval_args['class_column'] = args.class_column
        saxi_eval_args['csv_prediction_column'] = args.csv_prediction_column
        saxi_eval_args['nn'] = args.nn
        saxi_eval_args['eval_metric'] = args.eval_metric
        saxi_eval_args['mount_point'] = args.mount_point
        saxi_eval_args = Namespace(**saxi_eval_args)

        current_weighted_eval_metric = saxi_eval.main(saxi_eval_args)

        if current_weighted_eval_metric > best_eval_metric:
            best_eval_metric = current_weighted_eval_metric
            best_model_fold = f'{f}'

        print(bcolors.SUCCESS, "End evaluation for fold {f}".format(f=f), bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   Aggregate                                                                                       #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":

        print(bcolors.INFO, "Start aggregate for all folds".format(f=f), bcolors.ENDC)

        # Create a single dataframe and prob array
        out_prediction_agg = []

        if args.nn == "SaxiClassification":
            # Create a single dataframe and prob array
            out_prediction_probs_agg = []

            for f in range(0, args.folds):
                
                out_prediction_fn = aggregate(ext, args, f, out_prediction_agg)
                out_prediction_agg.append(pd.read_csv(out_prediction_fn))

                probs_fn = out_prediction_fn.replace("_prediction.csv", "_probs.pickle")
                out_prediction_probs_agg.append(pickle.load(open(probs_fn, 'rb')))
                
            # Concatenate all datragrames and probs
            out_prediction_agg = pd.concat(out_prediction_agg)
            if args.csv is not None:
                fname = os.path.basename(args.csv.replace('.csv', '_train.csv'))
            else:
                fname = os.path.basename(args.csv_train)
            out_prediction_agg_fn = os.path.join(args.out, 'test', fname.replace(ext, "_aggregate_prediction" + ext))
            out_prediction_agg.to_csv(out_prediction_agg_fn, index=False)

            out_prediction_probs_agg = np.concatenate(out_prediction_probs_agg)
            out_prediction_probs_agg_fn = out_prediction_agg_fn.replace("_prediction.csv", "_probs.pickle")
            pickle.dump(out_prediction_probs_agg, open(out_prediction_probs_agg_fn, 'wb'))

        elif args.nn == "SaxiRegression":
            
            for f in range(0, args.folds):

                out_prediction_fn = aggregate(ext, args, f, out_prediction_agg)
                out_prediction_agg.append(pd.read_csv(out_prediction_fn))
            
            # Concatenate all datragrames and probs
            out_prediction_agg = pd.concat(out_prediction_agg)
            out_prediction_agg_fn = os.path.join(args.out, 'test', fname.replace(ext, "_aggregate_prediction" + ext))
            out_prediction_agg.to_csv(out_prediction_agg_fn, index=False)
        
        
        #Run the evaluation for the aggregate
        saxi_eval_args = get_argparse_dict(saxi_eval.get_argparse())
        saxi_eval_args['csv'] = out_prediction_agg_fn
        saxi_eval_args['csv_true_column'] = args.class_column
        saxi_eval_args['nn'] = args.nn
        saxi_eval_args['eval_metric'] = args.eval_metric
        saxi_eval_args['mount_point'] = args.mount_point
        saxi_eval_args = Namespace(**saxi_eval_args)    
        saxi_eval.main(saxi_eval_args)

        print(bcolors.SUCCESS, "END aggregate prediction for ALL folds", bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                Explainability                                                                                     #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


    for f in range(0, args.folds):
        print(bcolors.INFO, "Start explainability for fold {f}".format(f=f), bcolors.ENDC)
        if args.csv is not None:
            csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
        else:
            csv_test = args.csv_train.replace(ext, '_fold{f}_test.csv').format(f=f)
        fname = os.path.basename(csv_test)
        path_to_csv_test = os.path.join(args.mount_point, csv_test)
        df_test = pd.read_csv(path_to_csv_test)
        saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))    
        best_model_path = get_best_checkpoint(saxi_train_args_out)
        saxi_predict_args_out = os.path.join(args.out, 'test', 'fold{f}'.format(f=f))
        out_prediction = os.path.join(saxi_predict_args_out, os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

        if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
            saxi_gradcam_args = get_argparse_dict(saxi_gradcam.get_argparse())
            saxi_gradcam_args['nn'] = args.nn
            saxi_gradcam_args['csv_test'] = out_prediction
            saxi_gradcam_args['surf_column'] = args.surf_column
            saxi_gradcam_args['class_column'] = args.class_column
            saxi_gradcam_args['num_workers'] = args.num_workers
            saxi_gradcam_args['model'] = best_model_path
            saxi_gradcam_args['target_layer'] = args.target_layer
            saxi_gradcam_args['mount_point'] = args.mount_point
            saxi_gradcam_args['fps'] = args.fps
            saxi_gradcam_args['device'] = 'cuda:0'

            if args.nn == "SaxiClassification":
                for target_class in df_test[args.class_column].unique():
                    saxi_gradcam_args['target_class'] = target_class

            else:
                saxi_gradcam_args['target_class'] = None
            
            saxi_gradcam_args = Namespace(**saxi_gradcam_args)
            saxi_gradcam.main(saxi_gradcam_args)


        elif args.nn == "SaxiIcoClassification" or args.nn == "SaxiIcoClassification_fs":
            
            if args.csv is not None:
                csv_train = args.csv.replace(ext, '_train_fold{f}_train_train.csv').format(f=f)
                csv_valid = args.csv.replace(ext, '_train_fold{f}_train_test.csv').format(f=f)
            else:
                csv_train = args.csv_train.replace(ext, '_fold{f}_train_train.csv').format(f=f)
                csv_valid = args.csv_train.replace(ext, '_fold{f}_train_test.csv').format(f=f)

            saxi_gradcam_args = get_argparse_dict(saxi_gradcam.get_argparse())
            saxi_gradcam_args['nn'] = args.nn
            saxi_gradcam_args['csv_test'] = out_prediction
            saxi_gradcam_args['csv_train'] = csv_train
            saxi_gradcam_args['csv_valid'] = csv_valid
            saxi_gradcam_args['surf_column'] = args.surf_column
            saxi_gradcam_args['class_column'] = args.class_column
            saxi_gradcam_args['num_workers'] = args.num_workers
            saxi_gradcam_args['model'] = best_model_path
            saxi_gradcam_args['target_layer'] = args.target_layer
            saxi_gradcam_args['mount_point'] = args.mount_point
            saxi_gradcam_args['fps'] = args.fps
            saxi_gradcam_args['path_ico_right'] = args.path_ico_right
            saxi_gradcam_args['path_ico_left'] = args.path_ico_left
            saxi_gradcam_args['target_class'] = 1
            saxi_gradcam_args['device'] = 'cuda:0'
            saxi_gradcam_args['fs_path'] = args.fs_path
            saxi_gradcam_args['out'] = os.path.join(saxi_predict_args_out, os.path.basename(best_model_path))
            saxi_gradcam_args = Namespace(**saxi_gradcam_args)
            saxi_gradcam.main(saxi_gradcam_args)

    
        print(bcolors.SUCCESS, "End explainability for fold {f}".format(f=f), bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                      Test and Evaluation of the Best Model                                                                        #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


    # Print the best model and its weighted F1 score
    print(bcolors.PROC,"Best model fold :", best_model_fold,bcolors.ENDC)
    print(bcolors.PROC,f"Best Weighted {args.eval_metric} score:", best_eval_metric, bcolors.ENDC)

    if args.csv is not None:
        csv_test = args.csv.replace(ext, '_test.csv')
    else:
        csv_test = args.csv_test
    saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=best_model_fold))

    print(bcolors.INFO, f"Start testing of the best model (fold {best_model_fold})", bcolors.ENDC)
    best_model_path = get_best_checkpoint(saxi_train_args_out)
    saxi_predict_args = get_argparse_dict(saxi_predict.get_argparse())

    saxi_predict_args['csv'] = csv_test
    saxi_predict_args['model'] = best_model_path
    saxi_predict_args['surf_column'] = args.surf_column
    saxi_predict_args['class_column'] = args.class_column
    saxi_predict_args['mount_point'] = args.mount_point
    saxi_predict_args['nn'] = args.nn
    saxi_predict_args['crown_segmentation'] = args.crown_segmentation
    saxi_predict_args['path_ico_right'] = args.path_ico_right
    saxi_predict_args['path_ico_left'] = args.path_ico_left
    saxi_predict_args['fdi'] = args.fdi
    saxi_predict_args['fs_path'] = args.fs_path
    saxi_predict_args['device'] = 'cuda:0'
    saxi_predict_args['out'] = os.path.join(args.out, f'best_test_fold{best_model_fold}')

    saxi_predict_args = Namespace(**saxi_predict_args)
    fname = os.path.basename(csv_test)
    out_prediction = os.path.join(saxi_predict_args.out, os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

    if not os.path.exists(out_prediction):
        saxi_predict.main(saxi_predict_args)

    print(bcolors.SUCCESS, "End testing of the best model", bcolors.ENDC)

    ############# Run the evaluation for the prediction #########################

    print(bcolors.INFO, "Start evaluation of the best model", bcolors.ENDC)
    saxi_eval_args = get_argparse_dict(saxi_eval.get_argparse())

    saxi_eval_args['csv'] = out_prediction
    saxi_eval_args['class_column'] = args.class_column
    saxi_eval_args['nn'] = args.nn
    saxi_eval_args['eval_metric'] = args.eval_metric
    saxi_eval_args['mount_point'] = args.mount_point
    saxi_eval_args = Namespace(**saxi_eval_args)

    saxi_eval.main(saxi_eval_args)

    print(bcolors.SUCCESS, "End evaluation of the best model".format(f=f), bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                       Main                                                                                        #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def cml():
    # Command line interface
    parser = argparse.ArgumentParser(description='Automatically train and evaluate a N fold cross-validation model for Shape Analysis Explainability and Interpretability')
    # Arguments used for split the data into the different folds
    split_group = parser.add_argument_group('Split')
    split_group.add_argument('--csv', type=str, help='CSV with columns surf,class', default=None)
    split_group.add_argument('--csv_train', type=str, help='CSV with column surf', default=None)
    split_group.add_argument('--csv_test', type=str, help='CSV with column surf', default=None)
    split_group.add_argument('--folds', type=positive_int, help='Number of folds', default=5)
    split_group.add_argument('--valid_split', type=positive_float, help='Split float [0-1]', default=0.2)
    split_group.add_argument('--group_by', type=str, help='GroupBy criteria in the CSV. For example, SubjectID in case the same subjects has multiple timepoints/data points and the subject must belong to the same data split', default=None)

    # Arguments used for training
    train_group = parser.add_argument_group('Train')
    train_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification, SaxiIcoClassification_fs, SaxiRing, SaxiRingClassification', required=True, choices=['SaxiClassification', 'SaxiRegression', 'SaxiSegmentation', 'SaxiIcoClassification', 'SaxiIcoClassification_fs', 'SaxiRing', 'SaxiRingClassification', 'SaxiRingMT'])
    train_group.add_argument('--model', type=str, help='Model to continue training', default= None)
    train_group.add_argument('--train_sphere_samples', type=int, help='Number of samples for the training sphere', default=10000)
    train_group.add_argument('--surf_column', type=str, help='Surface column name', default="surf")
    train_group.add_argument('--class_column', type=str, help='Class column name', default="class")
    train_group.add_argument('--scale_factor', type=float, help='Scale factor for the shapes', default=1.0)
    train_group.add_argument('--column_scale_factor', type=str, help='Specify the name if there already is a column with scale factor in the input file', default='surf_scale')
    train_group.add_argument('--profiler', type=str, help='Profiler', default=None)
    train_group.add_argument('--compute_scale_factor', help='Compute a global scale factor for all shapes in the population.', type=int, default=0)
    train_group.add_argument('--compute_features', help='Compute features for the shapes in the population.', type=int, default=0)
    train_group.add_argument('--mount_point', type=str, help='Dataset mount directory', default="./")
    train_group.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    train_group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
    train_group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
    train_group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
    train_group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.35)
    train_group.add_argument('--image_size', type=int, help='Image resolution size', default=256)  
    train_group.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    train_group.add_argument('--epochs', type=int, help='Max number of epochs', default=200)   
    train_group.add_argument('--batch_size', type=int, help='Batch size', default=3)    
    train_group.add_argument('--patience', type=int, help='Patience for early stopping', default=30)
    train_group.add_argument('--log_every_n_steps', type=int, help='Log every n steps', default=10)    
    train_group.add_argument('--tb_dir', type=str, help='Tensorboard output dir', default=None)
    train_group.add_argument('--tb_name', type=str, help='Tensorboard experiment name', default="tensorboard")
    train_group.add_argument('--neptune_project', type=str, help='Neptune project', default=None)
    train_group.add_argument('--neptune_tags', type=str, help='Neptune tags', default=None)
    train_group.add_argument('--path_ico_right', type=str, help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)', default='./3DObject/sphere_f327680_v163842.vtk')
    train_group.add_argument('--path_ico_left', type=str, help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)', default='./3DObject/sphere_f327680_v163842.vtk',)
    train_group.add_argument('--layer', type=str, help="Layer, choose between 'Att','IcoConv2D','IcoConv1D','IcoLinear' (default: IcoConv2D)", default='IcoConv2D')
    train_group.add_argument('--fs_path', type=str, help='Path to freesurfer folder', default=None)
    train_group.add_argument('--num_images', type=int, help='Number of images to use for the training', default=12)
    train_group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=1)

    # Arguments used for prediction
    pred_group = parser.add_argument_group('Prediction group')
    pred_group.add_argument('--crown_segmentation', type=bool, help='Isolation of each different tooth in a specific vtk file', default=False)
    pred_group.add_argument('--fdi', type=int, help='numbering system. 0: universal numbering; 1: FDI world dental Federation notation', default=0)

    # Arguments used for testing
    eval_group = parser.add_argument_group('Test group')
    eval_group.add_argument('--csv_true_column', type=str, help='Which column to do the stats on', default="class")
    eval_group.add_argument('--csv_tag_column', type=str, help='Which column has the actual names', default=None)
    eval_group.add_argument('--csv_prediction_column', type=str, help='csv true class', default="pred")
    eval_group.add_argument('--eval_metric', type=str, help='Score you want to choose for picking the best model : F1 or AUC', default='F1', choices=['F1', 'AUC'])

    # Arguments used for explainability
    explain_group = parser.add_argument_group('Explainability group')
    explain_group.add_argument('--target_layer', type=str, help='Target layer for explainability', default='layer4')
    explain_group.add_argument('--fps', type=int, help='Frames per second', default=24) 

    # Arguments used for evaluation
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--out', type=str, help='Output', default="./")

    args = parser.parse_args()
    
    if not ((args.csv is not None) ^ (args.csv_train is not None and args.csv_test is not None)):
        parser.error('Either --csv or both --csv_train and --csv_test must be provided, but not --csv, --csv_train and --csv_test.')


    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}

    main(args, arg_groups)


if __name__ == '__main__':
    cml()