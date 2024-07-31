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

from shapeaxi import compute_min_scale, split_train_eval, saxi_eval, saxi_predict, saxi_train, saxi_gradcam, saxi_nets
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


def compute_scale_factor(args, ext):
    print(bcolors.INFO, "Start computing the scale factor", bcolors.ENDC)
    output_path = os.path.join(args.out, os.path.basename(args.csv).replace(ext, '_scale_factor' + ext))
    if os.path.exists(output_path):
        df_scale = pd.read_csv(output_path)
        return np.min(df_scale['surf_scale'])
    else:
        compute_args = get_argparse_dict(compute_min_scale.get_argparse())
        compute_args.update({'csv': args.csv, 'mount_point': args.mount_point, 'surf_column': args.surf_column, 'out': output_path})
        compute_min_scale.main(Namespace(**compute_args))
        return np.min(pd.read_csv(output_path)['surf_scale'])


def normalize_features(args):
    print(bcolors.INFO, "Start normalizing features", bcolors.ENDC)
    feature_args = get_argparse_dict(compute_features.get_argparse())
    feature_args.update({'csv': args.csv, 'out': "/CMF/data/floda", 'fs_path': args.fs_path})
    compute_features.main(Namespace(**feature_args))
    print(bcolors.SUCCESS, "End normalizing features", bcolors.ENDC)


def get_scale_factor_from_csv(args):
    csv_path = os.path.join(args.mount_point, args.csv or args.csv_train)
    df = pd.read_csv(csv_path)
    return df[args.column_scale_factor].min() if args.column_scale_factor in df.columns else None


def check_create_folds(args, ext):
    for f in range(args.folds):
        csv_train = get_output_filename(args.csv if args.csv else args.csv_train, f'train_fold{f}_train_train.csv')
        if not os.path.exists(csv_train):
            return True
    return False


def get_fold_filenames(args, ext, f):
    if args.csv:
        return (
            args.csv.replace(ext, f'_train_fold{f}_train_train.csv'),
            args.csv.replace(ext, f'_train_fold{f}_train_test.csv'),
            args.csv.replace(ext, f'_train_fold{f}_test.csv')
        )
    return (
        args.csv_train.replace(ext, f'_fold{f}_train_train.csv'),
        args.csv_train.replace(ext, f'_fold{f}_train_test.csv'),
        args.csv_train.replace(ext, f'_fold{f}_test.csv')
    )


def update_args_with_groups(args_dict, arg_groups, groups):
    for group in groups:
        args_dict.update(arg_groups[group])


def get_output_filename(csv_path, suffix):
    return csv_path.replace(os.path.splitext(csv_path)[1], f'_{suffix}')


def create_and_split_folds(args, arg_groups, ext):
    if args.csv_train is None and args.csv_test is None:
        print(bcolors.INFO, "Start splitting data", bcolors.ENDC)
        split_args = get_argparse_dict(split_train_eval.get_argparse())
        split_args.update({'csv': args.csv, 'split': args.valid_split, 'group_by': args.group_by, 'mount_point': args.mount_point})
        split_train_eval.first_split_data(Namespace(**split_args))
        print(bcolors.SUCCESS, "End splitting data", bcolors.ENDC)

    print(f"{bcolors.INFO}Start creating {args.folds} folds{bcolors.ENDC}")
    split_args = get_argparse_dict(split_train_eval.get_argparse())
    split_args.update({k: v for k, v in arg_groups['Split'].items()})
    split_args.update({'csv': args.csv.replace(ext, '_train.csv') if args.csv else args.csv_train, 'mount_point': args.mount_point})
    split_train_eval.split_data_folds_test_train(Namespace(**split_args))

    for f in range(args.folds):
        csv_train = get_output_filename(args.csv if args.csv else args.csv_train, f'train_fold{f}_train.csv')
        split_args.update({'csv': csv_train})
        split_train_eval.split_data_folds_train_eval(Namespace(**split_args))
    print(f"{bcolors.SUCCESS}End of creating the {args.folds} folds {bcolors.ENDC}")


def train_test_eval_folds(args, arg_groups, scale_factor, ext):
    best_eval_metric = 0.0
    best_model_fold = ""
    for f in range(args.folds):
        print(bcolors.INFO, f"Start training for fold {f}", bcolors.ENDC)
        csv_train, csv_valid, csv_test = get_fold_filenames(args, ext, f)
        train_args = get_argparse_dict(saxi_train.get_argparse())
        update_args_with_groups(train_args, arg_groups, ['Train', args.nn])
        train_args.update({'csv_train': csv_train, 'csv_test': csv_test, 'csv_valid': csv_valid, 'scale_factor': scale_factor, 'out': os.path.join(args.out, 'train', f'fold{f}')})
        last_checkpoint = get_last_checkpoint(train_args['out'])

        if last_checkpoint is None:
            command = [sys.executable, '-m', 'shapeaxi.saxi_train']
            for k in train_args:
                if train_args[k]:
                    command.append('--' + str(k))
                    command.append(str(train_args[k]))
            subprocess.run(command)
        print(bcolors.SUCCESS, f"End training for fold {f}", bcolors.ENDC)

        print(bcolors.INFO, f"Start test for fold {f}", bcolors.ENDC)
        out_prediction = test_model(args, arg_groups, ext, f)
        print(bcolors.SUCCESS, f"End test for fold {f}", bcolors.ENDC)

        print(bcolors.INFO, f"Start evaluation for fold {f}", bcolors.ENDC)
        best_eval_metric, best_model_fold = evaluation_model(args, f, best_eval_metric, best_model_fold, out_prediction, ext)
        print(bcolors.SUCCESS, f"End evaluation for fold {f}", bcolors.ENDC)
    
    return best_eval_metric, best_model_fold


def test_model(args, arg_groups, ext, f):
    test_args = get_argparse_dict(saxi_predict.get_argparse())
    update_args_with_groups(test_args, arg_groups, [args.nn])
    test_args.update({
        'csv': get_fold_filenames(args, ext, f)[2],
        'model': get_best_checkpoint(os.path.join(args.out, 'train', f'fold{f}')),
        'surf_column': args.surf_column,
        'class_column': args.class_column,
        'mount_point': args.mount_point,
        'nn': args.nn,
        'out': os.path.join(args.out, 'test', f'fold{f}')
    })
    out_prediction = os.path.join(test_args['out'], os.path.basename(test_args['model']), os.path.basename(test_args['csv']).replace(ext, "_prediction" + ext))

    if not os.path.exists(out_prediction):
        saxi_predict.main(Namespace(**test_args))
    
    return out_prediction


def evaluation_model(args, f, best_eval_metric, best_model_fold, out_prediction, ext):
    eval_args = get_argparse_dict(saxi_eval.get_argparse())
    eval_result_path = out_prediction.replace("_prediction" + ext, "_eval" + ext)
    # if not os.path.exists(eval_result_path):
    eval_args.update({
        'csv': out_prediction,
        'class_column': args.class_column,
        'nn': args.nn,
        'eval_metric': args.eval_metric,
        'mount_point': args.mount_point
    })
    current_weighted_eval_metric = saxi_eval.main(Namespace(**eval_args))

    if current_weighted_eval_metric > best_eval_metric:
        best_eval_metric = current_weighted_eval_metric
        best_model_fold = f'{f}'
    
    with open(eval_result_path, 'w') as f_out:
        f_out.write(str(current_weighted_eval_metric))
    
    return best_eval_metric, best_model_fold


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   Aggregate                                                                                       #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def aggregate_predictions(args, ext):
    print(bcolors.INFO, "Start aggregate for all folds", bcolors.ENDC)
    out_prediction_agg, out_prediction_probs_agg = [], []

    for f in range(args.folds):
        out_prediction_fn = aggregate(ext, args, f, out_prediction_agg)
        out_prediction_agg.append(pd.read_csv(out_prediction_fn))
        if args.nn == "SaxiClassification":
            probs_fn = out_prediction_fn.replace("_prediction.csv", "_probs.pickle")
            out_prediction_probs_agg.append(pickle.load(open(probs_fn, 'rb')))

    out_prediction_agg = pd.concat(out_prediction_agg)
    output_agg_path = os.path.join(args.out, 'test', os.path.basename(args.csv.replace('.csv', '_train.csv')).replace(ext, "_aggregate_prediction" + ext))
    out_prediction_agg.to_csv(output_agg_path, index=False)

    if args.nn == "SaxiClassification":
        pickle.dump(np.concatenate(out_prediction_probs_agg), open(output_agg_path.replace("_prediction.csv", "_probs.pickle"), 'wb'))

    eval_aggregate_predictions(args, output_agg_path)
    print(bcolors.SUCCESS, "END aggregate prediction for ALL folds", bcolors.ENDC)


def eval_aggregate_predictions(args, output_agg_path):
    eval_args = get_argparse_dict(saxi_eval.get_argparse())
    eval_args.update({
        'csv': output_agg_path,
        'csv_true_column': args.class_column,
        'nn': args.nn,
        'eval_metric': args.eval_metric,
        'mount_point': args.mount_point
    })
    saxi_eval.main(Namespace(**eval_args))


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                Explainability                                                                                     #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def explainability_analysis(args, arg_groups, ext):
    for f in range(args.folds):
        print(bcolors.INFO, f"Start explainability for fold {f}", bcolors.ENDC)
        csv_test = get_fold_filenames(args, ext, f)[2]
        out_prediction = os.path.join(args.out, 'test', f'fold{f}', os.path.basename(get_best_checkpoint(os.path.join(args.out, 'train', f'fold{f}'))), os.path.basename(csv_test).replace(ext, "_prediction" + ext))

        gradcam_args = get_argparse_dict(saxi_gradcam.get_argparse())
        update_args_with_groups(gradcam_args, arg_groups, [args.nn])
        gradcam_args.update({
            'nn': args.nn,
            'csv_test': out_prediction,
            'surf_column': args.surf_column,
            'class_column': args.class_column,
            'num_workers': args.num_workers,
            'model': get_best_checkpoint(os.path.join(args.out, 'train', f'fold{f}')),
            'target_layer': args.target_layer,
            'mount_point': args.mount_point,
            'fps': args.fps,
        })

        if args.nn in ["SaxiClassification", "SaxiRegression", "SaxiRingClassification"]:
            gradcam_args['target_class'] = None if args.nn != "SaxiClassification" else pd.read_csv(os.path.join(args.mount_point, csv_test))[args.class_column].unique()
        elif args.nn == "SaxiRing":
            gradcam_args.update({
                'target_class': 1.0,
                'out': os.path.join(args.out, 'test', f'fold{f}', os.path.basename(gradcam_args['model']))
            })

        saxi_gradcam.main(Namespace(**gradcam_args))
        print(bcolors.SUCCESS, f"End explainability for fold {f}", bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                      Test and Evaluation of the Best Model                                                                        #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def evaluate_best_model(args, ext, arg_groups, best_model_fold, best_eval_metric):
    print(bcolors.PROC, "Best model fold :", best_model_fold, bcolors.ENDC)
    print(bcolors.PROC, f"Best Weighted {args.eval_metric} score:", best_eval_metric, bcolors.ENDC)
    csv_test = args.csv.replace(ext, '_test.csv') if args.csv else args.csv_test
    best_model_path = os.path.join(args.out, 'train', f'fold{best_model_fold}', get_best_checkpoint(os.path.join(args.out, 'train', f'fold{best_model_fold}')))

    print(bcolors.INFO, f"Start testing of the best model (fold {best_model_fold})", bcolors.ENDC)
    test_args = get_argparse_dict(saxi_predict.get_argparse())
    update_args_with_groups(test_args, arg_groups, [args.nn])
    test_args.update({
        'csv': csv_test,
        'model': best_model_path,
        'surf_column': args.surf_column,
        'class_column': args.class_column,
        'mount_point': args.mount_point,
        'nn': args.nn,
        'out': os.path.join(args.out, f'best_test_fold{best_model_fold}')
    })
    out_prediction = os.path.join(test_args['out'], os.path.basename(best_model_path), os.path.basename(csv_test).replace(ext, "_prediction" + ext))

    if not os.path.exists(out_prediction):
        saxi_predict.main(Namespace(**test_args))
    print(bcolors.SUCCESS, "End testing of the best model", bcolors.ENDC)

    print(bcolors.INFO, "Start evaluation of the best model", bcolors.ENDC)
    eval_args = get_argparse_dict(saxi_eval.get_argparse())
    eval_args.update({
        'csv': out_prediction,
        'class_column': args.class_column,
        'nn': args.nn,
        'eval_metric': args.eval_metric,
        'mount_point': args.mount_point
    })
    saxi_eval.main(Namespace(**eval_args))
    print(bcolors.SUCCESS, "End evaluation of the best model", bcolors.ENDC)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                       Main                                                                                        #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def cml():
    '''
    Command line interface for the saxi_folds.py script

    Args : 
        None
    '''
    # Command line interface
    parser = argparse.ArgumentParser(description='Automatically train and evaluate a N fold cross-validation model for Shape Analysis Explainability and Interpretability')
    # Arguments used for split the data into the different folds
    scale_group = parser.add_argument_group('Scale')
    scale_group.add_argument('--column_scale_factor', type=str, help='Specify the name if there already is a column with scale factor in the input file', default='surf_scale')

    split_group = parser.add_argument_group('Split')
    split_group.add_argument('--csv', type=str, help='CSV with columns surf,class', default=None)
    split_group.add_argument('--csv_train', type=str, help='CSV with column surf', default=None)
    split_group.add_argument('--csv_test', type=str, help='CSV with column surf', default=None)
    split_group.add_argument('--folds', type=positive_int, help='Number of folds', default=5)
    split_group.add_argument('--valid_split', type=positive_float, help='Split float [0-1]', default=0.2)
    split_group.add_argument('--group_by', type=str, help='GroupBy criteria in the CSV. For example, SubjectID in case the same subjects has multiple timepoints/data points and the subject must belong to the same data split', default=None)

    # Arguments used for training
    train_group = parser.add_argument_group('Train')
    train_group.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification, SaxiIcoClassification_fs, SaxiRing, SaxiRingClassification', required=True, choices=['SaxiClassification', 'SaxiRegression', 'SaxiSegmentation', 'SaxiIcoClassification', 'SaxiIcoClassification_fs', 'SaxiRing', 'SaxiRingClassification', 'SaxiRingMT', 'SaxiMHA', 'SaxiMHAClassification'])
    train_group.add_argument('--epochs', type=int, help='Max number of epochs', default=200)   
    train_group.add_argument('--model', type=str, help='Model to continue training', default= None)
    train_group.add_argument('--surf_column', type=str, help='Surface column name', default="surf")
    train_group.add_argument('--class_column', type=str, help='Class column name', default="class")
    train_group.add_argument('--scale_factor', type=float, help='Scale factor for the shapes', default=1.0)
    train_group.add_argument('--profiler', type=str, help='Profiler', default=None)
    train_group.add_argument('--compute_scale_factor', help='Compute a global scale factor for all shapes in the population.', type=int, default=0)
    train_group.add_argument('--compute_features', help='Compute features for the shapes in the population.', type=int, default=0)
    train_group.add_argument('--mount_point', type=str, help='Dataset mount directory', default="./")
    train_group.add_argument('--num_workers', type=int, help='Number of workers for loading', default=4)
    train_group.add_argument('--batch_size', type=int, help='Batch size', default=3)    
    train_group.add_argument('--patience', type=int, help='Patience for early stopping', default=30)
    train_group.add_argument('--freesurfer', help='Use freesurfer data', type=int, default=0)


    # Arguments used for prediction
    pred_group = parser.add_argument_group('Prediction group')
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

    ##Logger
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', type=int, help='Log every n steps', default=10)    
    logger_group.add_argument('--tb_dir', type=str, help='Tensorboard output dir', default=None)
    logger_group.add_argument('--tb_name', type=str, help='Tensorboard experiment name', default="tensorboard")
    logger_group.add_argument('--neptune_project', type=str, help='Neptune project', default=None)
    logger_group.add_argument('--neptune_tags', type=str, help='Neptune tags', default=None)
    logger_group.add_argument('--neptune_token', type=str, help='Neptune token', default=None)
    logger_group.add_argument('--num_images', type=int, help='Number of images to log', default=12)

    initial_args, unknownargs = parser.parse_known_args()
    model_args = getattr(saxi_nets, initial_args.nn)
    model_args.add_model_specific_args(parser)

    args = parser.parse_args()
    
    if not ((args.csv is not None) ^ (args.csv_train is not None and args.csv_test is not None)):
        parser.error('Either --csv or both --csv_train and --csv_test must be provided, but not --csv, --csv_train and --csv_test.')

    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}

    main(args, arg_groups)


def main(args, arg_groups):
    os.makedirs(args.out, exist_ok=True)
    ext = os.path.splitext(args.csv_train if args.csv_train else args.csv)[1]
    if ext not in ['.csv', '.parquet']:
        raise ValueError(f'Invalid file extension {ext}')
    
    scale_factor = None

    # Compute scale factor if needed
    if args.compute_scale_factor:
        scale_factor = compute_scale_factor(args, ext)
    elif args.compute_features:
        normalize_features(args)
    else:
        scale_factor = get_scale_factor_from_csv(args)

    # Split the data into the different folds
    create_folds = check_create_folds(args, ext)

    if create_folds:
        create_and_split_folds(args, arg_groups, ext)

    # Train and evaluate the model for each fold
    best_eval_metric, best_model_fold = train_test_eval_folds(args, arg_groups, scale_factor, ext)
    
    # Aggregate the predictions
    if args.nn in ["SaxiClassification", "SaxiRegression"]:
        aggregate_predictions(args, ext)
    
    # Compute the gradcam
    explainability_analysis(args, arg_groups, ext)

    # Evaluate the best model
    evaluate_best_model(args, ext, arg_groups, best_model_fold, best_eval_metric)


if __name__ == '__main__':
    cml()