import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools

import pickle 
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from shapeaxi import utils
from shapeaxi.colors import bcolors

# This file is used to evaluate the results of a classification or segmentation task (after the model has been trained and predictions have been made)

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    return cm


def choose_score(args,report):
    if args.eval_metric == 'F1':
      # Calculate F1 score
      weighted_f1_score = report["weighted avg"]["f1-score"]
      # Print or store F1 score
      print(bcolors.PROC, "Weighted F1 Score:", weighted_f1_score, bcolors.ENDC)
      return weighted_f1_score

    elif args.eval_metric == 'AUC':
      # Calculate AUC score
      weighted_auc_score = report["weighted avg"]["auc"]
      # Print or store AUC score
      print(bcolors.PROC, "Weighted AUC Score:", weighted_auc_score, bcolors.ENDC)
      return weighted_auc_score

    else:
      sys.exit("The value of score is not F1 or AUC. You must specify F1 or AUC.")


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                        Classification, Ico and Ico_fs Classification                                                              #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def SaxiClassification_eval(df, args, y_true_arr, y_pred_arr, path_to_csv):
    # For the classification, evaluating a classification model, generating classification metrics, creating confusion matrix visualizations
    # It also responsible for plotting ROC curves, aggregating and reporting classification metrics in a structured format
    if(args.csv_tag_column):
      class_names = df[[args.csv_tag_column, args.csv_prediction_column]].drop_duplicates()[args.csv_tag_column]
      class_names.sort()
    else:
      class_names = pd.unique(df[args.class_column])
      class_names.sort()
      print("Class names:", class_names)

    for idx, row in df.iterrows():
      y_true_arr.append(row[args.class_column])
      y_pred_arr.append(row[args.csv_prediction_column])

    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=1)

    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=args.figsize)

    plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)
    confusion_filename = os.path.splitext(path_to_csv)[0] + "_confusion.png"
    fig.savefig(confusion_filename)


    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=args.figsize)
    cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')

    norm_confusion_filename = os.path.splitext(path_to_csv)[0] + "_norm_confusion.png"
    fig2.savefig(norm_confusion_filename)


    probs_fn = args.csv.replace("_prediction.csv", "_probs.pickle")
    
    if os.path.exists(probs_fn) and os.path.splitext(probs_fn)[1] == ".pickle":
        
        with open(probs_fn, 'rb') as f:
            y_scores = pickle.load(f)
        
        y_onehot = pd.get_dummies(y_true_arr)

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            report.get(str(i), {})["auc"] = auc_score

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )

        roc_filename = os.path.splitext(path_to_csv)[0] + "_roc.png"

        fig.write_image(roc_filename)

        support = []
        auc = []
        for i in range(y_scores.shape[1]):
            support.append(report.get(str(i), {}).get("support", 0))
            auc.append(report.get(str(i), {}).get("auc", 0))

        support = np.array(support)
        auc = np.array(auc)

        if np.sum(support) != 0:
            report["weighted avg"]["auc"] = np.average(auc, weights=support)
        else:
            report["weighted avg"]["auc"] = 0

        df_report = pd.DataFrame(report).transpose()
        report_filename = os.path.splitext(path_to_csv)[0] + "_classification_report.csv"
        df_report.to_csv(report_filename)
        print(json.dumps(report, indent=4))
        print(f"Saved classification report to {report_filename}")

        score = choose_score(args, report)
        return score


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                  Segmentation                                                                                     #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def SaxiSegmentation_eval(df, args, y_true_arr, y_pred_arr, path_to_csv):
   # For the segmentation, evaluating a segmentation model, generating segmentation metrics, creating confusion matrix visualizations
    dice_arr = []

    for idx, row in df.iterrows():
      path_surf = os.path.join(args.mount_point, row["surf"])
      print("Reading:", path_surf)
      surf = utils.ReadSurf(path_surf)
      path_pred = os.path.join(args.mount_point, row["pred"])
      print("Reading:", path_pred)
      pred = utils.ReadSurf(path_pred)
      surf_features_np = vtk_to_numpy(surf.GetPointData().GetScalars(args.surf_id))
      pred_features_np = vtk_to_numpy(pred.GetPointData().GetScalars(args.pred_id))
      pred_features_np[pred_features_np==-1] = 1
      surf_features_np = np.reshape(surf_features_np, -1)
      pred_features_np = np.reshape(pred_features_np, -1)
      unique_v = np.unique(np.union1d(surf_features_np, pred_features_np))

      for v in range(1, 34):
        if v not in unique_v:
          surf_features_np = np.concatenate([surf_features_np, np.repeat([v], 95)])
          surf_features_np = np.concatenate([surf_features_np, np.repeat([v + 1], 5)])
          pred_features_np = np.concatenate([pred_features_np, np.repeat([v], 95)])
          pred_features_np = np.concatenate([pred_features_np, np.repeat([v], 5)])

      jaccard = jaccard_score(surf_features_np, pred_features_np, average=None)
      dice = 2.0*jaccard/(1.0 + jaccard)
      dice_arr.append(dice)
      y_true_arr.extend(surf_features_np)
      y_pred_arr.extend(pred_features_np)

    dice_arr = np.array(dice_arr)
    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    fig = plt.figure(figsize=(20, 20))
    plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32","33"], title="Confusion Matrix Segmentation")
    confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
    fig.savefig(confusion_filename)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print(cnf_matrix)
    FP = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2 * (PPV * TPR)/(PPV + TPR)
    print("True positive rate, sensitivity or recall:", TPR)
    print("True negative rate or specificity:", TNR)
    print("Positive predictive value or precision:", PPV)
    print("Negative predictive value:", NPV)
    print("False positive rate or fall out", FPR)
    print("False negative rate:", FNR)
    print("False discovery rate:", FDR)
    print("Overall accuracy:", ACC)
    print("F1 score:", F1)
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True)
    print(report)
    jaccard = jaccard_score(y_true_arr, y_pred_arr, average=None)
    print("jaccard score:", jaccard)
    l_dice = 2.0*jaccard/(1.0 + jaccard)
    print("dice:", 2.0*jaccard/(1.0 + jaccard))
    print ('dice ', l_dice)
    print(f'average dice : {sum(l_dice)/len(l_dice)}')
    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=(20, 20))
    cm = plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32","33"], normalize=True, title="Confusion Matrix Segmentation - normalized")
    norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
    fig2.savefig(norm_confusion_filename)
    fig3 = plt.figure() 
    # Creating plot
    print("Dice coefficient shape : ", dice_arr.shape)
    dice_del = np.delete(dice_arr,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,32],1)
    s = sns.violinplot(data=dice_del, scale='count',cut=0)
    plt.xticks([0,1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13], ["18", "19", "20", "21", "22", "23","24","25","26","27","28","29","30","31"]) # lower
    s.set_title('Dice coefficients: lower jaw')
    box_plot_filename = os.path.splitext(path_to_csv)[0] + "_violin_plot.png"
    ax = plt.gca()
    ax.set_ylim([0.75, 1.005])
    plt.show()
    fig3.savefig(box_plot_filename)


    # Extraction of the score (AUC or F1)
    score = choose_score(args,report)

    return score


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   Regression                                                                                      #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def SaxiRegression_eval(df, args, y_true_arr, y_pred_arr, path_to_csv):
   #Visualization of the distribution of absolute errors and prediction errors
    y_true_arr = [] 
    y_pred_arr = []

    y_true_arr = df[args.class_column]
    y_pred_arr = df[args.csv_prediction_column]
    df['abs'] = np.abs(y_true_arr - y_pred_arr)
    df['error'] = y_true_arr - y_pred_arr
    fig = px.violin(df, y="abs")
    abs_filename = os.path.splitext(path_to_csv)[0] + "_abs.png"
    fig.write_image(abs_filename)

    fig = px.violin(df, y="error")
    error_filename = os.path.splitext(path_to_csv)[0] + "_error.png"
    fig.write_image(error_filename)



def main(args):
    y_true_arr = [] 
    y_pred_arr = []

    path_to_csv = os.path.join(args.mount_point, args.csv)

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df = pd.read_csv(path_to_csv)
    else:        
        df = pd.read_parquet(path_to_csv)

    eval_functions = {
        "SaxiClassification": SaxiClassification_eval,
        "SaxiIcoClassification": SaxiClassification_eval,
        "SaxiIcoClassification_fs": SaxiClassification_eval,
        "SaxiRing": SaxiClassification_eval,
        "SaxiRingClassification": SaxiClassification_eval,
        "SaxiRingMT": SaxiClassification_eval,
        "SaxiMHA": SaxiClassification_eval,
        "SaxiSegmentation": SaxiSegmentation_eval,
        "SaxiRegression": SaxiRegression_eval,
    }
    
    if args.nn in eval_functions:
        score = eval_functions[args.nn](df, args, y_true_arr, y_pred_arr, path_to_csv)
    else:
        raise NotImplementedError(f"Neural network {args.nn} is not implemented")

    return score





def get_argparse():
  # Function to parse arguments for the evaluation script
  parser = argparse.ArgumentParser(description='Evaluate classification result', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--csv', type=str, help='CSV file', required=True)
  parser.add_argument('--class_column', type=str, help='Which column to do the stats on', default='class')
  parser.add_argument('--csv_tag_column', type=str, help='Which column has the actual names', default=None)
  parser.add_argument('--csv_prediction_column', type=str, help='csv true class', default='pred')
  parser.add_argument('--nn', type=str, help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification, SaxiRing, SaxiRingMT, SaxiRingClassification', required=True, choices=['SaxiClassification', 'SaxiRegression', 'SaxiSegmentation', 'SaxiIcoClassification', 'SaxiIcoClassification_fs', 'SaxiRing', 'SaxiRingMT', 'SaxiRingClassification', 'SaxiMHA', 'SaxiOctree'])
  parser.add_argument('--title', type=str, help='Title for the image', default='Confusion matrix')
  parser.add_argument('--figsize', type=str, nargs='+', help='Figure size', default=(6.4, 4.8))
  parser.add_argument('--surf_id', type=str, help='Name of array in point data for the labels', default='UniversalID')
  parser.add_argument('--pred_id', type=str, help='Name of array in point data for the predicted labels', default='PredictedID')
  parser.add_argument('--eval_metric', type=str, help='Score you want to choose for picking the best model : F1 or AUC', default='F1', choices=['F1', 'AUC'])
  parser.add_argument('--mount_point', type=str, help='Mount point for the data', default='./')

  return parser

if __name__ == '__main__':
  parser = get_argparse()
  initial_args, unknownargs = parser.parse_known_args()
  model_args = getattr(saxi_nets, initial_args.nn)
  model_args.add_model_specific_args(parser)
  args = parser.parse_args()
  main(args)





