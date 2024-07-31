import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                        Split the initial dataset into training, validation and test subsets for each fold                                         #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def check_input_file(args, fname):
    csv = True
    path = os.path.join(args.mount_point, fname)
    if(os.path.splitext(fname)[1] == ".csv"):
        df = pd.read_csv(path)
    elif(os.path.splitext(fname)[1] == ".parquet"):
        df = pd.read_parquet(path)
        csv = False
    else:
        print("File format not supported : .csv or .parquet")
        return
    return csv, df


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                       Split the data into training and test sets                                                                  #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def first_split_data(args):
    fname = args.csv
    csv, df = check_input_file(args, fname)

    if csv:
        # Split the data into train and test sets (80/20 by default)
        train_df, test_df = train_test_split(df, test_size=args.split)
        # Save the data into CSV files
        train_fn = fname.replace('.csv', '_train.csv')
        test_fn = fname.replace('.csv', '_test.csv')
        path_to_save_train = os.path.join(args.mount_point, train_fn)
        path_to_save_test = os.path.join(args.mount_point, test_fn)
        train_df.to_csv(path_to_save_train, index=False)
        test_df.to_csv(path_to_save_test, index=False)

    else:
        train_df, test_df = train_test_split(df, test_size=args.split)
        train_fn = fname.replace('.parquet', '_train.parquet')
        test_fn = fname.replace('.parquet', '_test.parquet')
        path_to_save_train = os.path.join(args.mount_point, train_fn)
        path_to_save_test = os.path.join(args.mount_point, test_fn)
        train_df.to_csv(path_to_save_train, index=False)
        test_df.to_csv(path_to_save_test, index=False)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                 Split the data into training and test sets for each fold                                                          #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


def split_data_folds_test_train(args):
    fname = args.csv
    csv, df = check_input_file(args, fname)
    split, df_split = check_split(args)

    if args.group_by:
        # If there is a column to group by, split the data based on the values of this column
        if args.csv_split:
            group_ids = df_split[args.group_by]
        
        else:
            group_ids = df[args.group_by].unique()
            np.random.shuffle(group_ids)

        samples = int(len(group_ids)*split)
        
        # Split the data into train and eval for each fold
        start_f = 0
        end_f = samples
        for i in range(args.folds):

            id_test = group_ids[start_f:end_f]

            df_train = df[~df[args.group_by].isin(id_test)]
            df_test = df[df[args.group_by].isin(id_test)]

            save_data_folds(args,df_train,df_test,fname,csv,i)

            start_f += samples
            end_f += samples

    else:
        # If there is no column to group by, split the data randomly
        group_ids = np.array(range(len(df.index)))
        samples = int(len(group_ids)*split)
        np.random.shuffle(group_ids)
        start_f = 0
        end_f = samples

        for i in range(args.folds):
            id_test = group_ids[start_f:end_f]
            df_train = df[~df.index.isin(id_test)]
            df_test = df.iloc[id_test]

            save_data_folds(args,df_train,df_test,fname,csv,i)

            start_f += samples
            end_f += samples


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                              Split the data into training and validation sets for each fold                                                       #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################



def split_data_folds_train_eval(args):
    fname = args.csv
    csv, df = check_input_file(args, fname)
    split, df_split = check_split(args)

    if args.group_by:
        # If there is a column to group by, split the data based on the values of this column
        if args.csv_split:
            group_ids = df_split[args.group_by]     
        else:
            group_ids = df[args.group_by].unique()
            np.random.shuffle(group_ids)

        samples = int(len(group_ids)*split)

        # Split the data into train and eval if there is just one fold
        id_test = group_ids[0:samples]
        id_train = group_ids[samples:]
        df_train = df[df[args.group_by].isin(id_train)]
        df_test = df[df[args.group_by].isin(id_test)]

        save_data(args,df_train,df_test,fname,csv)

    else:
        # If there is no column to group by, split the data randomly
        group_ids = np.array(range(len(df.index)))
        samples = int(len(group_ids)*split)
        np.random.shuffle(group_ids)
        id_test = group_ids[0:samples]
        id_train = group_ids[samples:]
        df_train = df.iloc[id_train]
        df_test = df.iloc[id_test]

        save_data(args,df_train,df_test,fname,csv)


def save_data(args,df_train,df_test,fname,csv):
    if csv:
        train_fn = fname.replace('.csv', '_train.csv')
        path_to_save_train = os.path.join(args.mount_point, train_fn)
        df_train.to_csv(path_to_save_train, index=False)
        eval_fn = fname.replace('.csv', '_test.csv')
        path_to_save_eval = os.path.join(args.mount_point, eval_fn)
        df_test.to_csv(path_to_save_eval, index=False)
    else:
        train_fn = fname.replace('.parquet', '_train.parquet')
        path_to_save_train = os.path.join(args.mount_point, train_fn)
        df_train.to_parquet(path_to_save_train, index=False)
        eval_fn = fname.replace('.parquet', '_test.parquet')
        path_to_save_eval = os.path.join(args.mount_point, eval_fn)
        df_test.to_parquet(path_to_save_eval, index=False)


def save_data_folds(args,df_train,df_test,fname,csv,i):
    if csv:
        train_fn = fname.replace('.csv', '_fold' + str(i) + '_train.csv')
        path_to_save_train = os.path.join(args.mount_point, train_fn)
        df_train.to_csv(path_to_save_train, index=False)
        eval_fn = fname.replace('.csv', '_fold' + str(i) + '_test.csv')
        path_to_save_eval = os.path.join(args.mount_point, eval_fn)
        df_test.to_csv(path_to_save_eval, index=False)
    else:
        train_fn = fname.replace('.parquet', '_fold' + str(i) + '_train.parquet')
        path_to_save_train = os.path.join(args.mount_point, train_fn)
        df_train.to_parquet(path_to_save_train, index=False)
        eval_fn = fname.replace('.parquet', '_fold' + str(i) + '_test.parquet')
        path_to_save_eval = os.path.join(args.mount_point, eval_fn)
        df_test.to_parquet(path_to_save_eval, index=False)


def check_split(args):
    split = args.split
    df_split = None
    if split == 0.0 and args.folds > 0:
        split = 1.0/args.folds
    if args.csv_split:
        df_split = pd.read_csv(args.csv_split)
    return split, df_split


def get_argparse():
    # The arguments are defined for the script
    parser = argparse.ArgumentParser(description='Splits data into train/eval', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv', type=str, help='CSV file', required=True)
    parser.add_argument('--mount_point', type=str, help='Mount point', default="./")
    parser.add_argument('--split', type=float, help='Split float [0-1]', default=0.2)
    parser.add_argument('--group_by', type=str, help='Group the rows by column', default=None)
    parser.add_argument('--folds', type=int, help='Number of folds to generate', default=0)
    parser.add_argument('--csv_split', type=str, help='Split the data using the ids from this dataframe', default=None)

    return parser
    

if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    first_split_data(args)
    split_data_folds_test_train(args)
    split_data_folds_train_eval(args)