import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--algorithm', action='store', dest='A', help='Which algorithm(s) to use', type=str,
                        choices=['knn', 'ncentroid', 'dtree', 'linsvc', 'rbfsvc', 'rforest', 'ada', 'bag', 'binlr', 'qda', 'lda', 'xgboost', 'gradboost', 'extratree'])
    parser.add_argument('-S', '--scaling', action='store', dest='S',
                        help='Which scaling tactic to use', type=str, choices=['Z', 'MinMax', 'No'], default='No')
    parser.add_argument('-M', '--multiclass', action='store', dest='M',
                        help='Do multiclass prediction', type=bool, default=False)
    parser.add_argument('-D', '--day', action='store', dest='D',
                        help='Which day should be processed?', type=int, default=1)
    parser.add_argument('-O', '--optimization', action='store', dest='O',
                        help='Perform parameter optimization', type=bool, default=False)
    parser.add_argument('-R', '--reduction', action='store', dest='R',
                        help='Perform feature selection / reduction', type=str, default='')
    parser.add_argument('--datadir', action='store', dest='datadir',
                        help='The folder in which the CSV files are stored', type=str, default='data/CSV/')
    parser.add_argument('--resultdir', action='store', dest='resultdir',
                        help='The root folder for storing results', type=str, default='results/')
    parser.add_argument('--disk', action='store', dest='disk',
                        help='write results to disk', type=int, choices=[0, 1], default=1)
    parser.add_argument('--export', action='store', dest='export',
                        help='export the trained model to disk', type=int, choices=[0, 1], default=0)
    parser.add_argument('--trainpercent', action='store', dest='trainpercent',
                        help='which percentage of the data is used for training', type=float, default=0.33)
    parser.add_argument('--reduceby', action='store', dest='reduceby',
                        help='cut out how many features', type=int, choices=[0, 1, 2, 3, 4], default=0)
    return parser


def dataframe_printer(dataframe, rows=1, cols=None):
    with pd.option_context('display.max_rows', rows, 'display.max_columns', cols):
        print(dataframe)


def look_for_anomalies(dataframe):
    print(dataframe.dtypes)
    print('NaN', np.where(np.isnan(dataframe)))
    print('Inf', np.where(np.isinf(dataframe)))
    print('> max np.float64', np.where(
        dataframe.values >= np.finfo(np.float64).max))
    print('< min np.float64', np.where(
        dataframe.values <= np.finfo(np.float64).min))
    print('Strings in frame', np.where(dataframe.values is str))

# This function is an extra preparation step for use with decision trees
# The attacking machines are known and a split on IP would yield 100% accuracy
# This however is not realistic, because real-world data wouldn't have attacking traffic from a couple of known sources


def remove_ip(dataframe):
    # Store the source and destination ip locations in the dataframe
    print('Dataframe shape before IP removal ', dataframe.shape)
    df = dataframe.drop(axis=1, columns='source')
    df = df.drop(axis=1, columns='destination')
    print('Dataframe shape after IP removal ', df.shape)
    return df


def split_label_off(dataframe):
    # Look for the label, split the label from the frame
    label_loc = dataframe.columns.get_loc('Label')
    array = dataframe.values
    Y = array[:, label_loc]
    X = np.delete(array, label_loc, 1)
    # Store the data
    data = {'X': X, 'Y': Y}
    print('Label column shape: ', Y.shape, 'Data columns shape: ', X.shape)
    return data

# small abstraction to do a train-test split on the data


def train_test_splitter(_input, train_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        _input['X'], _input['Y'], train_size=train_size, random_state=None, stratify=_input['Y'])
    print("X train samples:", len(X_train))
    print("Y train samples:", len(Y_train),
          np.bincount(Y_train.astype(np.int64)))
    print("X test samples:", len(X_test))
    print("Y test samples:", len(Y_test), np.bincount(Y_test.astype(np.int64)))
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test
    }
    return data
