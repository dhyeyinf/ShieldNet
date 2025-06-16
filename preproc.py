import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


def df_string_impute(dataframe, string, col_names):
    dataframe = dataframe.replace(string, np.nan)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataframe = imp_mean.fit_transform(dataframe.values)
    dataframe = pd.DataFrame(dataframe, columns=col_names)
    return dataframe


def cast_frame(dataframe):
    # Explicitly cast the dataframe, otherwise sklearn's scalers will do that and throw a warning
    dataframe = dataframe.astype(dtype='float64')
    return dataframe


def z_scaler(dataframe, col_names):
    cols_to_scale = dataframe.columns.difference(['Label'])
    scaler = StandardScaler()
    dataframe[cols_to_scale] = scaler.fit_transform(dataframe[cols_to_scale])
    dataframe = pd.DataFrame(dataframe, columns=col_names)
    return dataframe


def minmax_scaler(dataframe, col_names):
    cols_to_scale = dataframe.columns.difference(['Label'])
    scaler = MinMaxScaler()
    dataframe[cols_to_scale] = scaler.fit_transform(dataframe[cols_to_scale])
    dataframe = pd.DataFrame(dataframe, columns=col_names)
    return dataframe
