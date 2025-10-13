import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import struct
import socket
import pprint
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("-F", "--file", action="store",
                    dest="F", help="Which file to process", type=str)
parser.add_argument("-S", "--scaling", action="store", dest="S",
                    help="Which scaling tactic to use", type=str, choices=["Z", "MinMax", "No"], default="No")
parsed_opts = parser.parse_args()


# All columns
col_names = np.array(["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
                      "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
                      "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                      "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
                      "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
                      "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
                      "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
                      "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                      "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
                      "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
                      "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                      "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
                      "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                      "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
                      "Idle Std", "Idle Max", "Idle Min", "Label"])

# Read the raw data
dataframe = pd.read_csv(parsed_opts.F, names=col_names, skiprows=1)

# Drop flow id, because it is redundant
dataframe.drop(axis=1, columns='Flow ID', inplace=True)
col_names = col_names[1::]

# Source ip -> numeric
dataframe["Source IP"] = dataframe["Source IP"].apply(
    lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
)

# Destination ip -> numeric
dataframe["Destination IP"] = dataframe["Destination IP"].apply(
    lambda ip: struct.unpack("!I", socket.inet_aton(ip))[0]
)

# Time, split date off, translate time to seconds


def process_time(time):
    time = time.split(" ")[-1]
    h, m = time.split(":")
    return 3600*int(h)+60*int(m)


dataframe["Timestamp"] = dataframe["Timestamp"].apply(
    lambda time: process_time(time)
)

# Translate string attack types to numbers
attack_types = dataframe["Label"].unique()
_ = np.where(attack_types == "BENIGN")
attack_types = np.delete(attack_types, _)
attack_dict = {"BENIGN": 0}
for i in range(0, len(attack_types)):
    attack_dict[attack_types[i]] = 1

dataframe["Label"] = dataframe["Label"].apply(
    lambda x: attack_dict[x]
)

# Print the distribution of the labels
print(dataframe['Label'].value_counts())

# Neat printing
# with pd.option_context('display.max_rows', 1, 'display.max_columns', None):
#     print(dataframe)

# Fixing the Infinity values
dataframe = dataframe.replace("Infinity", np.nan)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
dataframe = imp_mean.fit_transform(dataframe.values)
dataframe = pd.DataFrame(dataframe, columns=col_names)

# Explicitly cast the dataframe, otherwise sklearn's scalers will do that and throw a warning
dataframe = dataframe.astype(dtype='float64')

# Looking for irregularities in the processed data
# print(dataframe.dtypes)
# print(np.where(np.isnan(dataframe)))
# print(np.where(np.isinf(dataframe)))
# print(np.where(dataframe.values >= np.finfo(np.float64).max))
# print(np.where(dataframe.values <= np.finfo(np.float64).min))
# print(np.where(dataframe.values is str))

# # Preprocessing -> scaling
if parsed_opts.S == "Z":
    cols_to_scale = dataframe.columns.difference(['Label'])
    scaler = StandardScaler()
    dataframe[cols_to_scale] = scaler.fit_transform(
        dataframe[cols_to_scale])
    dataframe = pd.DataFrame(dataframe, columns=col_names)


elif parsed_opts.S == "MinMax":
    cols_to_scale = dataframe.columns.difference(['Label'])
    scaler = MinMaxScaler()
    dataframe[cols_to_scale] = scaler.fit_transform(
        dataframe[cols_to_scale])
    dataframe = pd.DataFrame(dataframe, columns=col_names)

elif parsed_opts.S == "No":
    None

with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
    print(dataframe)


def removeIP(dataframe, col_names):
    # Store the source and destination ip locations in the dataframe
    #
    print("Dataframe shape before IP removal ",
          dataframe.shape, " col_names length ", len(col_names))
    df = dataframe.drop(axis=1, columns='Source IP')
    df = df.drop(axis=1, columns='Destination IP')
    col_names = np.setdiff1d(col_names, np.array(
        ['Source IP', 'Destination IP']))
    print("Dataframe shape after IP removal ", df.shape,
          " col_names length ", len(col_names))
    return df


dataframe = removeIP(dataframe, col_names)
dataframe = dataframe.sample(frac=.0001)

for col in dataframe.columns:
    if dataframe[col].nunique() != 1:
        dataframe[col].plot(kind='kde', subplots='True',
                            figsize=(18, 10), title='Distributions')
    else:
        print(col)

plt.show()
