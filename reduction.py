from sklearn.decomposition import PCA
import pandas as pd


def tree_important_drop(dataframe, number):
    # Removing columns identified as most discriminative for the tree classifiers
    to_remove = [
        'stopDateTime', 'startDateTime', 'sourceTCPFlagsDescription', 'sourcePort', 
        'destinationTCPFlagsDescription', 'totalSourcePackets', 'destinationPort', 
        'totalSourceBytes', 'source', 'direction', 'totalDestinationBytes', 
        'appName', 'destination', 'totalDestinationPackets', 'protocolName'
    ]
    dataframe = dataframe.drop(labels=to_remove[0:number], axis=1)
    return dataframe


def non_unique_drop(dataframe):
    # Removing columns without enough unique values
    to_remove = []
    for col in dataframe.columns:
        if len(dataframe[col].unique()) == 1:
            to_remove.append(col)
    dataframe = dataframe.drop(labels=to_remove, axis=1)
    return dataframe


def run_pca(dataframe):
    # pca = PCA(n_components='mle',svd_solver='full')
    pca = PCA()
    X = dataframe.values[:, :-1]
    print(X.shape)
    pca.fit(X)
    red_cols = ['pca_%i' % i for i in range(pca.n_components_)]
    labelcol = dataframe['Label']
    dataframe = pd.DataFrame(pca.transform(X),
                             columns=red_cols,
                             index=dataframe.index)
    dataframe = pd.concat([dataframe, labelcol], axis=1)
    return dataframe
