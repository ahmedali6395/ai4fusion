import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

###

def cropData(dataset, config):
# {
    return (dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]).drop(columns=config["dataloader"]["ignore_columns"]);
# }

def getDatasetX(dataset, config):
# {
    return dataset.drop(columns=config["dataloader"]["label_column"]);
# }

def getDatasetY(dataset, config):
# {
    return dataset[config["dataloader"]["label_column"]].values;
# }

def createLoaders(config, dataset, regression=False):
# {
    dataset = dataset.replace(np.inf, np.nan);
    dataset = dataset.replace(-np.inf, np.nan);

    dataset = cropData(dataset, config).dropna();

    datasetX = getDatasetX(dataset, config);
    datasetY = getDatasetY(dataset, config);

    # if useScale
    if config["dataloader"]["scale"]:
    # {
        # Scale data
        featureScaler = MinMaxScaler(tuple(config["dataloader"]["feature_scale"]))
        featureScaler.fit(datasetX);
        datasetX = featureScaler.transform(datasetX.astype('float32'));
        datasetX = torch.from_numpy(datasetX);

        if regression:
        # {
            labelScaler = torch.from_numpy(MinMaxScaler(tuple(config["dataloader"]["feature_scale"])));
            labelScaler.fit(datasetY);
            datasetY = featureScaler.transform(datasetY.astype('float32'));
            datasetY = torch.from_numpy(datasetY);
        # }
        else:
        # {
            datasetY = torch.from_numpy(datasetY);
        # }
    # }

    # Tensor Dataset
    
    print(f"the shape of datasetX is {datasetX.shape}")
    print(f"the shape of datasetY is {datasetY.shape}")
    dataset = torch.utils.data.TensorDataset(datasetX, datasetY);

    # Random split
    trainingDataset, testingDataset, validationDataset = torch.utils.data.random_split(dataset, list(config["dataloader"]["data_split"]));

    # Plot for the funzies
    if config["plot"]:
    # {
        # TODO
        torch.histogram(datasetX, bins=10);
    # }

    training_loader = torch.utils.data.DataLoader(
        trainingDataset,
        config["dataloader"]["batch_sizes"]["training"]
    );

    testing_loader = torch.utils.data.DataLoader(
        testingDataset,
        config["dataloader"]["batch_sizes"]["testing"]
    );

    validation_loader = torch.utils.data.DataLoader(
        validationDataset,
        config["dataloader"]["batch_sizes"]["validation"]
    );

    return training_loader, testing_loader, validation_loader;
# }