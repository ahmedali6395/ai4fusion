import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats

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
    """Create training, testing, and validation dataloaders

    Args:
        config (.json): _description_
        dataset (pd Dataset): pandas dataset
        regression (bool, optional): Regression mode. Defaults to False.

    Returns:
        DataLoader: training_loader
        DataLoader: testing_loader
        DataLoader: validation_loader
    """
# {
    dataset = dataset.replace(np.inf, np.nan);
    dataset = dataset.replace(-np.inf, np.nan);

    dataset = cropData(dataset, config).dropna();
    
    
    if config["gaussian_filter"]["filter_by_gaussian"]: 
        dataset = dataset[(np.abs(stats.zscore(dataset.select_dtypes(include=[np.number]))) < config["gaussian_filter"]["std_dev"]).all(axis=1)]
    
    datasetX = getDatasetX(dataset, config);
    datasetY = getDatasetY(dataset, config);
    

    if config["plot"]:
        datasetX.hist(figsize=(15,15),bins=50,density=True)



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
        
        to_plot = np.arange(0, (datasetX).shape[1]-1)
    
        fig, axes = plt.subplots(6, 8, figsize=(16, 12))
        axes = axes.ravel()  # flatten into 1D array
        
        for i in to_plot: 
            data_to_plot = datasetX[:, i].detach().cpu().numpy()
            axes[i].hist(data_to_plot, bins=50)
            axes[i].set_title(f"Sample {i}")
    
                

        
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