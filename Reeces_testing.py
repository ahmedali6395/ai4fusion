# Reece's testing file for data loading and preprocessing.
#The goal is to get data from the cmod tokamak and preprocess it for use in a machine learning model. This includes loading the data, pruning unnecessary columns, splitting the data into training, testing, and validation sets, and creating dataloaders for each set.

#Then we create a classification model to predict if there will be a disruption or not. In the data, disruptions are labeled as 1 and non-disruptions are labeled as 0. Our goal is to try and predict using a amchine learning model based on all the data we have available.

#We have a config file that specifies which columns to drop and other parameters for data loading and preprocessing. The createLoaders function takes care of all the data loading and preprocessing steps as per the config file.


import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from torch import nn
from sklearn.preprocessing import MinMaxScaler
from utils.createLoader import createLoaders


##############################


### Constants
DATA_PATH = "./data/cmod_clean_200ms.csv";
CONFIG_PATH = "class_ti_columnsdropped.json"

####


pd.set_option("display.max_columns", None);
config = None
### Config
with open(CONFIG_PATH, 'r', encoding = 'utf-8') as file: 
    config = json.load(file)


#CreateLoaders fetches data and prunes unnecessary columns as per the config file, then splits the data into train, test and validation sets and finally creates dataloaders for each set.

class_ti_train, class_ti_test, class_ti_val = createLoaders(config, dataset=pd.read_csv(DATA_PATH,sep=',',index_col=None));

# Step 6: Model
class MLP(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MLP, self).__init__()

        self.init_layer = nn.Linear(input_shape, 64)
        self.B0 = nn.BatchNorm1d(64)
        self.L1 = nn.Linear(64, 256)
        self.B1 = nn.BatchNorm1d(256)
        self.L2 = nn.Linear(256, 512)
        self.B2 = nn.BatchNorm1d(512)
        self.L3 = nn.Linear(512, 256)
        self.B3 = nn.BatchNorm1d(256)
        self.L4 = nn.Linear(256, 128)
        self.B4 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, output_shape)

        self.activation = nn.SELU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.B0(self.init_layer(x)))
        x = self.activation(self.B1(self.L1(x)))
        x = self.activation(self.B2(self.L2(x)))
        x = self.activation(self.B3(self.L3(x)))
        x = self.activation(self.B4(self.L4(x)))
        x = self.output_activation(self.output(x))

        return x


sample_x, sample_y = next(iter(class_ti_train))

input_shape = sample_x.shape[1]
output_shape = config["model"]["output_shape"]
classifier = MLP(input_shape=input_shape, output_shape=output_shape)

print(classifier)

sample_prediction = classifier(sample_x.float())
print("Sample input shape:", sample_x.shape)
print("Sample label shape:", sample_y.shape)
print("Sample prediction shape:", sample_prediction.shape)



# Step 7: Config json



# Step 8: Trainer
