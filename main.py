import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from utils.createLoader import createLoaders


##############################


### Constants
DATA_PATH = "./data/raw_data/cmod.csv";
CONFIG_PATH = "class_ti_columnsdropped.json"

####


pd.set_option("display.max_columns", None);
config = None
### Config
with open(CONFIG_PATH, 'r', encoding = 'utf-8') as file: 
    config = json.load(file)

#################################

# Step 1: Get data and prune unnecessary columns
# Step 2: Plot Data (optional)
# Step 3: DataSet
# Step 4: Split Data
# Step 5: DataLoader

class_ti_train, class_ti_test, class_ti_val = createLoaders(config, dataset=pd.read_csv(DATA_PATH,sep=',',index_col=None));

# Step 6: Model

# Step 7: Config json

# Step 8: Trainer
