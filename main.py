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

########################### XGBOOST Model #####################################################
#from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#model = XGBClassifier(eval_metric="logloss", random_state=42)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

##############################################

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#load the dataset
df_xgb = pd.read_csv(DATA_PATH, sep=",")

df_xgb = df_xgb.replace([np.inf, -np.inf], np.nan) #clean the data
df_xgb = df_xgb.drop(columns=config["dataloader"]["ignore_columns"], errors="ignore") #drop the ignored columns
df_xgb = df_xgb.dropna() #drop the NaNs

#define label and features
label_col = config["dataloader"]["label_column"]
y = df_xgb[label_col]
X = df_xgb.drop(columns=[label_col])

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model = XGBClassifier(eval_metric="logloss",random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred) #confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

##############################################################################################

# Step 7: Config json

# Step 8: Trainer