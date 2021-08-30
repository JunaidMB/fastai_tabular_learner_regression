from fastai.tabular.all import *
import pandas as pd
import numpy as np
import missingno
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Import data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

## Drop ID column
train_df = train_df.drop('id', axis = 1)
test_df = test_df.drop('id', axis = 1)

## Look at the distribution of the target variable - we have a regression problem
## {min: 0, max: 10, mean: 8}
## No Missing Values, features are approximately normalised
train_df.describe()

# Pre-process Data
## Make Tabular Pandas Object
cat_names = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
cont_names = ['cont0', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13']
procs = [Categorify, FillMissing] 
y_names = "target"
block_y = RegressionBlock()
splits = RandomSplitter()(range_of(train_df))
y_range = torch.tensor([0, 15]) # Restrict the target variable to be between 0 and 15

to = TabularPandas(train_df, procs = procs, cat_names = cat_names, cont_names = cont_names, y_names = y_names, y_block = block_y, splits = splits)

# Make DataLoaders with batch size 4096 (must be a multiple of 8)
dls = to.dataloaders(bs = 4096)

# Setup Tabular Learner
## Metrics to track: RMSE
## 3 network layers with 250, 750 and 100 neurons each
learn = tabular_learner(dls, layers = [250, 750, 100], y_range = y_range, metrics = rmse, loss_func = MSELossFlat())

# Find learning rate
learn.lr_find()

# Fit model for 10 epochs with weight decay of 0.2 
learn.fit_one_cycle(10, 1e-3, wd = 0.2)

# Obtain Validation RMSE
dl = learn.dls.test_dl(train_df.iloc[:])
learn.validate(dl=dl)

## Alternate method of validation RMSE
preds = learn.get_preds(dl=dl)
np.array(rmse(preds[0], preds[1]))

# Predict on test set
test_dl = learn.dls.test_dl(test_df)
test_preds = learn.get_preds(dl = test_dl)

# Random Forest
# Split training data into training and valid sets
X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_valid, y_valid = to.valid.xs, to.valid.ys.values.ravel()

tree = RandomForestRegressor(n_estimators = 500, verbose = 2, random_state=123, n_jobs = -1)
tree.fit(X_train, y_train)

# Extract Class Probabilities
forest_preds = tree.predict(X_valid)

# Ensemble Neural Network and Random Forest Predictions
avgs = (preds[0] + forest_preds)/2

# Tabular Learner RMSE
np.array(rmse(preds[0], preds[1]))

# Random Forest RMSE
rmse(tensor(forest_preds), tensor(y_valid))

# Ensemble RMSE
rmse(tensor(avgs), tensor(y_valid))

