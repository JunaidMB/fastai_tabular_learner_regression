from fastai.tabular.all import *
import pandas as pd
import numpy as np
import missingno
from bayes_opt import BayesianOptimization

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
## 2 model layers with 1000 and 500 neurons each
learn = tabular_learner(dls, layers = [1000, 500], y_range = y_range, metrics = rmse, loss_func = MSELossFlat())

# Find learning rate
learn.lr_find()

# Fit model for 5 epochs with weight decay of 0.2 
learn.fit_one_cycle(5, 5e-3, wd = 0.2)

# Obtain Validation RMSE
dl = learn.dls.test_dl(train_df.iloc[:])
learn.validate(dl=dl)

## Alternate method of validation RMSE
preds = learn.get_preds(dl=dl)
np.array(rmse(preds[0], preds[1]))

# Predict on test set
test_dl = learn.dls.test_dl(test_df)
test_preds = learn.get_preds(dl = test_dl)


def fit_with(lr:float, n_layers: float, layer_1: float, layer_2: float, layer_3: float, y_range = torch.tensor([0, 15])):
    
    if int(n_layers) == 2:
        layers = [int(layer_1), int(layer_2)]
    elif int(n_layers) == 3:
        layers = [int(layer_1), int(layer_2), int(layer_3)]
    else:
        layers = [int(layer_1)]
    
    learn = tabular_learner(dls, layers = layers, y_range = y_range, metrics = rmse, loss_func = MSELossFlat())
    
    with learn.no_bar() and learn.no_logging():
        learn.fit(5, lr = float(lr))
    
    rmse_result = float(learn.validate()[1])
    print(rmse_result)

    return rmse_result

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

# Declare hyperparameters, the tuple gives min and max values
hps = dict(lr = (1e-5, 1e-01), n_layers = (1, 3), layer_1 = (500, 750), layer_2 = (1000, 1500), layer_3 = (200, 2000))

# Build Optimiser
optim = BayesianOptimization(f = fit_with, pbounds = hps, verbose = 2, random_state = 1)

# Fit Optimiser
optim.maximize(n_iter = 5)

# Grab best results
print(optim.max)
