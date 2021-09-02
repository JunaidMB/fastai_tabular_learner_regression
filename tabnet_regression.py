from fastai.tabular.all import *
from fast_tabnet.core import *
from fastai.basics import *
import pandas as pd
import numpy as np
import missingno
from bayes_opt import BayesianOptimization
from functools import lru_cache

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
splits = RandomSplitter(seed = 123)(range_of(train_df))
y_range = torch.tensor([0, 15]) # Restrict the target variable to be between 0 and 15

# Look at the cardinality of each of our categorical variables
train_df[cat_names].nunique()

to = TabularPandas(train_df, procs = procs, cat_names = cat_names, cont_names = cont_names, y_names = y_names, y_block = block_y, splits = splits)

# Make DataLoaders with batch size 4096 (must be a multiple of 8)
dls = to.dataloaders(bs = 4096)

# Get categorical embeddings
emb_szs = get_emb_sz(to)

# Setup Tabnet model
## Metrics to track: RMSE
model = TabNetModel(emb_szs, len(to.cont_names), dls.c, n_d = 8, n_a = 8, n_steps = 5, mask_type = 'entmax')
learn = Learner(dls, model, MSELossFlat(), opt_func = Adam, metrics = [rmse])

# Find learning rate
learn.lr_find()
learn.recorder.min_grad_lr

# Fit model for 10 epochs with weight decay of 0.2 
learn.fit_one_cycle(10, 1e-3)

# Obtain Validation RMSE
val_dl = TabDataLoader(to.valid, bs = 1024)
learn.validate(dl = val_dl)

## Alternate method of validation RMSE
preds = learn.get_preds(dl=val_dl)
np.array(rmse(preds[0], preds[1]))

# Predict on test set
test_dl = learn.dls.test_dl(test_df)
test_preds = learn.get_preds(dl = test_dl)
