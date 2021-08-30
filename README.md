# FastAI Tabular Regression

This repository contains 3 python scripts where I've tried to implement the lessons in the walk with fastai lesson [here](https://www.youtube.com/watch?v=-aCtDIgbxMw&list=PLFDkaGxp5BXDvj3oHoKDgEcH73Aze-eET&index=10). The scripts are:

1. `tabular_learner.py`: Builds a tabular leaner for a regression problem with no parameter tuning
2. `ensemble.py`: Builds a tabular learner for a regression problem and combines predictions with a Random Forest model, both with no parameter tuning.
3. `tabuler_learner_bayesian_opt.py`: Attempts to use [Bayesian Optimisation](https://github.com/fmfn/BayesianOptimization) to tune hyperparameters with a tabular learner. This script is not optimal at this point and needs improvement. It is not clear if Bayesian Optimisation works the same in the lecture because the loss function needs to be minimised, not maximised.

The dataset used to train the model is the kaggle dataset associated with the 30 days challenge.