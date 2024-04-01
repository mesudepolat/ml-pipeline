from pathlib import Path
import pandas as pd
import numpy as np
import os 
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import config
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from optuna.integration.mlflow import MLflowCallback
from src.models.train_search_models import train_search


study_name = "calories_NN_models"

mlflowcb = MLflowCallback(
    metric_name="my metric score", 
)


@mlflowcb.track_in_mlflow()
def objective(trial):

    params = {
        "n_neurons": trial.suggest_int("n_neurons", 64, 128 ),  
        "n_layers": trial.suggest_int("n_layers", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "regularizer": trial.suggest_loguniform("regularizer", 1e-10, 1e-2)
    }
    
    mean, std = train_search()
    
    mlflow.log_params(params)
    mlflow.log_metric("mean_mse", mean)
    mlflow.log_metric("std_mse", std)


    return np.mean(mean)


def create_study():

    study = optuna.create_study(study_name=study_name, direction = "minimize")
    study.optimize(objective, n_trials=100, callbacks=[mlflowcb])


create_study()