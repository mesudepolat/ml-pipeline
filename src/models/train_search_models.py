# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import os 
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from src.features import build_features, file_reader
import config
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping





def train_search():

    train = build_features.data_fold()

    splits = [0,1,2,3,4]    

    

    mse_list = list()
    for i in splits:
        train_i, val_i = build_features.data_train_val(train, i)
        #X_train, X_test_val, y_train, y_test_val = train_test_split(train_i, val_i, test_size=0.2, random_state=42)

        X_train = train_i[config.input_columns]
        y_train = train_i[config.output_columns] 
        X_val = val_i[config.input_columns]
        y_val = val_i[config.output_columns]

        mlflow.tensorflow.autolog()
        with mlflow.start_run(nested=True):
            
            mlflow.set_tag("Training Info", "Basic NN model for calories burnt data")

            

            model = Sequential([
                Dense(64, activation='relu', input_shape=config.input_shape),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','mean_absolute_error'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

            #model fit için önceden oluşturulmuş EarylStopping callbacki eklenecek
            history = model.fit(X_train, y_train, 
                                validation_data = (X_val, y_val), 
                                epochs=50, 
                                batch_size=64, 
                                verbose=1, 
                                validation_split=0.2,
                                callbacks=[early_stopping])
            
            y_pred = model.predict(X_val)
            error = mean_squared_error(y_val, y_pred)
            mse_list.append(error)
            mlflow.log_metric("mse", error)
    

    #return mean and std
    return np.mean(mse_list), np.std(mse_list)

        



train_search()