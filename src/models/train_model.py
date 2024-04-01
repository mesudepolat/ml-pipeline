from sklearn.model_selection import train_test_split
import tensorflow as tf
import os 
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append("../..")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.metrics import r2_score
import mlflow
from src.features import file_reader
import config



# def create_model(X, y, sum=None):

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=config.input_shape),
#         Dense(64, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(1, activation='linear')
# ])
#     if sum:
#         model.summary()
    
#     return model, X_train, X_test, y_train, y_test


# def compile_model(model, loss='mean_absolute_error', optimizer='adam', metrics=[]):
#     model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#     return model

# def fit_model(model, x, y, epochs, batch_size=64, verbose=1, val_split=0.2):
#     history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=val_split)
#     return history


# def eval(model,test, x_test):
#     y_pred = model.predict(x_test)
#     R2 = r2_score(test, y_pred)
#     print("R2 Score=",R2 )



def mlflow_train():

    data1 = file_reader.data_load(config.data_processed_path,'encode')
    data2 = file_reader.data_load(config.data_processed_path,'scale_data')

    X_train, X_test_val, y_train, y_test_val = train_test_split(data2, data1[config.output_columns], test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=.5, random_state=42)

    params = {
        "epochs": 5,
        "optimizer": 'adam',
        "metrics": ['mean_squared_error','mean_absolute_error'],
        "random_state": 42,
    }

    mlflow.set_tracking_uri("/Users/mesude/cookiecutter-data-science/fmendes/mlruns")

    experiment_name = "calories_burnt"
    run_name="calories_burnt_v0.1.0"
    try:
        exp = mlflow.create_experiment(experiment_name)
    except:
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp = exp.experiment_id
    
    with mlflow.start_run(experiment_id = exp, run_name=run_name):
        mlflow.log_params(params)
        
        mlflow.set_tag("Training Info", "Basic NN model for calories burnt data")

        mlflow.tensorflow.autolog()

        model = Sequential([
            Dense(64, activation='relu', input_shape=config.input_shape),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error','mean_absolute_error'])
        history = model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.2)
        y_pred = model.predict(X_test)

        R2 = r2_score(y_test, y_pred)

        mlflow.log_metric("r2score", R2)



if __name__ == "__main__":
    mlflow_train()
