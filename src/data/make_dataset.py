# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import os 
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from src.features import build_features, file_reader
from src.models import train_model
from src.visualization import visualize
import config


def main():
    
    calories = pd.read_csv('/Users/mesude/cookiecutter-data-science/fmendes/data/raw/calories.csv')
    exercies = pd.read_csv('/Users/mesude/cookiecutter-data-science/fmendes/data/raw/exercise.csv')
    df = pd.concat([exercies,calories["Calories"]], axis=1)
    print(df.head())
    return df


if __name__ == '__main__':
    
    df = main()
    data1 = build_features.encode(df)
    file_reader.data_save(config.data_processed_path, 'encode', data1)
    data2= build_features.scale(data1[config.input_columns])
    file_reader.data_save(config.data_processed_path,'scale_data', data2 )
    #ann_model, X_train, X_test, y_train, y_test = train_model.create_model(data2, data1[config.output_columns], True)
    #ann_compiled = train_model.compile_model(ann_model, metrics=['mean_squared_error','mean_absolute_error'])
    #file_reader.model_save(config.model_path, 'ann_model', ann_compiled)
    #history = train_model.fit_model(ann_compiled, X_train, y_train, 5)
    #new_model = file_reader.load_models(config.model_path, 'ann_model')
    #train_model.eval(new_model, y_test, X_test)
#    visualize.train_val_loss(history)
#    visualize.mae_loss_epoch(history)
#    visualize.acc_loss(history)
#    visualize.residual_plot(ann_compiled, X_test, y_test)
    build_features.data_fold()

