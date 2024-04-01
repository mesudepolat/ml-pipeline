import os 
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from src.data import make_dataset
from src.features import file_reader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import config



def encode(data):
    data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
    return data

def scale(data):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    X = pd.DataFrame(X, columns=config.input_columns)
    return X



def data_fold():

    data1 = file_reader.data_load(config.data_processed_path, 'encode')
    data2 = file_reader.data_load(config.data_processed_path, 'scale_data')
    data2 = pd.concat([data2, data1['Calories']], axis=1)
    train, test = train_test_split(data2, test_size=0.2, random_state=42)
        
    train_size = len(train)
    print("train size", train_size)
    
    train['Fold'] = -1
    
    fold_size = train_size // 5
    for fold_number in range(5):
        start_index = fold_number * fold_size
        print("start index: ",start_index)
        print("fold size",fold_size)
        end_index = min((fold_number + 1) * fold_size, train_size)
        print("end index", end_index)
        train.iloc[start_index:end_index, train.columns.get_loc('Fold')] = fold_number
    

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    print(train[0:10])
    
    return train
    

def data_train_val(data, fold_number):
    
    train = data[data['Fold'] != fold_number]
    val = data[data['Fold'] == fold_number]    
    
    return train, val