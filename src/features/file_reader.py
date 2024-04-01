import os
from keras.models import load_model
import pandas as pd


def data_save(path, name, data):
   data.to_csv(os.path.join(path,f'{name}.csv'), index = False)

def data_load(path, name):
   data = pd.read_csv(os.path.join(path, f'{name}.csv'))
   return data

def model_save(path, name, model):
   model.save(os.path.join(path,f'{name}.h5'))

def load_models(path,name):
   model = load_model(os.path.join(path, f'{name}.h5'))
   return model
