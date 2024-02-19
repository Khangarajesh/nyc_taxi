import pandas as pd 
import numpy as np
import pathlib 
from feature_definitions import test_fuiture_build


def save_data(train_data, test_data, output_path):
    pathlib.Path(output_path).mkdir(parents = True,exist_ok = True)
    train_data.to_csv(output_path + '/train.csv', index = False)
    test_data.to_csv(output_path + '/test.csv', index = False)
    
if __name__ == '__main__':
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    output_path = home_dir.as_posix() + "/data/processed"
    
    train_path = home_dir.as_posix() + '/data/raw/train.csv'
    test_path = home_dir.as_posix() + '/data/raw/test.csv'
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    train_data = test_fuiture_build(train_data,'train')
    test_data = test_fuiture_build(test_data,'test')
    
    save_data(train_data,test_data,output_path)
    
    
    
#EXPENATIO
#Take the data from data/raw folder 
#use test_fuiture_build function to create extra features and 
# save the newly created features file in data/processsed folder...