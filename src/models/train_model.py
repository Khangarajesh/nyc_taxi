import pathlib
from pathlib import Path
import sys
import yaml
import joblib
import pandas as pd 
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#def train_model(train_features, target,n_estimators,max_depth,seed):
#    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, random_state = seed)
#    print("model object created")
#    print(pd.DataFrame(train_features).head())
#    print(pd.Series(target))
#    #model.fit(train_features, target)
#    model.fit(train_features, target)
#    print("model fit")
#    return model



def find_the_best_model_with_params(X_train, y_train, X_test, y_test):
    
    
    ##CREATE HYPERPARAMETER SPACE 
    #finding the best hyperparameters for given model using hp.choice.
    # (you should dynamically add the best selected model from lazypredictor)
    hyperparameters = {
        "RandomForestRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
        "XGBRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "learning_rate": hp.uniform("learning_rate", 0.03, 0.3),
        },
    }
    
    
    ##CREATE A FUNCTION WHICH WILL TRAIN AND EVALUATE A MODEL ON GIVEN HYPERPARAMETR SPACE 
    def evaluate_model(hyperopt_params):  #hyperopt_params: here values will be taken from HYPERPARAMETR SPACE 
        #purpose of this step:
        #make a model fail proof first and # hyperopt supplies values as float but must be int
        params = hyperopt_params
        if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
        if 'min_child_weight' in params: params['min_child_weight'] = int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step'] = int(params['max_delta_step'])
        
        #train the model with corrrected parameters
        # (you should dynamically add the best selected model from lazypredictor)
        model = XGBRegressor(**params)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        #calculate model performance metric and track it with mlflow 
        model_rmse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric('RMSE', model_rmse)
        loss = model_rmse
        return {'loss':loss,'status':STATUS_OK}
    
    
    ##THIS STEP WILL GIVE THE BEST HYPERPARAMETERS USING GIVEN SPACE AND EVEAL_MODEL FUNCTION
    space = hyperparameters['XGBRegressor']
    with mlflow.start_run(run_name = 'XGBRegressor'):
        #argmin finds the model whoes loss function(i.e output of evaluate_model) is minimun and stores its Hyperparameters
        argmin = fmin(fn = evaluate_model,
                      space = space,
                      algo = tpe.suggest,
                      max_evals = 5,      #this means model will be evaluate maximum of 5 times based on different hyperprameters
                      trials = Trials(),  #trial keeps the track of hyperparameter and model metrics 
                      verbose = True)
    
    #THIS STEP WILL USE THE BEST FOUND HYPERPARAMETERS in above step AND USE IT TO TRAIN AND LOG THE MODEL
    run_ids = []
    with mlflow.start_run(run_name = 'XGB Best Model') as run:
        run_id = run.info.run_id
        run_name = run.data.tags['mlflow.runName']
        run_ids += [(run_id, run_name)]
        
        
        params = space_eval(space,argmin) #space_eval function takes the corresponding parameters from space with respect to argmin
        if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
        if 'min_child_weight' in params: params['min_child_weight'] = int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step'] = int(params['max_delta_step'])   
        mlflow.log_params(params)
        
        model =  XGBRegressor(**params) 
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        mlflow.log_metric('RMSE', np.sqrt(mean_squared_error(y_test, prediction)))
        mlflow.sklearn.log_model(model, 'model')
        
    return model
        
        
def save_model(model,output_path):
    joblib.dump(model, output_path + '/model.joblib')


def main():
    
    file_path = pathlib.Path(__file__)
    home_dir = file_path.parent.parent.parent
    param_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(param_file))['train_model']
    data_path = home_dir.as_posix() + '/data'
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents = True, exist_ok = True)
    print("path defined")
    
    target = 'trip_duration'
    train_features = pd.read_csv(data_path + '/processed/train.csv')
    x = train_features.drop(target, axis = 1)
    print(x.head())
    y = train_features[target]
    print(y.head())
    print("input and output data seperated")
    #train the model
    #trained_model = train_model(x,y,params['n_estimators'],params['max_depth'], params['seed'])
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
    trained_model = find_the_best_model_with_params(X_train, y_train, X_test, y_test)
    print("model trained")
    #save the model
    save_model(trained_model,output_path)
    print("model_saved")
    
    
if __name__ == "__main__":
    main()
    
    
#Explenation
#seperate the target from input data 
#train test split
#use find_the_best_model_with_params function for model training
#=> this function defines the hyperparametr space for each model in dictionary format
#=> evaluate_model=> does model evaluation for given hyperparameters
#=> fmin will minimize the out put of evaluate_model by using hyperparametrs present in hyperparametrs space
# and mlflow will keep the log of best model
# train and save the model


#Tip:
#you can use lazy predict for best model selection, once you got the best model then you can 
# perform hyperparameter tuning on top of it using hyper opt

