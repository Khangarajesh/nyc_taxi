from pydantic import BaseModel
from fastapi import FastAPI
from joblib import load
from src.features.feature_definitions import test_fuiture_build
import pandas as pd 

app = FastAPI()

class predictioninput(BaseModel):  #class predictioninput is inherited from BaseModel class
    vendor_id:float
    pickup_datetime:float
    passenger_count:float
    pickup_longitude:float
    pickup_latitude:float
    dropoff_longitude:float
    dropoff_latitude:float
    store_and_fwd_flag:float


model_path = 'models/model.joblib'
model = load(model_path)

@app.get("/")
def home():
    return "working fine"

@app.post('/predict')
def predict(input_data:predictioninput):   #predict function takes an argument input_data of type predictioninput
    features = {
        'vendor_id':input_data.vendor_id,
        'pickup_datetime':input_data.pickup_datetime,
        'passenger_count':input_data.passenger_count,
        'pickup_longitude':input_data.pickup_longitude,
        'pickup_latitude':input_data.pickup_latitude,
        'dropoff_longitude':input_data.dropoff_longitude,
        'dropoff_latitude':input_data.dropoff_latitude,
        'store_and_fwd_flag':input_data.store_and_fwd_flag
    }
    feature_df = pd.DataFrame(features, index = [0])
    features = test_fuiture_build(feature_df,'prod')
    
    prediction = model.predict(features)[0].item()
    return {"prediction" : prediction}

    
if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app, host = '127.0.0.1', port = 8000)
    
    #hey buddy.