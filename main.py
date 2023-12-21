from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

import joblib


app = FastAPI()
class Data_transfer(BaseModel):
    data_sample: str

model = joblib.load('xgb_optuna_model.joblib')




# root definiton like / is not neseccary
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/model/predict")
async def predict(data: Data_transfer):
    data = pd.read_json(data.data_sample, orient='split')
    model_prediction = pd.DataFrame(np.round(model.predict(data)))
    model_prediction = model_prediction.to_json()
    return model_prediction

#%%
