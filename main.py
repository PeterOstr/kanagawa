from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import f1_score, roc_auc_score

import joblib


app = FastAPI()
class Data_transfer(BaseModel):
    data_sample: str

class CombinedData(BaseModel):
    index: List
    gender: List
    age: List
    hypertension: List
    heart_disease: List
    smoking_history: List
    bmi: List
    HbA1c_level: List
    blood_glucose_level: List

model = joblib.load('xgb_optuna_model.joblib')




# root definiton like '/' is not neseccary
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/model/predict")
async def predict(data: CombinedData):
    data = pd.read_json(data.model_dump_json(), orient='list')
    data = data.set_index('index')
    model_prediction = pd.DataFrame(np.round(model.predict(data)))
    model_prediction = model_prediction.reset_index().to_json()
    return model_prediction

@app.post("/model/combined")
async def combined(data: CombinedData):
    predict_data_df = pd.read_json(data.predict_data.data_sample, orient='list')
    eval_data_df = pd.read_json(data.eval_data.data_sample, orient='list')

    if predict_data_df.shape[0] != eval_data_df.shape[0]:
        raise HTTPException(status_code=400, detail="Prediction and evaluation data have different lengths.")

    # Make predictions
    predictions = model.predict(predict_data_df)

    # Evaluate the model
    f1, roc_auc = evaluate_model(eval_data_df, target_df)

    return {"f1_score": f1, "roc_auc": roc_auc}

#%%
