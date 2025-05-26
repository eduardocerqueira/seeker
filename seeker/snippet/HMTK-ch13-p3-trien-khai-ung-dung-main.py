#date: 2025-05-26T16:56:11Z
#url: https://api.github.com/gists/cd68e6989b11cf48e8d8b5ed0d306e5c
#owner: https://api.github.com/users/cuongtv312

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("iris_model.joblib")

app = FastAPI(title="Iris API", version="1.0")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# API root
@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classification API"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: IrisInput):
    data = np.array([
        [input_data.sepal_length, input_data.sepal_width,
         input_data.petal_length, input_data.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
