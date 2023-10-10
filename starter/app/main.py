'''
Code for deploying machine learning model as an API
'''

from pydantic import BaseModel, Field
from fastapi import FastAPI
from os.path import dirname
from os.path import abspath
import sys

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

import joblib
import src.data as dt
import src.model as md
import pandas as pd

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

class Census(BaseModel):
    age: int = Field(example=28)
    workclass: str = Field(example='Private')
    education: str = Field(example='Prof-school')
    education_num: int = Field(example=15)
    marital_status: str = Field(example='Never-married')
    occupation: str = Field(example='Prof-specialty')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=55)
    native_country: str = Field(example='United-States')

@app.post("/predict")
async def predict(data: Census):
    model = joblib.load(f'{d}/training_artifacts/model.pkl')
    encoder = joblib.load(f'{d}/training_artifacts/encoder.pkl')

    df = pd.DataFrame([data.dict()])

    cat_features = df.select_dtypes(include=object).columns.tolist()

    X, _, _, _ = dt.process_data(df,
                        categorical_features=cat_features,
                        encoder=encoder,
                        training=False)

    pred = md.inference(model, X)

    # Get non-binarized prediction
    if pred == 1:
        return {"prediction": ">50K"}
    
    return {"prediction": "<=50K"}

