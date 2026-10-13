from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ada = joblib.load( BASE_DIR / "ai" / "model.pkl")
ct = joblib.load(BASE_DIR / "ai" / "preprocessor.pkl" )

app = FastAPI()

Threshold = 0.35

YN = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]

class Customer(BaseModel):

    model_config = ConfigDict(populate_by_name = True)

    gender: Literal["Male", "Female"] = Field(alias = "Gender")
    senior : YN = Field(alias = "Senior Citizen")
    partner : YN = Field(alias = "Partner")
    dependents : YN = Field(alias = "Dependents")
    tenure : int = Field(alias = "Tenure Months", ge = 0, le = 120)
    phone : YN = Field(alias = "Phone Service")
    lines: Literal["Yes", "No", "No phone service"] = Field(alias="Multiple Lines")
    internet: Literal["DSL", "Fiber optic", "No"] = Field(alias="Internet Service")
    security: NoInternet = Field(alias="Online Security")
    backup: NoInternet = Field(alias="Online Backup")
    protection: NoInternet = Field(alias="Device Protection")
    support: NoInternet = Field(alias="Tech Support")
    tv: NoInternet = Field(alias="Streaming TV")
    movies: NoInternet = Field(alias="Streaming Movies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless: YN = Field(alias="Paperless Billing")
    payment: Literal["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"] = Field(alias="Payment Method")
    monthly: float = Field(alias="Monthly Charges", ge=0)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

@app.post("/predict")
async def predict( request : Request):

    data = await request.json()

    data["Total Charges"] = data["Tenure Months"] * data["Monthly Charges"]

    df = pd.DataFrame([data])
    x = ct.transform(df)
    proba = float(ada.predict_proba(x)[0, 1])
    return {"churn": "Yes" if proba >= Threshold else "No",
            "probability": round(proba, 3)}

