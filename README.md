# Customer Churn Prediction

## Overview

The goal of this project is to predict which customers are likely to stay with a business and which are likely to churn.

Every successful business needs continuous customer growth. Acquiring a new customer is far more expensive than retaining an existing one, so it is cheaper and more efficient to identify customers who plan to leave and offer them incentives to stay.

**Example:** Jio provides internet service and competes with Airtel and Vi. If a competitor's service or offers look better, customers may switch. By predicting which customers are likely to leave, Jio can offer them targeted deals, such as 15% off their monthly recharge, so they think twice before churning. This improves retention and company growth.

## Challenges

1. **Imbalanced data:** About 75% of customers stay and 25% churn, so a model can reach 75% accuracy just by predicting "no churn" every time.
2. **False positives:** If we predict a customer will churn but they would have stayed, we lose money on unnecessary discounts.
3. **False negatives:** If we predict a customer will stay but they churn, we lose the customer and the revenue, which is the costliest mistake.
4. **Limited data:** With few churn examples, the model may not learn enough patterns.

## Solutions

1. **SMOTE:** Used to generate realistic synthetic churn samples and balance the dataset.
2. **Multiple models:** Trained and compared several models using ROC-AUC, F1-score, and recall to select the best one.
3. **SHAP:** Used to identify the most important features driving predictions and make the model explainable.

## Project Structure

```
Churn/
├── ai/
│   ├── data/            # Churn dataset
│   ├── model.py         # Builds and compares multiple models
│   ├── train.py         # Trains the best model from model.py
│   ├── model.pkl        # Saved trained model
│   └── requirements.txt
├── backend/
│   ├── main.py          # FastAPI app with API endpoint that calls the AdaBoost model's predict method
│   └── requirements.txt
├── frontend/
│   └── index.html       # HTML, CSS and JS for the frontend page
└── venv/                # Virtual environment
```

## Setup

1. Create a virtual environment:
```powershell
   python -m venv venv
```
2. Activate it:
```powershell
   venv\Scripts\activate
```
3. Install dependencies:
```powershell
   pip install -r ai\requirements.txt
   pip install -r backend\requirements.txt
```

## Run the Server

```powershell
cd backend
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`


## Tech Stack

Python, scikit-learn, imbalanced-learn (SMOTE), SHAP, FastAPI, HTML/CSS/JavaScript