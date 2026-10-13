from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

AI_DIR = Path(__file__).resolve().parent

dataset = pd.read_csv(AI_DIR / "data" / "Telco_customer_churn.csv")
dataset["Total Charges"] = pd.to_numeric(dataset["Total Charges"], errors="coerce").fillna(0)

x = dataset.drop("Churn Value", axis=1)
y = dataset["Churn Value"].to_numpy()

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

ordinal_cols = ["Contract"]
binary_cols = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]
one_hot_cols = ["Multiple Lines", "Internet Service", "Online Security", "Online Backup",
                "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Payment Method"]
num_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]

ct = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]), ordinal_cols),
        ("binary", OrdinalEncoder(), binary_cols),
        ("one hot", OneHotEncoder(handle_unknown="ignore"), one_hot_cols),
        ("Standard", StandardScaler(), num_cols),
    ],
    verbose_feature_names_out=False,
)
xtrain = ct.fit_transform(xtrain)

xtrain, ytrain = SMOTE(random_state=42).fit_resample(xtrain, ytrain)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42,
)
ada.fit(xtrain, ytrain)

joblib.dump(ada, AI_DIR / "model.pkl")
joblib.dump(ct, AI_DIR / "preprocessor.pkl")
print("Saved model.pkl and preprocessor.pkl in", AI_DIR)