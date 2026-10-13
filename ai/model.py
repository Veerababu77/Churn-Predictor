import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('ai/data/Telco_customer_churn.csv')

drop_columns = ['CustomerID', 'Count', 'Country', 'State', 'Lat Long', 'Churn Score', 'Churn Reason', 'City', 'Zip Code',
                'Latitude', 'Longitude', 'Churn Label', 'CLTV']
for col in drop_columns:
  dataset = dataset.drop(col, axis = 1)

dataset["Total Charges"] = pd.to_numeric(dataset["Total Charges"], errors="coerce")
totalcharges_missing = dataset["Total Charges"].isnull().sum()
if totalcharges_missing > 0:
  dataset.loc[dataset['Total Charges'].isnull(), 'Total Charges'] = 0

categorical_cols = dataset.select_dtypes(include = ['object', 'category']).columns.tolist()
numeric_cols = dataset.select_dtypes(include = ['number']).columns.tolist()

numeric_cols.remove('Churn Value')

x = dataset.drop('Churn Value', axis = 1)
y = dataset['Churn Value'].to_numpy()

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

from imblearn.over_sampling._smote.base import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ordinal_cols = ['Contract']
binary_cols = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]
one_hot_cols = ["Multiple Lines", "Internet Service", "Online Security", "Online Backup",
                "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Payment Method"]
num_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
ct = ColumnTransformer(
    transformers = [
        ("ordinal", OrdinalEncoder( categories = [["Month-to-month", "One year", "Two year"]]), ordinal_cols),
        ("binary", OrdinalEncoder(), binary_cols),
        ("one hot", OneHotEncoder(handle_unknown = 'ignore'), one_hot_cols),
        ('Standard', StandardScaler(), num_cols),
    ],
    verbose_feature_names_out = False,
)
xtrain = ct.fit_transform(xtrain)
xtest = ct.transform(xtest)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 42)
xtrain, ytrain = smote.fit_resample(xtrain, ytrain)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(xtrain, ytrain)
lr_predict = lr.predict(xtest)
from sklearn.metrics import f1_score, recall_score, r2_score, precision_score, confusion_matrix,classification_report
lr_report = classification_report(ytest, lr_predict, target_names=['No Churn', 'Churn'])
print(lr_report)
lr_f1 = f1_score(ytest, lr_predict)
print(lr_f1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(xtrain, ytrain)
knn_predict = knn.predict(xtest)
knn_report = classification_report(ytest, knn_predict, target_names=['No Churn', 'Churn'])
print(knn_report)
knn_f1 = f1_score(ytest, knn_predict)
print(knn_f1)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 42)
svc.fit(xtrain, ytrain)
svc_predict = svc.predict(xtest)
svc_report = classification_report(ytest, svc_predict, target_names=['No Churn', 'Churn'])
print(svc_report)
svc_f1 = f1_score(ytest, svc_predict)
print(svc_f1)

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(xtrain, ytrain)
naive_predict = naive.predict(xtest)
naive_report = classification_report(ytest, naive_predict, target_names=['No Churn', 'Churn'])
print(naive_report)
naive_f1 = f1_score(ytest, naive_predict)
print(naive_f1)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier( criterion = 'entropy', random_state = 42)
tree.fit(xtrain, ytrain)
tree_predict = tree.predict(xtest)
tree_report = classification_report(ytest, tree_predict, target_names=['No Churn', 'Churn'])
print(tree_report)
tree_f1 = f1_score(ytest, tree_predict)
print(tree_f1)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(xtrain, ytrain)
forest_predict = forest.predict(xtest)
forest_report = classification_report(ytest, forest_predict, target_names=['No Churn', 'Churn'])
print(forest_report)
forest_f1 = f1_score(ytest, forest_predict)
print(forest_f1)

from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(xtrain, ytrain)
xgb_predict = xgb.predict(xtest)
xgb_report = classification_report(ytest, xgb_predict, target_names=['No Churn', 'Churn'])
print(xgb_report)
xgb_f1 = f1_score(ytest, xgb_predict)
print(xgb_f1)

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)
lgbm.fit(
    xtrain, ytrain,
    eval_set=[(xtest, ytest)]
)
lgbm_predict = lgbm.predict(xtest)
lgbm_report = classification_report(ytest, lgbm_predict, target_names=['No Churn', 'Churn'])
print(lgbm_report)
lgbm_f1 = f1_score(ytest, lgbm_predict)
print(lgbm_f1)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(xtrain, ytrain)
ada_predict = ada.predict(xtest)
ada_report = classification_report(ytest, ada_predict, target_names=['No Churn', 'Churn'])
print(ada_report)
ada_f1 = f1_score(ytest, ada_predict)
print(ada_f1)

from sklearn.metrics import precision_recall_curve
probs = ada.predict_proba(xtest)[:, 1]
precision, recall, thresholds = precision_recall_curve(ytest, probs)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.title('Precision & Recall vs. Threshold')
plt.show()

import shap
feature_names = list(ct.get_feature_names_out())
xtrain_df = pd.DataFrame(xtrain, columns=feature_names)
xtest_df  = pd.DataFrame(xtest,  columns=feature_names)
def f(X_in):
    return ada.predict_proba(X_in)[:, 1]
background = shap.maskers.Independent(xtrain_df, max_samples=100)
explainer  = shap.Explainer(f, background)
sample = xtest_df.sample(min(len(xtest_df), 300), random_state=42)
sv = explainer(sample)
shap.plots.bar(sv, max_display=15)
shap.plots.beeswarm(sv, max_display=15)
shap.plots.waterfall(sv[0], max_display=10)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
base = AdaBoostClassifier(random_state=42)
cal_model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
cal_model.fit(xtrain, ytrain)
proba = cal_model.predict_proba(xtest)[:, 1]

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
raw = base.fit(xtrain, ytrain).predict_proba(xtest)[:, 1]
for name, p in [("Raw AdaBoost", raw), ("Calibrated", proba)]:
    frac_pos, mean_pred = calibration_curve(ytest, p, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier={brier_score_loss(ytest, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Perfect")
plt.xlabel("Predicted probability"); plt.ylabel("Actual churn rate"); plt.legend()
plt.show()

from sklearn.metrics import roc_auc_score, average_precision_score
fitted = {"LogReg": lr, "SVC": svc, "AdaBoost": ada, "RF": forest}
for name, m in fitted.items():
    if hasattr(m, "predict_proba"):
        s = m.predict_proba(xtest)[:, 1]
    else:
        s = m.decision_function(xtest)
    print(name, round(roc_auc_score(ytest, s), 3), round(average_precision_score(ytest, s), 3))

from sklearn.metrics import brier_score_loss
print(brier_score_loss(ytest, lr.predict_proba(xtest)[:, 1]))

# at the end of ai/model.py (or in a separate train.py)
import joblib
joblib.dump(ada, "ai/model.pkl")
joblib.dump(ct, "ai/preprocessor.pkl")