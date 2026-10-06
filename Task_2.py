import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/credit_risk_dataset.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

df["income_to_loan"] = (
    df["person_income"] /
    (df["loan_amnt"] + 1)
)

df["loan_to_income"] = (
    df["loan_amnt"] /
    (df["person_income"] + 1)
)

df["log_income"] = np.log1p(
    df["person_income"]
)


X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Test data:", X_test.shape)


numeric_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "income_to_loan",
    "loan_to_income",
    "log_income"
]

categorical_columns = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

ordinal_columns = ["loan_grade"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder(
            categories=[
                ["A", "B", "C", "D", "E", "F", "G"]
            ]
        )
    )
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
    ("ordinal", ordinal_pipeline, ordinal_columns)
])

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

cv_scores = cross_val_score(
    model_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nPipeline CV Scores:")
print(cv_scores)

print(
    "Pipeline CV accuracy:",
    round(cv_scores.mean(), 4)
)

print(
    "CV standard deviation:",
    round(cv_scores.std(), 4)
)


model_pipeline.fit(
    X_train,
    y_train
)

print("\nPipeline fitted successfully")


joblib.dump(
    model_pipeline,
    "models/loan_pipeline.joblib"
)

print("Pipeline saved successfully")


loaded_pipeline = joblib.load(
    "models/loan_pipeline.joblib"
)

print("Pipeline loaded successfully")


new_customers = pd.DataFrame({
    "person_age": [25, 35, 45],
    "person_income": [40000, 70000, 100000],
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN"],
    "person_emp_length": [3, 8, 15],
    "loan_intent": ["EDUCATION", "HOMEIMPROVEMENT", "PERSONAL"],
    "loan_grade": ["B", "C", "A"],
    "loan_amnt": [5000,15000,10000],
    "loan_int_rate": [10.5,12.5,8.5],
    "loan_percent_income": [0.12, 0.20, 0.10],
    "cb_person_default_on_file": ["N", "N", "N"],
    "cb_person_cred_hist_length": [4,10,20]
})

new_customers["income_to_loan"] = (
    new_customers["person_income"] /
    (new_customers["loan_amnt"] + 1)
)

new_customers["loan_to_income"] = (
    new_customers["loan_amnt"] /
    (new_customers["person_income"] + 1)
)

new_customers["log_income"] = np.log1p(
    new_customers["person_income"]
)


predictions = loaded_pipeline.predict(
    new_customers
)

print("\nPredictions for 3 new rows:")
print(predictions)


knn_imputer = KNNImputer(n_neighbors=5)

numeric_data = X_train[numeric_columns]
numeric_knn = knn_imputer.fit_transform(numeric_data)

print("\nKNN Imputer result shape:")
print(numeric_knn.shape)

