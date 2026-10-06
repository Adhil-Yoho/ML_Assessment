import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,OrdinalEncoder,StandardScaler)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
from sklearn.cluster import KMeans

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

X_train, X_temp, y_train, y_temp = train_test_split(
    X,y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=2/3,
    random_state=42,
    stratify=y_temp
)

print("Training:", X_train.shape)
print("Validation:",X_val.shape)
print("Test:", X_test.shape)

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

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train,y_train)

logistic_prediction = logistic_model.predict(X_val)
logistic_accuracy = accuracy_score(y_val,logistic_prediction)

print("\n Logistic Regression:")
print("Validation Accuracy:", logistic_accuracy)

tree_model = Pipeline([
    ("preprocessing",preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ))
])

tree_model.fit(X_train,y_train)

tree_prediction = tree_model.predict(X_val)

tree_accuracy = accuracy_score(y_val,tree_prediction)

print("\n Decisition Trees:")
print("Validation Accuracy", tree_accuracy)


oob_score=True
forest_model = Pipeline([
    ("preprocessing",preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        oob_score=True,
        n_jobs=1
    ))
])

forest_model.fit(X_train,y_train)

forest_prediction = forest_model.predict(X_val)
forest_accuracy = accuracy_score(y_val, forest_prediction)

print("\n Random Forest:")
print('Validation Accuracy', forest_accuracy)

forest_classifier = forest_model.named_steps["model"]
print("OOB Score:", forest_classifier.oob_score_)

preprocessor_xgb = preprocessor
X_train_processed = preprocessor_xgb.fit_transform(X_train)
X_val_processed = preprocessor_xgb.transform(X_val)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_processed,
    y_train,
    eval_set=[
        (X_val_processed,y_val)
    ],
    verbose=False
)

xgb_prediction = xgb_model.predict(X_val_processed)
xgb_accuracy= accuracy_score(y_val, xgb_prediction)

print("\n XGBoost:")
print("Validation Accuracy:", xgb_accuracy)
print("Best Iteration:",xgb_model.best_iteration)


regression_df=df.copy()
y_reg = regression_df["loan_amnt"]

X_reg = regression_df.drop(columns=["loan_status","loan_amnt","loan_percent_income"])

reg_numeric_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_int_rate",
    "cb_person_cred_hist_length"
]

reg_categorical_columns = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

reg_ordinal_columns = ["loan_grade"]

reg_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

reg_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

reg_ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder(
            categories=[
                ["A", "B", "C", "D", "E", "F", "G"]
            ]
        )
    )
])

reg_preprocessor = ColumnTransformer([
    ("numeric", reg_numeric_pipeline, reg_numeric_columns),
    ("categorical", reg_categorical_pipeline, reg_categorical_columns),
    ("ordinal", reg_ordinal_pipeline, reg_ordinal_columns)
])

X_reg_train, X_reg_test,y_reg_train, y_reg_test = train_test_split(
    X_reg,y_reg,
    test_size=0.20,
    random_state=42
)

linear_model = Pipeline([
    ("preprocessing",reg_preprocessor),
    ("model", LinearRegression()),
])
linear_model.fit(
    X_reg_train,y_reg_train
)
linear_prediction = linear_model.predict(X_reg_test)
linear_r2 = r2_score(y_reg_test,linear_prediction)

print("\n Linear Regression:")
print("R2 Score:", linear_r2)

ridge_model = Pipeline([
    ("preprocessing",reg_preprocessor),
    ("model", Ridge(alpha=1.0)),
])
ridge_model.fit(
    X_reg_train,y_reg_train
)
ridge_prediction = ridge_model.predict(X_reg_test)
ridge_r2 = r2_score(y_reg_test,ridge_prediction)

print("\n Ridge Regression:")
print("R2 Score:", ridge_r2)

lasso_model = Pipeline([
    ("preprocessing",reg_preprocessor),
    ("model",Lasso(alpha=1.0)),
])
lasso_model.fit(
    X_reg_train,y_reg_train
)
lasso_prediction = lasso_model.predict(X_reg_test)
lasso_r2 = r2_score(y_reg_test,lasso_prediction)

print("\n Lasso Regression:")
print("R2 Score:", lasso_r2)

reg_feature_names = reg_preprocessor.get_feature_names_out()

linear_coefficients = (
    linear_model.named_steps["model"].coef_
)
ridge_coefficients = (
    ridge_model.named_steps["model"].coef_
)
lasso_coefficients = (
    lasso_model.named_steps["model"].coef_
)

coefficient_table = pd.DataFrame({
    "Feature":reg_feature_names,
    "Linear":linear_coefficients,
    "Ridge":ridge_coefficients,
    "Lasso":lasso_coefficients
})
print("\n Regression Coefficients:")
print(coefficient_table)

zero_lasso = np.sum(lasso_coefficients== 0)

print("\n Number of zero Lasso Coeffecients:",zero_lasso)


cluster_features =[
    "person_income",
    "loan_amnt",
    "loan_percent_income"
]

cluster_data = df[cluster_features].copy()

cluster_data = cluster_data.fillna(cluster_data.median())
cluster_scaler = StandardScaler()
cluster_scaled= cluster_scaler.fit_transform(cluster_data)

inertias =[]

for k in range(2,9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(cluster_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5)) 
plt.plot( range(2, 9), inertias, marker="o" ) 

plt.xlabel("Number of Clusters (k)") 
plt.ylabel("Inertia") 
plt.title("K-Means Elbow Curve") 
plt.grid(True) 

plt.savefig( "outputs/elbow_plot.png" ) 
plt.show()

silhouette_scores = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(cluster_scaled)
    score = silhouette_score(cluster_scaled, labels)
    silhouette_scores.append(score)

    print(f"k={k}: silhouette score={score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), silhouette_scores,marker="o")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")
plt.grid(True)

plt.savefig("outputs/silhouette_plot.png")
plt.show()


best_k = 3

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["customer_segment"] = final_kmeans.fit_predict(cluster_scaled)

print("\nCustomer Segment Counts:")
print(df["customer_segment"].value_counts())

segment_summary = df.groupby(
    "customer_segment"
)[
    ["person_income","loan_amnt","loan_percent_income"]].mean()

print("\nCustomer Segment Summary:")
print(segment_summary)

classification_results = pd.DataFrame({
    "Model": ["Logistic Regression","Decision Tree","Random Forest","XGBoost"],
    "Validation Score": [logistic_accuracy,tree_accuracy, forest_accuracy, xgb_accuracy]
})

print("\nFinal Classification Results")
print(classification_results)

classification_results.to_csv(
    "outputs/classification_results.csv",
    index=False
)