import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,GridSearchCV,RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,OrdinalEncoder,StandardScaler)
from sklearn.linear_model import (LogisticRegression,LinearRegression)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score, mean_absolute_error, mean_squared_error,r2_score)
from xgboost import XGBClassifier


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
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
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

tree_model = Pipeline([
    ("preprocessing",preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ))
])

tree_model.fit(X_train,y_train)

forest_model = Pipeline([
    ("preprocessing",preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=1
    ))
])

forest_model.fit(X_train,y_train)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

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

print("XGBoost trained successfully")
print("Best Iteration:",xgb_model.best_iteration)


def evaluate_model(name, model, X_data, y_data):
    prediction = model.predict(X_data)
    accuracy = accuracy_score(y_data,prediction)
    precision = precision_score(y_data,prediction,zero_division=0)
    recall = recall_score(y_data,prediction,zero_division=0)
    f1 = f1_score(y_data,prediction,zero_division=0)

    print("\n" + "=" *50)
    print(name)
    print("="*50)

    print("Accuracy:",round(accuracy,4))
    print("precision:", round(precision,4))
    print("Recall:", round(recall,4))
    print("F1 Score:", round(f1,4))

    cm = confusion_matrix(y_data,prediction)

    print("\n Confusion Matrix:")
    print(cm)

    return{
        "Model":name,
        "Accuracy":accuracy,
        "Precision": precision,
        "Recall":recall,
        "F1": f1
    }

tree_result = evaluate_model(
    "Decision Tree",
    tree_model,
    X_val,
    y_val
)

forest_result = evaluate_model(
    "Random Forest",
    forest_model,
    X_val,
    y_val
)

xgb_prediction = xgb_model.predict(
    X_val_processed
)

xgb_accuracy = accuracy_score(y_val, xgb_prediction)

xgb_precision = precision_score(y_val,xgb_prediction, zero_division=0)
xgb_recall = recall_score(y_val,xgb_prediction,zero_division=0)
xgb_f1 = f1_score(y_val,xgb_prediction,zero_division=0)
xgb_cm = confusion_matrix(y_val,xgb_prediction)

print("\n" + "=" * 50)
print("XGBoost")
print("=" * 50)

print("Accuracy :", round(xgb_accuracy, 4))
print("Precision:", round(xgb_precision, 4))
print("Recall :", round(xgb_recall, 4))
print("F1 Score :", round(xgb_f1, 4))

print("\nConfusion Matrix:")
print(xgb_cm)


ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm,
    display_labels=["No Default", "Default"]
).plot()

plt.title("XGBoost Confusion Matrix")

plt.savefig("outputs/xgb_confusion_matrix.png")
plt.show()


logistic_prob = logistic_model.predict_proba(X_val)[:, 1]
tree_prob = tree_model.predict_proba(X_val)[:, 1]
forest_prob = forest_model.predict_proba(X_val)[:, 1]

logistic_fpr, logistic_tpr, _ = roc_curve(y_val,logistic_prob)
tree_fpr, tree_tpr, _ = roc_curve(y_val,tree_prob)
forest_fpr, forest_tpr, _ = roc_curve(y_val,forest_prob)

logistic_auc = roc_auc_score(y_val,logistic_prob)
tree_auc = roc_auc_score(y_val,tree_prob)
forest_auc = roc_auc_score(y_val,forest_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression AUC = {logistic_auc:.3f}"
)

plt.plot(
    tree_fpr,
    tree_tpr,
    label=f"Decision Tree AUC = {tree_auc:.3f}"
)

plt.plot(
    forest_fpr,
    forest_tpr,
    label=f"Random Forest AUC = {forest_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)

plt.savefig("outputs/roc_curve.png")
plt.show()


xgb_prob = xgb_model.predict_proba(X_val_processed)[:, 1]

thresholds = np.arange(0.10,0.91, 0.01)

threshold_results = []

for threshold in thresholds:

    threshold_prediction = (
        xgb_prob >= threshold
    ).astype(int)

    recall = recall_score(y_val, threshold_prediction,zero_division=0)
    precision = precision_score(y_val,threshold_prediction,zero_division=0)

    threshold_results.append({
        "Threshold": threshold,
        "Recall": recall,
        "Precision": precision
    })

threshold_table = pd.DataFrame(
    threshold_results
)

print("\nThreshold Results:")
print(threshold_table)

valid_thresholds = threshold_table[
    threshold_table["Recall"] >= 0.75
]

best_threshold_row = valid_thresholds.iloc[-1]

best_threshold = best_threshold_row[
    "Threshold"
]

best_threshold_recall = best_threshold_row[
    "Recall"
]

best_threshold_precision = best_threshold_row[
    "Precision"
]

print("\nSelected Threshold:", best_threshold)
print("Recall:", best_threshold_recall)
print("Precision:", best_threshold_precision)


rf_pipeline = Pipeline([
    ("preprocessing",preprocessor),
    ("model",
     RandomForestClassifier(
         random_state=42,
         n_jobs=1
     ))
])

rf_parameters = {
    "model__n_estimators":[100,200],
    "model__max_depth":[5,10,None],
    "model__min_samples_split":[2,5]
}

rf_grid= GridSearchCV(
    rf_pipeline,
    rf_parameters,
    cv=5,
    scoring="f1",
    n_jobs=1
)
rf_grid.fit(
    X_train,
    y_train
)

print("\n Random Forest GridSearchCV")
print("Best Parameters:")
print(rf_grid.best_params_)

print(
    "Best CV F1:",round(rf_grid.best_score_,4)
)


xgb_parameters = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3,4,5,6,8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7,0.8,0.9,1.0],
    "colsample_bytree": [0.7,0.8,0.9,1.0]
}

xgb_search_model = XGBClassifier(random_state=42,
    eval_metric="logloss"
)

xgb_random = RandomizedSearchCV(
    xgb_search_model,
    xgb_parameters,
    n_iter=15,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(
    X_train_processed,
    y_train
)

print("\nXGBoost RandomizedSearchCV")

print("Best Parameters:")
print(
    xgb_random.best_params_
)

print(
    "Best CV F1:", round(xgb_random.best_score_, 4)
)



X_reg = df.drop(
    ["loan_amnt", "loan_status", "loan_percent_income"],
    axis=1
)

y_reg = df["loan_amnt"]

reg_numeric_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "income_to_loan",
    "loan_to_income",
    "log_income"
]

reg_categorical_columns = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

reg_ordinal_columns = [
    "loan_grade"
]

reg_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

reg_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
            sparse_output=False
        ))
])

reg_ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
            categories=[
                ["A", "B", "C", "D", "E", "F", "G"]
            ]
        ))
])

reg_preprocessor = ColumnTransformer([
    ("numeric", reg_numeric_pipeline, reg_numeric_columns),
    ("categorical", reg_categorical_pipeline, reg_categorical_columns),
    ("ordinal", reg_ordinal_pipeline, reg_ordinal_columns)
])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg,
    test_size=0.20,
    random_state=42
)

linear_model = Pipeline([
    ("preprocessing",reg_preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_reg_train, y_reg_train)
linear_prediction = linear_model.predict(X_reg_test)

linear_mae = mean_absolute_error(y_reg_test, linear_prediction)

linear_rmse = np.sqrt(
    mean_squared_error(y_reg_test, linear_prediction)
)

linear_r2 = r2_score(y_reg_test,linear_prediction)

print("\nLinear Regression")
print("MAE :", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2 :", round(linear_r2, 4))

mean_prediction = np.full(
    len(y_reg_test),
    y_reg_train.mean()
)

baseline_mae = mean_absolute_error(y_reg_test, mean_prediction)
baseline_rmse = np.sqrt(
    mean_squared_error(y_reg_test, mean_prediction)
)
baseline_r2 = r2_score(y_reg_test, mean_prediction)

print("\nMean Baseline")
print("MAE :", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2 :", round(baseline_r2, 4))

regression_results = pd.DataFrame({
    "Model": ["Mean Baseline", "Linear Regression"],
    "MAE": [baseline_mae, linear_mae],
    "RMSE": [baseline_rmse, linear_rmse],
    "R2": [baseline_r2, linear_r2]
})

print("\nRegression Results:")
print(regression_results)


X_test_processed = preprocessor.transform(X_test)

final_prediction = xgb_model.predict(X_test_processed)
final_probability = xgb_model.predict_proba(X_test_processed)[:, 1]
final_accuracy = accuracy_score(y_test, final_prediction)
final_precision = precision_score(y_test, final_prediction, zero_division=0)
final_recall = recall_score(y_test, final_prediction, zero_division=0)
final_f1 = f1_score(y_test, final_prediction, zero_division=0)
final_auc = roc_auc_score(y_test, final_probability)

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

print("Accuracy :", round(final_accuracy, 4))
print("Precision:", round(final_precision, 4))
print("Recall :", round(final_recall, 4))
print("F1 Score :", round(final_f1, 4))
print("ROC-AUC :", round(final_auc, 4))

print("\nConfusion Matrix:")
print(
    confusion_matrix(y_test, final_prediction)
)