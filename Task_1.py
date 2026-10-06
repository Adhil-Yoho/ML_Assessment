import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/credit_risk_dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X,y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Training Data:", X_train.shape)
print("Validation Data:", X_val.shape)
print("Test Data:", X_test.shape)

print("Train:")
print(y_train.value_counts(normalize=True))

print("Validation:")
print(y_val.value_counts(normalize=True))

print("Test")
print(y_test.value_counts(normalize=True))

X_all = pd.concat([X_train, X_val, X_test])
X_all = pd.get_dummies(X_all, drop_first=True)
X_all = X_all.fillna(X_all.median())

X_train_processed = X_all.iloc[:len(X_train)]
X_val_processed = X_all.iloc[len(X_train):len(X_train) + len(X_val)]
X_test_processed = X_all.iloc[len(X_train) + len(X_val):]

print("Processed training shape:", X_train_processed.shape)

depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    train_prediction = model.predict(X_train_processed)
    val_prediction = model.predict(X_val_processed)

    train_accuracy = accuracy_score(y_train, train_prediction)

    val_accuracy = accuracy_score(y_val, val_prediction)

    train_scores.append(train_accuracy)
    val_scores.append(val_accuracy)

    print(
        f"Depth {depth}: "
        f"train {train_accuracy:.4f} "
        f"val {val_accuracy:.4f}"
    )


plt.figure(figsize=(8, 5))

depth_labels = ["1", "3", "5", "10", "None"]

plt.plot(
    depth_labels,
    train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_labels,
    val_scores,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Decision Tree Max Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("outputs/overfitting_plot.png")
plt.show()


numeric_features = ["person_age",
                    "person_income",
                    "loan_amnt",
                    "loan_percent_income",
                    "cb_person_cred_hist_length"
]

X_train_numeric = X_train[numeric_features]

logistic_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000))
])

cv_scores = cross_val_score(
    logistic_pipeline,
    X_train_numeric,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n Cross Validation Score:")
print(cv_scores)

print(f"\n CV Mean:{cv_scores.mean():.4f}")

print(f"\n Standard Deviation:{cv_scores.std():.4f}")


#Data Leakage
leakage_features = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

X_leak = df[leakage_features]
y_leak = df["loan_status"]


scaler_leak = StandardScaler()

X_scaled = scaler_leak.fit_transform(X_leak)

X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X_scaled,
    y_leak,
    test_size=0.20,
    random_state=42,
    stratify=y_leak
)

leak_model = LogisticRegression(max_iter=1000)

leak_model.fit(
    X_train_leak,
    y_train_leak
)

leak_prediction = leak_model.predict(X_test_leak)

leak_accuracy = accuracy_score(
    y_test_leak,
    leak_prediction
)

print("\nAccuracy with leakage:")
print(f"{leak_accuracy:.4f}")

X_train_correct, X_test_correct, y_train_correct, y_test_correct = train_test_split(
    X_leak,
    y_leak,
    test_size=0.20,
    random_state=42,
    stratify=y_leak
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_correct)

X_test_scaled = scaler.transform(X_test_correct)

correct_model = LogisticRegression(max_iter=1000)

correct_model.fit(
    X_train_scaled,
    y_train_correct
)

correct_prediction = correct_model.predict(X_test_scaled)

correct_accuracy = accuracy_score(
    y_test_correct,
    correct_prediction
)

print("\nAccuracy without leakage:")
print(f"{correct_accuracy:.4f}")