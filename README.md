Week 2 Weekly Assessment - Machine Learning
Project Overview
This project is part of the Week 2 Machine Learning assessment.

The main objective is to build a complete machine learning workflow using a credit risk dataset.

The dataset contains information about customers and their loans. The main classification target is:

loan_status = 0 → No default
loan_status = 1 → Default
The project covers data splitting, preprocessing, feature engineering, classification, regression, clustering, model evaluation, and hyperparameter tuning.

Dataset
Dataset used:

credit_risk_dataset.csv

Dataset size:

Rows: 32,581
Columns: 12
Some important columns are:

person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
loan_percent_income
cb_person_default_on_file
cb_person_cred_hist_length
loan_status
The dataset contains some missing values, so preprocessing was required.

Task 1 - Data Splitting, Overfitting and Cross Validation
1. Train, Validation and Test Split
The dataset was divided into three parts:

Training data: 60%
Validation data: 20%
Test data: 20%
random_state=42 was used so that the same split can be reproduced.

Stratification was also used to keep the same class distribution in the three datasets.

Dataset sizes
Training:   19,548
Validation: 6,516
Test:       6,517
The class distribution was approximately:

No Default: 78%
Default:    22%
2. Decision Tree and Overfitting
Decision Tree models were tested with different values of max_depth:

Depth 1
Depth 3
Depth 5
Depth 10
Depth None
Results:

Depth 1:    Train 0.8288   Validation 0.8332
Depth 3:    Train 0.8794   Validation 0.8751
Depth 5:    Train 0.9066   Validation 0.9006
Depth 10:   Train 0.9385   Validation 0.9207
Depth None: Train 1.0000   Validation 0.8832
The unlimited-depth tree achieved 100% training accuracy but lower validation accuracy.

This shows overfitting.

A decision tree can memorize the training data when it becomes too deep.

3. Cross Validation
5-fold cross validation was performed using Logistic Regression.

Results:

CV Mean: 0.8192
CV Standard Deviation: 0.0041
Cross validation helps check whether the model performs consistently on different parts of the training data.

4. Data Leakage Demonstration
Two approaches were compared:

Accuracy with leakage:    0.8246
Accuracy without leakage: 0.8246
The example did not show a difference in accuracy for this dataset.

However, preprocessing should still be performed correctly by fitting preprocessing steps only on training data.

5. Machine Learning Types
Examples were classified as follows:

Problem	Type
Spam Detection	Supervised Learning
Customer Segmentation	Unsupervised Learning
House Price Prediction	Supervised Learning
Anomaly Detection	Unsupervised Learning
6. Basic Machine Learning Workflow
The basic workflow followed in this project is:

Understand the problem
Load and explore the data
Clean and preprocess the data
Split the data
Perform feature engineering
Train machine learning models
Evaluate and compare the models
Task 2 - Preprocessing and Feature Engineering
1. Missing Value Handling
Missing values were handled using SimpleImputer.

For numerical columns:

Median imputation
For categorical columns:

Most frequent value
KNN Imputer was also tested for numerical data.

KNN Imputer output:

(26064, 10)
2. Encoding
Categorical features were converted into numerical form.

One-Hot Encoding
One-Hot Encoding was used for categorical columns such as:

person_home_ownership
loan_intent
cb_person_default_on_file
Ordinal Encoding
loan_grade was treated as an ordered feature:

A < B < C < D < E < F < G
3. Feature Engineering
Additional features were created to provide more information to the models.

Examples include:

Income to loan ratio
Loan to income ratio
Log transformation of income
The dataset does not contain a date column, so date decomposition was not applicable.

4. Pipeline
A Scikit-learn Pipeline was created to combine preprocessing and model training.

A ColumnTransformer was used to apply different preprocessing methods to numerical and categorical columns.

This helps keep preprocessing organized and reduces the chance of applying preprocessing incorrectly.

5. Pipeline Cross Validation
Pipeline CV results:

[0.8659 0.8575 0.8603 0.8569 0.8624]

Mean Accuracy: 0.8606
Standard Deviation: 0.0033
6. Saving the Pipeline
The fitted pipeline was saved using joblib.

The pipeline was then loaded again and used to make predictions.

Example:

Predictions for 3 new rows:
[0 0 0]
Task 3 - Machine Learning Models
Four classification models were trained:

Logistic Regression
Decision Tree
Random Forest
XGBoost
1. Logistic Regression
Validation Accuracy:

0.8591
2. Decision Tree
Validation Accuracy:

0.9263
3. Random Forest
Validation Accuracy:

0.9306
OOB Score:

0.9313
OOB (Out-of-Bag) score gives an additional estimate of Random Forest performance using samples that were not used for individual trees during training.

4. XGBoost
Validation Accuracy:

0.9346
Best Iteration:

493
XGBoost gave the highest validation accuracy among the four classification models.

Classification Model Comparison
Model	Validation Accuracy
Logistic Regression	0.8591
Decision Tree	0.9263
Random Forest	0.9306
XGBoost	0.9346
Based on validation accuracy, XGBoost performed the best.

Classification Probability
predict_proba() was used with Logistic Regression to see the probability of each class.

This is useful because a classification model can provide probabilities instead of only giving a final class.

Example:

Class 0 probability
Class 1 probability
The final prediction is based on the probability threshold.

Regression
Regression models were used to predict the loan amount.

The following models were compared:

Linear Regression
Ridge Regression
Lasso Regression
Results
Linear Regression R2: 0.1226
Ridge Regression R2:  0.1226
Lasso Regression R2:  0.1225
The coefficients of the three models were also compared.

Lasso produced:

Number of zero coefficients: 3
Lasso can reduce some feature coefficients to zero, which can help with feature selection.

K-Means Clustering
K-Means clustering was used to divide customers into different groups.

Values of k from 2 to 8 were tested.

Silhouette scores:

k=2: 0.4364
k=3: 0.4397
k=4: 0.4233
k=5: 0.3571
k=6: 0.3647
k=7: 0.3635
k=8: 0.3497
The highest silhouette score was obtained with:

k = 3
Therefore, three customer segments were created.

Customer Segment Counts
Segment 0: 20,037
Segment 1:  8,986
Segment 2:  3,558
The segments showed different patterns in income and loan amounts.

For example, one segment had relatively high income and higher loan amounts, while another had lower income and smaller loans.

Task 4 - Model Evaluation and Hyperparameter Tuning
Different classification metrics were calculated:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC
Decision Tree
Accuracy:  0.9320
Precision: 0.9648
Recall:    0.7143
F1 Score:  0.8209
Random Forest
Accuracy:  0.9302
Precision: 0.9782
Recall:    0.6953
F1 Score:  0.8128
XGBoost
Accuracy:  0.9389
Precision: 0.9749
Recall:    0.7389
F1 Score:  0.8407
XGBoost provided the best overall validation performance among the tested classification models.

Class Imbalance
The target variable is imbalanced:

No Default: approximately 78%
Default:    approximately 22%
Because of this imbalance, accuracy alone may not be enough to judge model performance.

For example, a model could predict the majority class most of the time and still achieve high accuracy.

Therefore, precision, recall, F1-score and ROC-AUC were also used.

Threshold Tuning
Different probability thresholds were tested to find a threshold where recall was at least 75%.

Selected threshold:

0.41
Results:

Recall:    0.7509
Precision: 0.9535
This means the model can achieve approximately 75% recall while maintaining high precision.

Random Forest GridSearchCV
GridSearchCV was used to find better Random Forest parameters.

Best parameters:

max_depth = None
min_samples_split = 5
n_estimators = 200
Best CV F1 Score:

0.8187
XGBoost RandomizedSearchCV
RandomizedSearchCV was used for XGBoost.

Best parameters:

subsample = 1.0
n_estimators = 200
max_depth = 6
learning_rate = 0.1
colsample_bytree = 0.9
Best CV F1 Score:

0.8313
Regression Evaluation
Regression performance was evaluated using:

MAE
RMSE
R²
The model was also compared against a simple mean baseline.

Mean Baseline
MAE:  4919.03
RMSE: 6347.41
R2:   -0.0001
Linear Regression
MAE:  1676.35
RMSE: 2534.28
R2:   0.8406
The Linear Regression model performed much better than the mean baseline in this experiment.

Final Test Evaluation
The final classification model was evaluated on the test dataset.

Results:

Accuracy : 0.9379
Precision: 0.9585
Recall   : 0.7475
F1 Score : 0.8400
ROC-AUC  : 0.9497
Confusion Matrix:

[[5049   46]
 [ 359 1063]]
The test set was kept separate from the training and validation process and was used for the final evaluation.

Final Conclusion
This assessment helped demonstrate a complete machine learning workflow.

The main concepts covered were:

Train/validation/test splitting
Stratified splitting
Overfitting
Cross validation
Data leakage
Missing value imputation
One-Hot Encoding
Ordinal Encoding
Feature engineering
Pipelines
ColumnTransformer
Logistic Regression
Decision Tree
Random Forest
XGBoost
Prediction probabilities
Regression
Ridge and Lasso
K-Means clustering
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC
Threshold tuning
GridSearchCV
RandomizedSearchCV
MAE
RMSE
R²
Overall, XGBoost gave the best classification performance among the tested models, with a validation accuracy of approximately 93.46% and a final test accuracy of approximately 93.79%.

This project provided practical experience in building and evaluating machine learning models from raw data through final model evaluation.