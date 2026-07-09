# Disease Prediction using Machine Learning

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)



# -----------------------------------
# Load Dataset
# -----------------------------------

print("Loading Dataset...")

data = pd.read_csv("heart.csv")


print("\nFirst 5 rows:")
print(data.head())



# -----------------------------------
# Basic Information
# -----------------------------------

print("\nDataset Information:")
print(data.info())


print("\nChecking Missing Values:")
print(data.isnull().sum())



# -----------------------------------
# Data Visualization
# -----------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x="target",
    data=data
)

plt.title(
    "Disease Distribution"
)

plt.xlabel(
    "Heart Disease (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Number of Patients"
)

plt.show()



# -----------------------------------
# Separating Features and Target
# -----------------------------------


X = data.drop(
    "target",
    axis=1
)


y = data["target"]



# -----------------------------------
# Train Test Split
# -----------------------------------


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# -----------------------------------
# Feature Scaling
# -----------------------------------


scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)


X_test = scaler.transform(
    X_test
)



# -----------------------------------
# Model 1: Logistic Regression
# -----------------------------------


logistic_model = LogisticRegression()


logistic_model.fit(
    X_train,
    y_train
)


logistic_prediction = logistic_model.predict(
    X_test
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)



print("\nLogistic Regression Accuracy:")
print(
    logistic_accuracy*100,
    "%"
)



# -----------------------------------
# Model 2: Random Forest
# -----------------------------------


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


rf_prediction = rf_model.predict(
    X_test
)



rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)



print("\nRandom Forest Accuracy:")
print(
    rf_accuracy*100,
    "%"
)



# -----------------------------------
# Model 3: XGBoost
# -----------------------------------


xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_prediction = xgb_model.predict(
    X_test
)



xgb_accuracy = accuracy_score(
    y_test,
    xgb_prediction
)



print("\nXGBoost Accuracy:")
print(
    xgb_accuracy*100,
    "%"
)



# -----------------------------------
# Selecting Best Model
# -----------------------------------


models_accuracy = {

    "Logistic Regression":
    logistic_accuracy,

    "Random Forest":
    rf_accuracy,

    "XGBoost":
    xgb_accuracy
}



best_model_name = max(
    models_accuracy,
    key=models_accuracy.get
)



print(
    "\nBest Performing Model:",
    best_model_name
)



# -----------------------------------
# Detailed Evaluation
# -----------------------------------


if best_model_name == "Logistic Regression":

    final_prediction = logistic_prediction


elif best_model_name == "Random Forest":

    final_prediction = rf_prediction


else:

    final_prediction = xgb_prediction




print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_prediction
    )
)



# Confusion Matrix


matrix = confusion_matrix(
    y_test,
    final_prediction
)


plt.figure(
    figsize=(5,4)
)


sns.heatmap(
    matrix,
    annot=True,
    fmt="d"
)


plt.title(
    "Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.show()



# -----------------------------------
# Predict New Patient
# -----------------------------------


def predict_patient(patient_data):

    patient_data = np.array(
        patient_data
    ).reshape(
        1,-1
    )


    patient_data = scaler.transform(
        patient_data
    )


    result = xgb_model.predict(
        patient_data
    )


    if result[0] == 1:

        print(
            "Prediction: Patient may have heart disease"
        )

    else:

        print(
            "Prediction: Patient does not have heart disease"
        )



# Example Patient

# Values:
# age, sex, cp, trestbps, chol,
# fbs, restecg, thalach,
# exang, oldpeak, slope, ca, thal


sample_patient = [

    55,
    1,
    2,
    140,
    250,
    0,
    1,
    150,
    0,
    1.2,
    1,
    0,
    2

]


predict_patient(
    sample_patient
)