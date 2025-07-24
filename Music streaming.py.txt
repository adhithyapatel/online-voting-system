# Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
%matplotlib inline

# Load Dataset
df = pd.read_csv('dataset.csv')

# Resample Dataset to Balance It
df = resample(df, replace=True, n_samples=1000, random_state=42)

# Drop missing values
df = df.dropna()

# Encode Categorical Columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Split features and label
X = df.drop(columns=['rainfall'])
y = df['rainfall']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluation Metrics Storage
accuracy, precision, recall, fscore = [], [], [], []

# Define Labels
labels = ['yes', 'no']

# Define Metric Calculation Function
def calculateMetrics(algorithm, testY, predict):
    testY = testY.astype('int')
    predict = predict.astype('int')
    
    a = accuracy_score(testY, predict) * 100
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    print(f"{algorithm} Accuracy    : {a}")
    print(f"{algorithm} Precision   : {p}")
    print(f"{algorithm} Recall      : {r}")
    print(f"{algorithm} FSCORE      : {f}")

    print(f"\n{algorithm} classification report\n", classification_report(predict, testY, target_names=labels))

    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

# Train or Load Decision Tree Classifier
if os.path.exists('model/DecisionTreeClassifier.pkl'):
    DTC = joblib.load('model/DecisionTreeClassifier.pkl')
    print("DecisionTreeClassifier model loaded successfully.")
else:
    DTC = DecisionTreeClassifier(max_depth=4)
    DTC.fit(X_train, y_train)
    os.makedirs('model', exist_ok=True)
    joblib.dump(DTC, 'model/DecisionTreeClassifier.pkl')
    print("DecisionTreeClassifier model trained and saved successfully.")

# Predict and Evaluate Decision Tree
predict = DTC.predict(X_test)
calculateMetrics("DecisionTreeClassifier", y_test, predict)

# Train or Load Random Forest Classifier
if os.path.exists('model/RandomForestClassifier.pkl'):
    RFC = joblib.load('model/RandomForestClassifier.pkl')
    print("RandomForestClassifier model loaded successfully.")
else:
    RFC = RandomForestClassifier(n_estimators=40, max_depth=8)
    RFC.fit(X_train, y_train)
    joblib.dump(RFC, 'model/RandomForestClassifier.pkl')
    print("RandomForestClassifier model trained and saved successfully.")

# Predict and Evaluate Random Forest
predict = RFC.predict(X_test)
calculateMetrics("RandomForestClassifier", y_test, predict)

# --- PREDICTION ON NEW TEST DATA ---
# Load New Test Data
test = pd.read_csv('test.csv')

# Encode if necessary (assuming test data structure matches training data)
for column in test.columns:
    if test[column].dtype == 'object':
        test[column] = le.fit_transform(test[column])

# Scale Features
test_scaled = scaler.transform(test)

# Predict with Decision Tree
test['predict'] = DTC.predict(test_scaled)

# Show Predictions
print("\nPredictions on Test Data:\n")
print(test)
