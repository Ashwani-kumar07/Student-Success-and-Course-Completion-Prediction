# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:44:59 2025

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 22:53:54 2025

PROJECT: Student Success & Engagement Prediction System
OBJECTIVE: Predict course completion using Classification Models
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


sns.set(style="whitegrid")

# =========================================================
# STEP 1: LOAD DATASET
# =========================================================

df = pd.read_csv(r"C:\Users\Asus\OneDrive\Desktop\CA_2(predictive_ana.)\Course_Completion_Prediction.csv")

# Drop non-informative columns
cols_to_drop = ['Student_ID', 'Name', 'Enrollment_Date', 'Course_ID']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print("\nDataset Info:")
print(df.info())

# View first 5 rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# View last 5 rows
print("\nLast 5 rows of dataset:")
print(df.tail())

print("\nMissing Values:")
print(df.isnull().sum())


# ===========================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA) AND VISUALIZATION
# ===========================================================
print("\nTarget Variable Distribution:")
print(df['Completed'].value_counts())

#COUNTPLOT
plt.figure(figsize=(6, 4))
sns.countplot(x='Completed',hue='Completed', data=df, palette='Set2')
plt.title("Course Completion Distribution")
plt.show()

#HEATMAP
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


#BOX PLOT
plt.figure(figsize=(8, 5))
sns.boxplot(x='Completed',y='Time_Spent_Hours',hue='Completed',data=df,palette='coolwarm',legend=False)
plt.title("Time Spent vs Course Completion")
plt.show()

# =========================================================
# STEP 3: ENCODING
# =========================================================
le = LabelEncoder()

categorical_cols = df.select_dtypes(include='object').columns.drop('Completed')

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

target_encoder = LabelEncoder()
df['Completed'] = target_encoder.fit_transform(df['Completed'])

X = df.drop(columns='Completed')
y = df['Completed']

print("\nFeature Matrix Shape:", X.shape)

# =========================================================
# STEP 4: TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# STEP 5: FEATURE SCALING
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODEL 1: LOGISTIC REGRESSION
# =========================================================
print("\nLogistic Regression")

log_model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

acc_log = accuracy_score(y_test, y_pred_log) * 100
print("Accuracy:", round(acc_log, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# =========================================================
# MODEL 2: DECISION TREE (SIMPLIFIED)
# =========================================================
print("\nDecision Tree")

tree_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

acc_tree = accuracy_score(y_test, y_pred_tree) * 100
print("Accuracy:", round(acc_tree, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

plt.figure(figsize=(14, 7))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=target_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree for Course Completion")
plt.show()

# =========================================================
# MODEL 3: KNN
# =========================================================
print("\nK-Nearest Neighbors")

knn_model = KNeighborsClassifier(n_neighbors=7)

knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

acc_knn = accuracy_score(y_test, y_pred_knn) * 100
print("Accuracy:", round(acc_knn, 2))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# =========================================================
# MODEL 4: NAIVE BAYES
# =========================================================
print("\nNaive Bayes")

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

acc_nb = accuracy_score(y_test, y_pred_nb) * 100
print("Accuracy:", round(acc_nb, 2))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))


# MODEL 5: RANDOM FOREST
# =========================================================
print("\nRandom Forest")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model (scaled data use kar rahe hain – same pattern)
rf_model.fit(X_train_scaled, y_train)

# Prediction
y_pred_rf = rf_model.predict(X_test_scaled)

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf) * 100
print("Accuracy:", round(acc_rf, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# =========================================================
# FINAL MODEL COMPARISON
# =========================================================
accuracy_summary = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'KNN',
        'Naive Bayes',
        'Random Forest'
    ],
    'Accuracy (%)': [
        acc_log,
        acc_tree,
        acc_knn,
        acc_nb,
        acc_rf
    ]
}).sort_values(by='Accuracy (%)', ascending=False)

print("\nFinal Accuracy Summary:")
print(accuracy_summary)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=accuracy_summary,
    x='Accuracy (%)',
    y='Model',
    hue='Model',
    palette='magma',
    legend=False
)
plt.title("Model Accuracy Comparison")
plt.xlim(0, 100)
plt.show()

# =========================================================
# SAVE PREDICTIONS
# =========================================================
prediction_df = pd.DataFrame({
    'Actual': y_test.values,
    'Logistic_Regression': y_pred_log,
    'Decision_Tree': y_pred_tree,
    'KNN': y_pred_knn,
    'Naive_Bayes': y_pred_nb,
    'Random_Forest': y_pred_rf
})

prediction_df['Actual_Label'] = target_encoder.inverse_transform(prediction_df['Actual'])
prediction_df.to_csv("student_predictions_all_models.csv", index=False)

