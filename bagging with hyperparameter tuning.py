import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv(r'D:\Sleep_health_and_lifestyle_dataset.csv')  # Adjust the path as necessary

# Label encoding
label_encoder = LabelEncoder()
for column in ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']:
    data[column] = label_encoder.fit_transform(data[column])

# Splitting data
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Define parameter grid for Bagging Classifier
param_grid_bagging = {
    'n_estimators': [10, 50, 100],  # Number of base estimators in the ensemble
    'max_samples': [0.5, 1.0],  # The maximum number of samples to train each base estimator
    'max_features': [0.5, 1.0]  # The maximum number of features to draw from X to train each base estimator
}

# Setup GridSearchCV for Bagging Classifier
grid_bagging = GridSearchCV(BaggingClassifier(random_state=42), param_grid_bagging, cv=2, verbose=2)  # Using 2 folds for speed
grid_bagging.fit(X_train, y_train)

# Best model
best_bagging = grid_bagging.best_estimator_

# Prediction and evaluation
y_pred_bagging = best_bagging.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
precision_bagging = precision_score(y_test, y_pred_bagging, average='macro')
recall_bagging = recall_score(y_test, y_pred_bagging, average='macro')
f1_bagging = f1_score(y_test, y_pred_bagging, average='macro')

# Print results
print("Bagging Classifier best parameters:", grid_bagging.best_params_)
print("Bagging Classifier accuracy:", accuracy_bagging)
print("Bagging Classifier precision:", precision_bagging)
print("Bagging Classifier recall:", recall_bagging)
print("Bagging Classifier F1 Score:", f1_bagging)
