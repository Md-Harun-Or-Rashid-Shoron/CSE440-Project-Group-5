import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv(r'D:\Sleep_health_and_lifestyle_dataset.csv')

# Label encoding
label_encoder = LabelEncoder()
for column in ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']:
    data[column] = label_encoder.fit_transform(data[column])

# Splitting data
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs'],  # Algorithm to use in the optimization problem
    'max_iter': [200, 400]  # Increased maximum number of iterations
}

# Setup GridSearchCV for Logistic Regression
grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=2, verbose=2)
grid_lr.fit(X_train_scaled, y_train)

# Best model
best_lr = grid_lr.best_estimator_

# Prediction and evaluation
y_pred_lr = best_lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='macro')
recall_lr = recall_score(y_test, y_pred_lr, average='macro')
f1_lr = f1_score(y_test, y_pred_lr, average='macro')

# Print results
print("Logistic Regression best parameters:", grid_lr.best_params_)
print("Logistic Regression accuracy:", accuracy_lr)
print("Logistic Regression precision:", precision_lr)
print("Logistic Regression recall:", recall_lr)
print("Logistic Regression F1 Score:", f1_lr)
