import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
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

# Define a simpler parameter grid for SVM
param_grid_svm = {
    'C': [1, 10],  # Regularization parameter
    'kernel': ['rbf'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

# Setup GridSearchCV for SVM with reduced CV folds
grid_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=2, verbose=2)  # Reduced from 3 to 2 folds for speed
grid_svm.fit(X_train, y_train)

# Best model
best_svm = grid_svm.best_estimator_

# Prediction and evaluation
y_pred_svm = best_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
f1_svm = f1_score(y_test, y_pred_svm, average='macro')

# Print results
print("SVM best parameters:", grid_svm.best_params_)
print("SVM accuracy:", accuracy_svm)
print("SVM precision:", precision_svm)
print("SVM recall:", recall_svm)
print("SVM F1 Score:", f1_svm)
