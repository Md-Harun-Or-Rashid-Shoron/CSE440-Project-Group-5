from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}

# Setup GridSearchCV
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=3, verbose=2)
grid_dt.fit(X_train, y_train)

# Best model
best_dt = grid_dt.best_estimator_

# Prediction and evaluation
y_pred_dt = best_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro')
recall_dt = recall_score(y_test, y_pred_dt, average='macro')
f1_dt = f1_score(y_test, y_pred_dt, average='macro')

# Print results
print("Decision Tree best parameters:", grid_dt.best_params_)
print("Decision Tree accuracy:", accuracy_dt)
print("Decision Tree precision:", precision_dt)
print("Decision Tree recall:", recall_dt)
print("Decision Tree F1 Score:", f1_dt)
