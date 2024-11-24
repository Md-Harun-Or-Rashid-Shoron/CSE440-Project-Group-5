from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Define parameter grid for KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Setup GridSearchCV for KNN
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3, verbose=2)
grid_knn.fit(X_train, y_train)

# Best model
best_knn = grid_knn.best_estimator_

# Prediction and evaluation
y_pred_knn = best_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='macro')
recall_knn = recall_score(y_test, y_pred_knn, average='macro')
f1_knn = f1_score(y_test, y_pred_knn, average='macro')

# Print results
print("KNN best parameters:", grid_knn.best_params_)
print("KNN accuracy:", accuracy_knn)
print("KNN precision:", precision_knn)
print("KNN recall:", recall_knn)
print("KNN F1 Score:", f1_knn)
