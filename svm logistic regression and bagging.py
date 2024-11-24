import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv('D:/Sleep_health_and_lifestyle_dataset.csv')

# Label encoding
label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# SVM
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
f1_svm = f1_score(y_test, y_pred_svm, average='macro')

# Logistic Regression
clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='macro')
recall_lr = recall_score(y_test, y_pred_lr, average='macro')
f1_lr = f1_score(y_test, y_pred_lr, average='macro')

# Bagging
clf_bagging = BaggingClassifier(random_state=42)
clf_bagging.fit(X_train, y_train)
y_pred_bagging = clf_bagging.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
precision_bagging = precision_score(y_test, y_pred_bagging, average='macro')
recall_bagging = recall_score(y_test, y_pred_bagging, average='macro')
f1_bagging = f1_score(y_test, y_pred_bagging, average='macro')

# Print metrics
print("SVM Metrics:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}".format(accuracy_svm, precision_svm, recall_svm, f1_svm))
print("\nLogistic Regression Metrics:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}".format(accuracy_lr, precision_lr, recall_lr, f1_lr))
print("\nBagging Metrics:\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}".format(accuracy_bagging, precision_bagging, recall_bagging, f1_bagging))
