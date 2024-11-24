import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r'D:\Sleep_health_and_lifestyle_dataset.csv')

# Label encoding
for column in ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']:
    data[column] = LabelEncoder().fit_transform(data[column])

# Split the data
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Data loaded and split successfully. Shape of train set:", X_train.shape)
