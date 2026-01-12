import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import os

os.makedirs('models', exist_ok=True)

# Load data
path = os.path.join('data', 'your_file.csv')
df = pd.read_csv(path)
print('Loaded', df.shape, 'rows')

# Basic cleaning: drop rows with missing target
if 'target_column' not in df.columns:
    raise SystemExit('target_column not found in data')

# Encode categorical
df = pd.get_dummies(df, columns=['department'], drop_first=True)

# Select numeric features
features = [c for c in df.columns if c != 'target_column']
X = df[features].select_dtypes(include=[np.number]).fillna(0)
y = df['target_column']

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)

print('Accuracy:', acc)
print(report)

# Save model and results
dump(clf, os.path.join('models', 'final_model.joblib'))
with open('results.txt', 'w') as f:
    f.write(f'Accuracy: {acc}\n')
    f.write(report)
print('Saved model to models/final_model.joblib and results.txt')
