# Example training script: backend/ml/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Drop Google Trends trend_score if present
if 'trend_score' in df.columns:
    df = df.drop(columns=['trend_score'])

# Define features and target
X = df.drop(columns=['target_column'])  # Replace 'target_column' with your actual target
y = df['target_column']

# Encode categorical variables
X = pd.get_dummies(X, columns=['business_idea', 'location'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'ml/model.pkl')
