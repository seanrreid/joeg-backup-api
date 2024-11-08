# ml/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from xgboost import XGBClassifier

def create_sample_data():
    data = {
        'business_idea': ['Coffee Shop', 'Bakery', 'Gym', 'Book Store', 'Spa Center'] * 20,
        'location': ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'San Francisco, CA'] * 20,
        'competitors': [10, 5, 15, 3, 8] * 20,
        'trend_score': [0.8, 0.6, 0.9, 0.5, 0.7] * 20,
        'economic_indicator': [0.8, 0.75, 0.7, 0.65, 0.85] * 20,
        'success': [1, 0, 1, 0, 1] * 20
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    df_encoded = pd.get_dummies(df, columns=['business_idea', 'location'], drop_first=True)
    X = df_encoded.drop('success', axis=1)
    y = df_encoded['success']
    return X, y

def train_model():
    df = create_sample_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(clf, 'model.pkl')
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_model()
