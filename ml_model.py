# backend/ml_model.py
import pickle

def load_model():
    with open("models/evaluation_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

def predict_success(features):
    return model.predict(features)
