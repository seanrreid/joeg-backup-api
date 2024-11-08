# ml/model.py
import joblib
import os

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set the model path relative to the current file
MODEL_PATH = os.path.join(CURRENT_DIR, 'model.pkl')

def load_model():
    print(f"Loading model from: {MODEL_PATH}")  # Optional: For debugging
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()
