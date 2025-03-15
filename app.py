from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import re
import os

app = Flask(__name__)

# Paths to model and vectorizer
MODEL_PATH = "hate_speech_xgb_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_PATH = "unified_hate_speech_data.csv"

# Load the trained XGBoost model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError("Model file not found. Train the model first!")

# Load the text vectorizer
if os.path.exists(VECTORIZER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError("Vectorizer file not found. Ensure it was saved during training!")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

@app.route('/')
def home():
    return render_template('index.html')

# Label Mapping Dictionary
LABEL_MAP = {
    0: "Neutral",
    1: "Offensive",
    2: "Hateful",
    3: "Profanity"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get("text", "").strip()
    
    if not user_text:
        return jsonify({"error": "No input text provided."}), 400

    cleaned_text = clean_text(user_text)
    print(f"Cleaned text: {cleaned_text}")  # Debugging Log

    try:
        transformed_text = vectorizer.transform([cleaned_text])  # Convert text
        print(f"Transformed text shape: {transformed_text.shape}")  # Debugging Log

        prediction = model.predict(transformed_text)[0]
        prediction = int(prediction)  # Convert np.int64 → Python int
        prediction_label = LABEL_MAP.get(prediction, "Unknown")  # Convert to label

        print(f"Prediction: {prediction_label}")  # Debugging Log

        return jsonify({"prediction": prediction_label})
    except Exception as e:
        print(f"Prediction Error: {e}")  # Debugging Log
        return jsonify({"error": "Prediction failed!"}), 500



@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    user_text = data.get("text", "").strip()
    user_label = data.get("label", "").strip()
    
    if not user_text or not user_label:
        return jsonify({"error": "Missing text or label."}), 400
    
    new_data = pd.DataFrame([[user_text, user_label]], columns=["text", "label"])
    
    # Ensure headers are written only if the file doesn't exist
    file_exists = os.path.exists(DATASET_PATH)
    new_data.to_csv(DATASET_PATH, mode='a', header=not file_exists, index=False)

    print(f"Feedback Saved: {user_text} -> {user_label}")  # Debugging Log

    return jsonify({"message": "Feedback saved!"})

@app.route('/download-dataset')
def download_dataset():
    if os.path.exists(DATASET_PATH):
        return send_file(DATASET_PATH, as_attachment=True)
    else:
        return jsonify({"error": "Dataset not found!"}), 404

if __name__ == '__main__':
    app.run(debug=True)
