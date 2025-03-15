import joblib

# Load the trained model and vectorizer
xgb_model = joblib.load("hate_speech_xgb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Test cases
test_sentences = [
    "I hate you!",  # Expected: Profanity or Offensive
    "You are amazing!",  # Expected: Neutral
    "This is so stupid.",  # Expected: Offensive
    "You should be punished for this!",  # Expected: Offensive or Hateful
    "Have a great day!",  # Expected: Neutral
    "Shut up, you idiot!",  # Expected: Offensive
    "I will kill you!",  # Expected: Hateful
    "This is a very bad idea.",  # Expected: Neutral or Offensive
]

# Convert test cases to TF-IDF features
X_test = vectorizer.transform(test_sentences)

# Predict using the XGBoost model
predictions = xgb_model.predict(X_test)

# Label mapping (reverse lookup)
label_mapping = {0: "Neutral", 1: "Offensive", 2: "Hateful", 3: "Profanity"}

# Print results
for text, pred in zip(test_sentences, predictions):
    print(f"Input: {text} → Prediction: {label_mapping[pred]}")
