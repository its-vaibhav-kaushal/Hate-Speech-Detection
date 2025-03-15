import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load dataset
df = pd.read_csv("unified_hate_speech_data.csv")

# Drop NaN values in text or label columns **before vectorizing**
df = df.dropna(subset=["text", "label"])  

# Encode Labels (Ensure they are integers)
label_mapping = {"neutral": 0, "offensive": 1, "hateful": 2, "profanity": 3}
df["label"] = df["label"].map(label_mapping)

# Drop any rows where labels didn't map correctly
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Now text and labels have the same number of rows
print(f"Dataset Size After Cleaning: {df.shape}")

# Feature Extraction (TF-IDF) **AFTER** filtering
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"].astype(str))

# Get final label array (y)
y = df["label"].values

# **Check that X and y have the same number of samples**
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=50, max_depth=4, tree_method="hist")  # Optimized parameters
xgb_model.fit(X_train, y_train)

# Save Model & Vectorizer
joblib.dump(xgb_model, "hate_speech_xgb_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained successfully and saved!")
