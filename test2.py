import pandas as pd

# df = pd.read_csv("balanced_hate_speech_data.csv")
# print(df['label'].value_counts())  # Check distribution

# Load dataset
df = pd.read_csv("unified_hate_speech_data.csv")  

# Drop NaN values in text and label columns
df = df.dropna(subset=['text', 'label'])  

# Convert labels to categorical values
label_mapping = {"hateful": 0, "offensive": 1, "neutral": 2, "profanity": 3}
df['label'] = df['label'].map(label_mapping)

# Remove any rows with unmapped labels
df = df.dropna(subset=['label'])

# Convert labels to integers
df['label'] = df['label'].astype(int)  

# Check for extra label values
print("Unique label values:", df['label'].unique())

# Proceed with training after these fixes...
df.to_csv("cleaned_hate_speech_data.csv", index=False)
