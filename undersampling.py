import pandas as pd

# Load dataset
df = pd.read_csv("unified_hate_speech_data.csv")

# Count label occurrences
label_counts = df['label'].value_counts()
min_samples = label_counts.min()

# Balance dataset by undersampling
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_samples)).reset_index(drop=True)

# Save balanced dataset
df_balanced.to_csv("balanced_hate_speech_data.csv", index=False)

print("✅ Dataset balanced and saved as 'balanced_hate_speech_data.csv'")
