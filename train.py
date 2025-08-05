import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Starting model training...")

# Data load karein
training_df = pd.read_csv('dataset.csv')

# Symptoms ki list banayein
symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]
all_symptoms = set()
for col in symptom_columns:
    training_df[col] = training_df[col].astype(str).str.strip().str.replace('_', ' ')
    unique_in_col = training_df[col][training_df[col] != 'nan'].unique()
    all_symptoms.update(unique_in_col)

symptom_list = sorted(list(all_symptoms))

# Features (X) aur Target (y) taiyar karein
X = pd.DataFrame(0, index=training_df.index, columns=symptom_list)
for index, row in training_df.iterrows():
    for col in symptom_columns:
        symptom = row[col]
        if symptom != 'nan' and symptom in symptom_list:
            X.loc[index, symptom] = 1

le = LabelEncoder()
y = le.fit_transform(training_df['Disease'])

# Model train karein
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Model aur zaroori files ko save karein
joblib.dump(model, 'disease_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(symptom_list, 'symptom_list.pkl')

print("Model, Label Encoder, aur Symptom List safaltapurvak 'pkl' files mein save ho gaye hain.")
