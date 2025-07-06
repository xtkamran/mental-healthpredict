import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("depression_anxiety_data.csv")

# Selecting important features
features = ['bmi', 'epworth_score', 'suicidal', 'depressiveness', 'anxiousness']
X = df[features]
y_phq = df['phq_score']
y_gad = df['gad_score']

# Convert categorical features to numeric (if any)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train_phq, y_test_phq = train_test_split(X, y_phq, test_size=0.2, random_state=42)
X_train, X_test, y_train_gad, y_test_gad = train_test_split(X, y_gad, test_size=0.2, random_state=42)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")  # Other options: "median", "most_frequent"
X_train = imputer.fit_transform(X_train)


# Train Random Forest Models
model_phq = RandomForestRegressor(n_estimators=100, random_state=42)
model_gad = RandomForestRegressor(n_estimators=100, random_state=42)

model_phq.fit(X_train, y_train_phq)
model_gad.fit(X_train, y_train_gad)

# Save models
pickle.dump(model_phq, open("phq_model.pkl", "wb"))
pickle.dump(model_gad, open("gad_model.pkl", "wb"))

print("Models trained and saved successfully!")

