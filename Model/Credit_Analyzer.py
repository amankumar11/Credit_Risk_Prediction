# train_credit_analyzer.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
DATA_PATH = "Loan_default.csv"  # <-- Change to your actual file name

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

# -------------------------------
# 2. Basic Cleaning
# -------------------------------
print("\n🧹 Cleaning data...")

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"✅ Encoded {len(categorical_cols)} categorical columns.")

# -------------------------------
# 3. Define Features and Target
# -------------------------------
X = df.drop(columns=["Default", "LoanID"], errors='ignore')
y = df["Default"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Scaling Numerical Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Model Training
# -------------------------------
print("\n🚀 Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_scaled, y_train)
print("✅ Model training complete.")

# -------------------------------
# 7. Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8. Save Model and Scaler
# -------------------------------
print("\n💾 Saving model and scaler...")

with open("credit_analyzer_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "features": list(X.columns)
    }, f)

print("✅ Model saved as credit_analyzer_model.pkl")

# -------------------------------
# 9. Example Usage
# -------------------------------
print("\n🎯 Example prediction:")
example = X_test.iloc[0:1]
example_scaled = scaler.transform(example)
pred = model.predict(example_scaled)
print(f"Prediction for example (LoanID={df.iloc[example.index[0]]['LoanID']}): {pred[0]}")
