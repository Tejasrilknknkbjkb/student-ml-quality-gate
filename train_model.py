import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic student data
data = {
    "attendance": [90, 85, 70, 60, 95, 50, 80, 75, 92, 45,
                   88, 65, 78, 55, 96, 40, 82, 72, 91, 48],
    "internal_marks": [85, 80, 65, 55, 90, 45, 75, 70, 88, 40,
                       82, 60, 73, 50, 95, 35, 78, 68, 87, 42],
    "assignment_marks": [90, 85, 70, 60, 95, 50, 80, 75, 92, 45,
                         88, 65, 78, 55, 96, 40, 82, 72, 91, 48],
    "previous_score": [88, 82, 68, 58, 92, 42, 78, 72, 90, 38,
                       85, 62, 75, 52, 94, 33, 80, 70, 89, 40]
}

df = pd.DataFrame(data)

# Calculate weighted score
df["weighted_score"] = (
    0.3 * df["attendance"] +
    0.3 * df["internal_marks"] +
    0.2 * df["assignment_marks"] +
    0.2 * df["previous_score"]
)

# Generate result labels
df["result"] = (df["weighted_score"] >= 60).astype(int)

X = df[[
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]]

y = df["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save generated files
df.to_csv("student_results.csv", index=False)
joblib.dump(model, "student_result_model.pkl")

metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed.")
print("Model Accuracy:", round(accuracy, 4))
