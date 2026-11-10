import pandas as pd
import pickle
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from app import pdf_buffer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"

# Check dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# Load dataset
data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("Dataset loaded successfully!")

# Clean column names
def clean_column(name):
    name = str(name).replace("\ufeff", "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")

data.columns = [clean_column(col) for col in data.columns]

print("Available columns:", data.columns.tolist())

# Required columns
features = [
    "attendence",
    "study_hours",
    "previous_marks",
    "assignment_score"
]

target = "final_marks"

# Check missing columns
required = features + [target]

missing = [col for col in required if col not in data.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {data.columns.tolist()}"
    )

# Convert data into numeric values
for col in required:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove incomplete records
data = data.dropna(subset=required)

if len(data) < 10:
    raise ValueError(
        "At least 10 valid student records are required."
    )

# Input and output
X = data[features]
y = data[target]

print("Valid student records:", len(data))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Train model
print("Training model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("\nModel trained successfully!")
print("Model saved at:", MODEL_PATH)
pdf = canvas.Canvas(pdf_buffer)