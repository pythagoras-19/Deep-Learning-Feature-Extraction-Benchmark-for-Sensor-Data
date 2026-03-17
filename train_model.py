from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent

# Candidate locations for CIC IDS 2017 CSV files
MONDAY_CANDIDATES = [
    BASE_DIR / "data/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
    BASE_DIR / "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
    BASE_DIR / "data/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
]

TUESDAY_CANDIDATES = [
    BASE_DIR / "data/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
    BASE_DIR / "data/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
    BASE_DIR / "data/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
]

THURSDAY_CANDIDATES = [
    BASE_DIR / "data/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    BASE_DIR / "data/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    BASE_DIR / "data/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

FRIDAY_CANDIDATES = [
    BASE_DIR / "data/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    BASE_DIR / "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    BASE_DIR / "data/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

WEDNESDAY_CANDIDATES = [
    BASE_DIR / "data/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
    BASE_DIR / "data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
    BASE_DIR / "data/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
]


def find_existing_file(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {label}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a CIC IDS dataframe:
    - strip whitespace from column names
    - replace inf/-inf with NaN
    - keep only numeric columns plus Label
    - drop missing rows
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)

    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in the dataset")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    df = df[numeric_columns + ["Label"]].dropna()

    return df


def collapse_to_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the multiclass CIC IDS labels into a binary problem:
    - BENIGN stays BENIGN
    - every other label becomes ATTACK
    """
    df = df.copy()
    df["Label"] = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
    return df


# Locate files
monday_path = find_existing_file(
    MONDAY_CANDIDATES,
    "Monday-WorkingHours.pcap_ISCX.csv"
)
tuesday_path = find_existing_file(
    TUESDAY_CANDIDATES,
    "Tuesday-WorkingHours.pcap_ISCX.csv"
)
thursday_path = find_existing_file(
    THURSDAY_CANDIDATES,
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)
friday_path = find_existing_file(
    FRIDAY_CANDIDATES,
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
)
wednesday_path = find_existing_file(
    WEDNESDAY_CANDIDATES,
    "Wednesday-workingHours.pcap_ISCX.csv"
)

print("Training files:")
print(f"  Monday:    {monday_path}")
print(f"  Tuesday:   {tuesday_path}")
print(f"  Thursday:  {thursday_path}")
print(f"  Friday:    {friday_path}")
print(f"Testing file:")
print(f"  Wednesday: {wednesday_path}")

# Load raw data
monday_df = pd.read_csv(monday_path)
tuesday_df = pd.read_csv(tuesday_path)
thursday_df = pd.read_csv(thursday_path)
friday_df = pd.read_csv(friday_path)
test_df = pd.read_csv(wednesday_path)

# Clean all datasets the same way
monday_df = clean_dataframe(monday_df)
tuesday_df = clean_dataframe(tuesday_df)
thursday_df = clean_dataframe(thursday_df)
friday_df = clean_dataframe(friday_df)
test_df = clean_dataframe(test_df)

# Convert to binary labels: BENIGN vs ATTACK
monday_df = collapse_to_binary_labels(monday_df)
tuesday_df = collapse_to_binary_labels(tuesday_df)
thursday_df = collapse_to_binary_labels(thursday_df)
friday_df = collapse_to_binary_labels(friday_df)
test_df = collapse_to_binary_labels(test_df)

# Combine training days
train_df = pd.concat(
    [monday_df, tuesday_df, thursday_df, friday_df],
    ignore_index=True
)

# Ensure feature columns align exactly
train_feature_columns = [col for col in train_df.columns if col != "Label"]
test_feature_columns = [col for col in test_df.columns if col != "Label"]

missing_in_test = set(train_feature_columns) - set(test_feature_columns)
extra_in_test = set(test_feature_columns) - set(train_feature_columns)

if missing_in_test:
    raise ValueError(
        f"Test dataset is missing feature columns present in training data: {sorted(missing_in_test)}"
    )

if extra_in_test:
    print(
        f"Warning: test dataset has extra feature columns not used for training: {sorted(extra_in_test)}"
    )

# Reorder test columns to match training exactly
test_df = test_df[train_feature_columns + ["Label"]]

# Encode labels using TRAINING labels only
le = LabelEncoder()
train_df["Label"] = le.fit_transform(train_df["Label"])
test_df["Label"] = le.transform(test_df["Label"])

# Build feature matrices and label vectors
X_train = train_df.drop("Label", axis=1).values
y_train = train_df["Label"].values

X_test = test_df.drop("Label", axis=1).values
y_test = test_df["Label"].values

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")

# Scale using TRAINING data only to avoid leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compute class weights from TRAINING labels only
classes = np.unique(y_train)
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

print("\nClass mapping:")
for idx, class_name in enumerate(le.classes_):
    print(f"  {idx}: {class_name}")

print("\nClass weights:")
for idx, weight in enumerate(class_weights):
    print(f"  {le.classes_[idx]}: {weight.item():.4f}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# MLP baseline
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, len(le.classes_))
)

# Weighted loss to address class imbalance
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train on Monday + Tuesday + Thursday + Friday
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Evaluate on Wednesday only
with torch.no_grad():
    test_logits = model(X_test)
    preds = test_logits.argmax(dim=1)
    acc = (preds == y_test).float().mean()

print(f"\nAccuracy: {acc.item():.4f}")
print("\nClassification Report:")
print(
    classification_report(
        y_test.numpy(),
        preds.numpy(),
        target_names=le.classes_,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test.numpy(), preds.numpy()))