from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

try:
    from models.cnn_model import TrafficCNN
except ImportError:
    from src.models.cnn_model import TrafficCNN


DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Train a 1D CNN on network traffic CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base_dir / "data" / "MachineLearningCVE",
        help="Directory containing one or more CSV files.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Name of the label column in the CSV files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of the dataset to reserve for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the DataLoader.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_label_column(columns: pd.Index, expected_label: str) -> str:
    normalized_map = {column.strip().lower(): column for column in columns}
    key = expected_label.strip().lower()

    if key not in normalized_map:
        raise ValueError(f"Missing required label column: {expected_label}")

    return normalized_map[key]


def find_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    csv_files = sorted(path for path in data_dir.rglob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    return csv_files


def load_csv_files(csv_files: list[Path], label_column: str) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []

    for csv_path in csv_files:
        try:
            dataframe = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            print(f"Skipping unreadable file {csv_path}: {exc}", file=sys.stderr)
            continue

        dataframe.columns = dataframe.columns.str.strip()
        resolved_label = resolve_label_column(dataframe.columns, label_column)
        if resolved_label != label_column:
            dataframe = dataframe.rename(columns={resolved_label: label_column})

        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError("No readable CSV files were loaded from the dataset directory.")

    return pd.concat(dataframes, ignore_index=True, sort=False)


def clean_dataframe(dataframe: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if label_column not in dataframe.columns:
        raise ValueError(f"Label column '{label_column}' was not found after loading the data.")

    dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.str.strip()
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    labels = dataframe[label_column].astype(str).str.strip()
    labels = labels.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    feature_frame = dataframe.drop(columns=[label_column])
    numeric_features = feature_frame.apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.loc[:, numeric_features.notna().any(axis=0)]

    if numeric_features.empty:
        raise ValueError("No numeric feature columns were found in the dataset.")

    valid_rows = labels.notna() & numeric_features.notna().all(axis=1)
    numeric_features = numeric_features.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].reset_index(drop=True)

    if numeric_features.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    if labels.nunique() < 2:
        raise ValueError("Training requires at least two distinct label classes.")

    return numeric_features, labels


def build_dataloaders(
    features: pd.DataFrame,
    labels: pd.Series,
    batch_size: int,
    test_size: float,
    random_state: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int, LabelEncoder]:
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    unique_classes, class_counts = np.unique(encoded_labels, return_counts=True)
    stratify_labels = encoded_labels if np.all(class_counts >= 2) else None

    x_train, x_test, y_train, y_test = train_test_split(
        features.values,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, len(unique_classes), label_encoder


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0

    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

        batch_size = batch_features.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

    return running_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate( model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float, float, float]:
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device, non_blocking=True)
        logits = model(batch_features)
        predictions = logits.argmax(dim=1).cpu().numpy()

        all_predictions.append(predictions)
        all_targets.append(batch_labels.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return accuracy, precision, recall, f1_score


def main() -> int:
    args = parse_args()
    set_seed(args.random_state)

    try:
        csv_files = find_csv_files(args.data_dir)
        raw_dataframe = load_csv_files(csv_files, args.label_column)
        features, labels = clean_dataframe(raw_dataframe, args.label_column)
        train_loader, test_loader, num_classes, label_encoder = build_dataloaders(
            features=features,
            labels=labels,
            batch_size=args.batch_size,
            test_size=args.test_size,
            random_state=args.random_state,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        print("Failed to prepare dataset:")
        traceback.print_exc()
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficCNN(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print(f"Loaded {len(csv_files)} CSV file(s) from {args.data_dir}")
    print(f"Using {features.shape[1]} numeric feature(s)")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Training on device: {device}")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        accuracy, precision, recall, f1_score = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"- loss: {epoch_loss:.4f} "
            f"- test_acc: {accuracy:.4f} "
            f"- precision: {precision:.4f} "
            f"- recall: {recall:.4f} "
            f"- f1: {f1_score:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
