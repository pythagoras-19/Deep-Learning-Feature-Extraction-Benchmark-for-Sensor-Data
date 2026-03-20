from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
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
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_VAL_SIZE = 0.1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_PATIENCE = 5
SOURCE_COLUMN = "__source_file__"


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
        "--split-mode",
        type=str,
        choices=("row", "file"),
        default="row",
        help="Split by rows across all files or by holding out entire files.",
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
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Fraction of the data reserved for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of the data reserved for testing.",
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
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience based on the validation metric.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=("f1", "accuracy"),
        default="f1",
        help="Validation metric used for checkpoint selection.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=base_dir / "checkpoints" / "traffic_cnn_best.pt",
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional path to export final test predictions as CSV.",
    )

    args = parser.parse_args()

    if args.val_size <= 0 or args.test_size <= 0:
        parser.error("--val-size and --test-size must both be greater than 0.")
    if args.val_size + args.test_size >= 1.0:
        parser.error("--val-size + --test-size must be less than 1.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0.")
    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0.")
    if args.patience < 0:
        parser.error("--patience must be greater than or equal to 0.")

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def load_csv_files(csv_files: list[Path], label_column: str) -> list[pd.DataFrame]:
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

        dataframe[SOURCE_COLUMN] = csv_path.name
        dataframes.append(dataframe)

    if not dataframes:
        raise RuntimeError("No readable CSV files were loaded from the dataset directory.")

    return dataframes


def clean_dataframe(
    dataframe: pd.DataFrame,
    label_column: str,
    metadata_columns: tuple[str, ...] = (SOURCE_COLUMN,),
) -> pd.DataFrame:
    if label_column not in dataframe.columns:
        raise ValueError(f"Label column '{label_column}' was not found after loading the data.")

    dataframe = dataframe.copy()
    dataframe.columns = dataframe.columns.str.strip()
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    labels = dataframe[label_column].astype(str).str.strip()
    labels = labels.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    feature_frame = dataframe.drop(columns=[label_column, *metadata_columns], errors="ignore")
    numeric_features = feature_frame.apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.loc[:, numeric_features.notna().any(axis=0)]

    if numeric_features.empty:
        raise ValueError("No numeric feature columns were found in the dataset.")

    valid_rows = labels.notna() & numeric_features.notna().all(axis=1)
    cleaned_features = numeric_features.loc[valid_rows].reset_index(drop=True)
    cleaned_labels = labels.loc[valid_rows].reset_index(drop=True)

    if cleaned_features.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    cleaned_dataframe = cleaned_features.copy()
    cleaned_dataframe[label_column] = cleaned_labels

    for column in metadata_columns:
        if column in dataframe.columns:
            cleaned_dataframe[column] = dataframe.loc[valid_rows, column].reset_index(drop=True)

    return cleaned_dataframe


def prepare_dataset(raw_dataframes: list[pd.DataFrame], label_column: str) -> pd.DataFrame:
    cleaned_frames: list[pd.DataFrame] = []
    feature_sets: list[set[str]] = []

    for dataframe in raw_dataframes:
        try:
            cleaned = clean_dataframe(dataframe, label_column)
        except ValueError as exc:
            source_name = dataframe[SOURCE_COLUMN].iloc[0] if SOURCE_COLUMN in dataframe.columns else "unknown"
            print(f"Skipping file after cleaning failure ({source_name}): {exc}", file=sys.stderr)
            continue

        feature_columns = {
            column for column in cleaned.columns
            if column not in {label_column, SOURCE_COLUMN}
        }

        if not feature_columns:
            source_name = cleaned[SOURCE_COLUMN].iloc[0] if SOURCE_COLUMN in cleaned.columns else "unknown"
            print(f"Skipping file with no usable numeric features ({source_name}).", file=sys.stderr)
            continue

        cleaned_frames.append(cleaned)
        feature_sets.append(feature_columns)

    if not cleaned_frames:
        raise RuntimeError("No CSV files produced usable cleaned data.")

    shared_features = sorted(set.intersection(*feature_sets))
    if not shared_features:
        raise ValueError("No common numeric feature columns remain across the cleaned CSV files.")

    combined = pd.concat(
        [frame[shared_features + [label_column, SOURCE_COLUMN]] for frame in cleaned_frames],
        ignore_index=True,
    )

    if combined[label_column].nunique() < 2:
        raise ValueError("Training requires at least two distinct label classes.")

    return combined


def get_stratify_labels(labels: pd.Series) -> pd.Series | None:
    class_counts = labels.value_counts()
    if class_counts.nunique() == 0:
        return None
    if len(class_counts) < 2:
        return None
    if (class_counts < 2).any():
        return None
    return labels


def safe_dataframe_split(
    dataframe: pd.DataFrame,
    test_size: float,
    label_column: str,
    random_state: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = get_stratify_labels(dataframe[label_column])

    try:
        left, right = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        if stratify_labels is None:
            raise
        print(
            f"Falling back to non-stratified {split_name} split because stratification is not feasible.",
            file=sys.stderr,
        )
        left, right = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    return left.reset_index(drop=True), right.reset_index(drop=True)


def safe_file_split(
    source_labels: pd.Series,
    test_size: float,
    random_state: int,
    split_name: str,
) -> tuple[list[str], list[str]]:
    file_names = source_labels.index.to_numpy()
    stratify_labels = get_stratify_labels(source_labels)

    try:
        left, right = train_test_split(
            file_names,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels.to_numpy() if stratify_labels is not None else None,
        )
    except ValueError:
        if stratify_labels is None:
            raise
        print(
            f"Falling back to non-stratified {split_name} file split because stratification is not feasible.",
            file=sys.stderr,
        )
        left, right = train_test_split(
            file_names,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    return list(left), list(right)


def split_dataset(
    dataframe: pd.DataFrame,
    label_column: str,
    split_mode: str,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    held_out_fraction = val_size + test_size
    relative_test_size = test_size / held_out_fraction

    if split_mode == "row":
        train_df, held_out_df = safe_dataframe_split(
            dataframe=dataframe,
            test_size=held_out_fraction,
            label_column=label_column,
            random_state=random_state,
            split_name="train vs held-out",
        )
        val_df, test_df = safe_dataframe_split(
            dataframe=held_out_df,
            test_size=relative_test_size,
            label_column=label_column,
            random_state=random_state,
            split_name="validation vs test",
        )
        return train_df, val_df, test_df

    if dataframe[SOURCE_COLUMN].nunique() < 3:
        raise ValueError("File split mode requires at least three distinct CSV files.")

    source_labels = dataframe.groupby(SOURCE_COLUMN)[label_column].agg(
        lambda values: values.value_counts().idxmax()
    )

    train_files, held_out_files = safe_file_split(
        source_labels=source_labels,
        test_size=held_out_fraction,
        random_state=random_state,
        split_name="train vs held-out",
    )

    held_out_labels = source_labels.loc[held_out_files]
    val_files, test_files = safe_file_split(
        source_labels=held_out_labels,
        test_size=relative_test_size,
        random_state=random_state,
        split_name="validation vs test",
    )

    train_df = dataframe[dataframe[SOURCE_COLUMN].isin(train_files)].reset_index(drop=True)
    val_df = dataframe[dataframe[SOURCE_COLUMN].isin(val_files)].reset_index(drop=True)
    test_df = dataframe[dataframe[SOURCE_COLUMN].isin(test_files)].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("File split produced an empty train, validation, or test split.")

    return train_df, val_df, test_df


def print_class_counts(split_name: str, labels: pd.Series) -> None:
    counts = labels.value_counts().sort_index()
    print(f"{split_name} class counts:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {int(count)}")


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    batch_size: int,
    num_workers: int,
    random_state: int,
) -> tuple[dict[str, DataLoader], LabelEncoder]:
    feature_columns = [
        column for column in train_df.columns
        if column not in {label_column, SOURCE_COLUMN}
    ]

    if not feature_columns:
        raise ValueError("No feature columns are available for training.")

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_df[label_column], val_df[label_column], test_df[label_column]]))

    x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
    x_val = val_df[feature_columns].to_numpy(dtype=np.float32)
    x_test = test_df[feature_columns].to_numpy(dtype=np.float32)

    y_train = label_encoder.transform(train_df[label_column])
    y_val = label_encoder.transform(val_df[label_column])
    y_test = label_encoder.transform(test_df[label_column])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(random_state)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return dataloaders, label_encoder


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
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> dict[str, object]:
    model.eval()
    running_loss = 0.0
    total_examples = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)
        predictions = logits.argmax(dim=1)

        batch_size = batch_features.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_labels.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics: dict[str, object] = {
        "loss": running_loss / max(total_examples, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }

    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred

    return metrics


def final_test_report(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    label_encoder: LabelEncoder,
    test_dataframe: pd.DataFrame,
    predictions_csv: Path | None,
) -> None:
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        return_predictions=True,
    )

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    class_names = list(label_encoder.classes_)

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    print("\nFinal test metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Weighted Precision: {metrics['precision']:.4f}")
    print(f"  Weighted Recall: {metrics['recall']:.4f}")
    print(f"  Weighted F1: {metrics['f1']:.4f}")

    print("\nPer-class metrics:")
    for index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: "
            f"precision={per_class_precision[index]:.4f}, "
            f"recall={per_class_recall[index]:.4f}, "
            f"f1={per_class_f1[index]:.4f}, "
            f"support={int(support[index])}"
        )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )
    print("Confusion matrix:")
    print(confusion_df.to_string())

    if predictions_csv is not None:
        export_df = test_dataframe[[SOURCE_COLUMN]].copy()
        export_df["true_label"] = label_encoder.inverse_transform(y_true)
        export_df["predicted_label"] = label_encoder.inverse_transform(y_pred)
        export_df["correct"] = export_df["true_label"] == export_df["predicted_label"]

        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(predictions_csv, index=False)
        print(f"\nSaved final predictions to {predictions_csv}")


def main() -> int:
    args = parse_args()
    set_seed(args.random_state)

    try:
        csv_files = find_csv_files(args.data_dir)
        raw_dataframes = load_csv_files(csv_files, args.label_column)
        dataset = prepare_dataset(raw_dataframes, args.label_column)
        train_df, val_df, test_df = split_dataset(
            dataframe=dataset,
            label_column=args.label_column,
            split_mode=args.split_mode,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        dataloaders, label_encoder = build_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_column=args.label_column,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.random_state,
        )
    except Exception as exc:
        print(f"Failed to prepare dataset: {exc}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficCNN(num_classes=len(label_encoder.classes_)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(csv_files)} CSV file(s) from {args.data_dir}")
    print(f"Split mode: {args.split_mode}")
    print(f"Using {dataset.drop(columns=[args.label_column, SOURCE_COLUMN]).shape[1]} numeric feature(s)")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Training on device: {device}")

    print_class_counts("Full dataset", dataset[args.label_column])
    print_class_counts("Train split", train_df[args.label_column])
    print_class_counts("Validation split", val_df[args.label_column])
    print_class_counts("Test split", test_df[args.label_column])

    if args.split_mode == "file":
        print("\nFile assignments:")
        print(f"  Train files: {sorted(train_df[SOURCE_COLUMN].unique())}")
        print(f"  Validation files: {sorted(val_df[SOURCE_COLUMN].unique())}")
        print(f"  Test files: {sorted(test_df[SOURCE_COLUMN].unique())}")

    best_score = float("-inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_metrics['loss']:.4f} "
            f"- val_acc: {val_metrics['accuracy']:.4f} "
            f"- val_precision: {val_metrics['precision']:.4f} "
            f"- val_recall: {val_metrics['recall']:.4f} "
            f"- val_f1: {val_metrics['f1']:.4f}"
        )

        score = float(val_metrics[args.selection_metric])
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "selection_metric": args.selection_metric,
                    "best_score": best_score,
                    "classes": list(label_encoder.classes_),
                },
                args.checkpoint_path,
            )
            print(
                f"  Saved new best checkpoint to {args.checkpoint_path} "
                f"({args.selection_metric}={best_score:.4f})"
            )
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(
                    f"Early stopping triggered after {epoch} epoch(s) "
                    f"without improvement in validation {args.selection_metric}."
                )
                break

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"\nReloaded best checkpoint from epoch {checkpoint['epoch']} "
        f"with best validation {checkpoint['selection_metric']}={checkpoint['best_score']:.4f}"
    )

    final_test_report(
        model=model,
        dataloader=dataloaders["test"],
        loss_fn=loss_fn,
        device=device,
        label_encoder=label_encoder,
        test_dataframe=test_df.reset_index(drop=True),
        predictions_csv=args.predictions_csv,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
