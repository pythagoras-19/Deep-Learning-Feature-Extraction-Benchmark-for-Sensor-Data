# Deep-Learning-Feature-Extraction-Benchmark-for-Sensor-Data
This project benchmarks multiple deep learning architectures for feature extraction and classification of sensor datasets. Models evaluated include MLP, CNN, and LSTM networks trained on network telemetry data.

## `train_model.py`
`train_model.py` trains a simple feedforward PyTorch classifier on the Tuesday CICIDS-style traffic CSV included in this repository.

The script:
- Locates the `Tuesday-WorkingHours.pcap_ISCX.csv` dataset from the repo's local `data/` directories
- Loads the CSV with pandas and normalizes the column names
- Keeps numeric feature columns plus the `Label` target column
- Removes rows with missing or infinite values
- Encodes the string labels into integer class IDs
- Standardizes the feature matrix with `StandardScaler`
- Splits the data into training and test sets
- Trains a small two-layer neural network for 10 epochs using Adam and cross-entropy loss
- Evaluates the model on the test split and prints the final accuracy

# Findings
The trained MLP model achieves around 99% accuracy on the test set, demonstrating that even a simple feedforward architecture can effectively classify the network traffic data when properly preprocessed. This serves as a strong baseline for future experiments with more complex models like CNNs and LSTMs on this dataset.

# Plan
I plan on splitting the train and test set between the different days of the week, so that the model is trained on one day and tested on another. This will help evaluate the model's ability to generalize to unseen data distributions, as network traffic patterns can vary significantly across different days. I want to bring in Wednesday's data for testing, while training on Tuesday's data. This will provide a more realistic assessment of the model's performance in real-world scenarios where it may encounter new traffic patterns.
## Findings
A simple MLP trained on one day of CIC IDS traffic does not generalize well to a different day containing substantially different attack types.
# Plan
Train on multiple days of the week to improve generalization, and then evaluate on a held-out day. This will help the model learn more robust features that are not specific to a single day's traffic patterns.

## Interpretation of Multi-Day Binary Classification Results

Although the model was trained on multiple days of traffic (Monday, Tuesday, Thursday, and Friday), performance on the held-out Wednesday dataset remained poor for the ATTACK class. In particular, ATTACK recall was approximately 1%, indicating that the model failed to generalize to most malicious flows in the Wednesday distribution.

This suggests that collapsing diverse attack families into a single binary ATTACK label did not produce a sufficiently coherent class for the MLP to learn robust cross-day features. Instead, the model primarily learned to recognize BENIGN traffic, resulting in high BENIGN recall but very weak attack detection.

These findings indicate that cross-day intrusion detection is substantially more difficult than same-day random-split evaluation and that stronger modeling, sampling, or anomaly-detection approaches may be needed to improve generalization.