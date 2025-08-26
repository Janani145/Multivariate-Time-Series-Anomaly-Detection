""" 
You can change the file name if error occurs

Multivariate Time Series Anomaly Detection (Ensemble)
Author:Janani V S
Date: 2025-08-24

Runs Isolation Forest, LSTM Autoencoder, and PCA Reconstruction.
Compares anomaly detection & computes ensemble score + severity distribution.
"""
#--------------IMPORT NECESSARY LIBRARIES-----------
import os
import random
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential

# ---------------- GLOBAL SETTINGS ---------------- #
# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

warnings.filterwarnings("ignore")


# ---------------- DATA PROCESSOR ---------------- #
class DataProcessor:
    """Handles data loading, cleaning, and splitting for anomaly detection."""

    def __init__(self, input_csv_path: str, timestamp_col: str = "Time") -> None:
        """
        Initialize the DataProcessor.

        Args:
            input_csv_path (str): Path to input CSV file.
            timestamp_col (str): Name of the timestamp column.
        """
        self.input_csv_path = input_csv_path
        self.timestamp_col = timestamp_col
        self.df: pd.DataFrame = pd.DataFrame()
        self.train_df: pd.DataFrame = pd.DataFrame()
        self.full_df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> None:
        """Loads and cleans dataset from the input CSV file."""
        if not os.path.exists(self.input_csv_path):
            raise FileNotFoundError(f"Input file not found: {self.input_csv_path}")

        self.df = pd.read_csv(self.input_csv_path)
        if self.timestamp_col not in self.df.columns:
            raise ValueError(f"CSV must contain a '{self.timestamp_col}' column")

        # Convert timestamp column
        self.df[self.timestamp_col] = pd.to_datetime(
            self.df[self.timestamp_col], errors="coerce"
        )
        if self.df[self.timestamp_col].isna().any():
            raise ValueError("Invalid timestamps found in the timestamp column.")

        # Fill missing numeric values with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].median()
        )

        # Sort by time
        self.df.sort_values(self.timestamp_col, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def split_data(self, train_start: str, train_end: str, analysis_end: str) -> None:
        """
        Splits dataset into training and analysis periods.

        Args:
            train_start (str): Start of training period.
            train_end (str): End of training period.
            analysis_end (str): End of analysis period.
        """
        train_mask = (self.df[self.timestamp_col] >= train_start) & (
            self.df[self.timestamp_col] <= train_end
        )
        analysis_mask = (self.df[self.timestamp_col] >= train_start) & (
            self.df[self.timestamp_col] <= analysis_end
        )

        self.train_df = self.df.loc[train_mask].copy()
        self.full_df = self.df.loc[analysis_mask].copy()

        if len(self.train_df) < 72:
            raise ValueError(
                "Insufficient training data: need at least 72 rows in the normal period."
            )


# ---------------- ISOLATION FOREST ---------------- #
class IsolationForestDetector:
    """Isolation Forest anomaly detection."""

    def __init__(self) -> None:
        """Initialize Isolation Forest detector."""
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, timestamp_col: str) -> None:
        """
        Fits Isolation Forest on training data.

        Args:
            df (pd.DataFrame): Training dataframe.
            timestamp_col (str): Timestamp column to exclude.
        """
        self.feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        if timestamp_col in self.feature_names:
            self.feature_names.remove(timestamp_col)

        X = df[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=200, contamination="auto", random_state=42
        )
        self.model.fit(X_scaled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts anomaly scores.

        Args:
            df (pd.DataFrame): Dataframe for inference.

        Returns:
            np.ndarray: Anomaly scores scaled 0–100.
        """
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        raw_scores = -self.model.decision_function(X_scaled)

        # Add noise to avoid identical values
        raw_scores += np.random.normal(1e-6, 1e-6, size=len(raw_scores))
        return np.interp(raw_scores, (raw_scores.min(), raw_scores.max()), (0, 100))


# ---------------- LSTM AUTOENCODER ---------------- #
class LSTMAutoencoderDetector:
    """LSTM Autoencoder for anomaly detection in sequences."""

    def __init__(self, timesteps: int = 10) -> None:
        """
        Initialize LSTM Autoencoder.

        Args:
            timesteps (int): Sequence length for training.
        """
        self.timesteps = timesteps
        self.scaler: StandardScaler = StandardScaler()
        self.model: Sequential | None = None
        self.feature_names: List[str] = []

    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create overlapping sequences for LSTM input.

        Args:
            X (np.ndarray): Scaled feature data.

        Returns:
            np.ndarray: Sequence data.
        """
        return np.array([X[i : i + self.timesteps] for i in range(len(X) - self.timesteps)])

    def fit(self, df: pd.DataFrame, timestamp_col: str) -> None:
        """
        Train LSTM Autoencoder on training data.

        Args:
            df (pd.DataFrame): Training dataframe.
            timestamp_col (str): Timestamp column to exclude.
        """
        self.feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        if timestamp_col in self.feature_names:
            self.feature_names.remove(timestamp_col)

        if len(self.feature_names) == 0:
            raise ValueError("No numeric features found for LSTM Autoencoder.")

        X = df[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)
        X_seq = self.create_sequences(X_scaled)

        self.model = Sequential(
            [
                LSTM(64, activation="relu", input_shape=(X_seq.shape[1], X_seq.shape[2])),
                RepeatVector(X_seq.shape[1]),
                LSTM(64, activation="relu", return_sequences=True),
                TimeDistributed(Dense(X_seq.shape[2])),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X_seq, X_seq, epochs=5, batch_size=32, verbose=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores using reconstruction error.

        Args:
            df (pd.DataFrame): Dataframe for inference.

        Returns:
            np.ndarray: Anomaly scores scaled 0–100.
        """
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        X_seq = self.create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.zeros(len(df))

        X_pred = self.model.predict(X_seq, verbose=0)
        errors = np.mean(np.abs(X_seq - X_pred), axis=(1, 2))

        scores_100 = np.interp(errors, (errors.min(), errors.max()), (0, 100))
        pad = [0] * (len(df) - len(scores_100))
        return np.array(pad + list(scores_100))


# ---------------- PCA RECONSTRUCTION ---------------- #
class PCADetector:
    """PCA-based anomaly detection."""

    def __init__(self, variance: float = 0.95) -> None:
        """
        Initialize PCA Detector.

        Args:
            variance (float): Explained variance for PCA components.
        """
        self.pca: PCA = PCA(n_components=variance)
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, timestamp_col: str) -> None:
        """
        Fit PCA model.

        Args:
            df (pd.DataFrame): Training dataframe.
            timestamp_col (str): Timestamp column to exclude.
        """
        self.feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        if timestamp_col in self.feature_names:
            self.feature_names.remove(timestamp_col)

        X = df[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores based on reconstruction error.

        Args:
            df (pd.DataFrame): Dataframe for inference.

        Returns:
            np.ndarray: Anomaly scores scaled 0–100.
        """
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_proj = self.pca.inverse_transform(X_pca)
        errors = np.mean((X_scaled - X_proj) ** 2, axis=1)

        errors += np.random.normal(1e-6, 1e-6, size=len(errors))
        return np.interp(errors, (errors.min(), errors.max()), (0, 100))


# ---------------- HELPER FUNCTIONS ---------------- #
def get_valid_path(prompt: str, must_exist: bool = True) -> str:
    """Keep asking user until a valid path is provided."""
    while True:
        path = input(prompt).strip('"')  # remove quotes if pasted
        if must_exist and not os.path.exists(path):
            print(f"Path not found: {path}")
        else:
            return path

def get_valid_column(df: pd.DataFrame, prompt: str) -> str:
    """Keep asking user until a valid column name is provided."""
    while True:
        col = input(prompt).strip()
        if col in df.columns:
            return col
        else:
            print(f"Column '{col}' not found in dataset.Available: {list(df.columns)}")

def save_results(
    processor: DataProcessor,
    ensemble_scores: np.ndarray,
    iso_scores: np.ndarray,
    lstm_scores: np.ndarray,
    pca_scores: np.ndarray,
    best_model: str,
    output_path: str,
) -> None:
    """
    Save CSV results for anomaly detection.

    Args:
        processor (DataProcessor): Data processor instance.
        ensemble_scores (np.ndarray): Ensemble anomaly scores.
        iso_scores (np.ndarray): Isolation Forest scores.
        lstm_scores (np.ndarray): LSTM Autoencoder scores.
        pca_scores (np.ndarray): PCA scores.
        best_model (str): Best model based on mean score.
        output_path (str): Output file path for results.
    """
    res = processor.full_df.copy()
    res["Abnormality_Score"] = ensemble_scores


    # Top contributors (variance ranking)
    numeric_cols = processor.full_df.select_dtypes(include=[np.number])
    variances = numeric_cols.var().sort_values(ascending=False).index[:7].tolist()
    for i, col in enumerate(variances, 1):
        res[f"top_feature_{i}"] = col

    res.to_csv(output_path, index=False)

    # Save per-model scores
    res1 = processor.full_df.copy()
    res1["IForest_Score"] = iso_scores
    res1["LSTM_Score"] = lstm_scores
    res1["PCA_Score"] = pca_scores
    res1["Ensemble_Score"] = ensemble_scores
    res1.to_csv("models_score.csv", index=False)


def generate_plots(
    processor: DataProcessor,
    iso_scores: np.ndarray,
    lstm_scores: np.ndarray,
    pca_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    iso_mean: float,
    lstm_mean: float,
    pca_mean: float,
    timestamp_col: str,
) -> None:
    """
    Generate plots for anomaly detection results.

    Args:
        processor (DataProcessor): Data processor instance.
        iso_scores (np.ndarray): Isolation Forest scores.
        lstm_scores (np.ndarray): LSTM scores.
        pca_scores (np.ndarray): PCA scores.
        ensemble_scores (np.ndarray): Ensemble scores.
        iso_mean (float): Mean Isolation Forest score.
        lstm_mean (float): Mean LSTM score.
        pca_mean (float): Mean PCA score.
        timestamp_col (str): Timestamp column name.
    """
    # Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(processor.full_df[timestamp_col], iso_scores, label="Isolation Forest", alpha=0.7)
    plt.plot(processor.full_df[timestamp_col], lstm_scores, label="LSTM Autoencoder", alpha=0.7)
    plt.plot(processor.full_df[timestamp_col], pca_scores, label="PCA Reconstruction", alpha=0.7)
    plt.plot(processor.full_df[timestamp_col], ensemble_scores, label="Ensemble", linewidth=2, color="black")
    plt.xlabel("Time")
    plt.ylabel("Anomaly Score (0-100)")
    plt.title("Anomaly Detection Scores Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_scores_over_time.png")
    plt.close()

    # Bar Plot
    plt.figure(figsize=(8, 5))
    model_means = [iso_mean, lstm_mean, pca_mean, np.mean(ensemble_scores)]
    model_names = ["Isolation Forest", "LSTM Autoencoder", "PCA", "Ensemble"]
    plt.bar(model_names, model_means, color=["green", "blue", "orange", "black"])
    plt.ylabel("Mean Anomaly Score (0-100)")
    plt.title("Comparison of Model Mean Scores")
    plt.tight_layout()
    plt.savefig("model_mean_scores.png")
    plt.close()

    # Severity Pie Chart
    severity_counts = {
        "Normal (0-10)": np.sum(ensemble_scores <= 10),
        "Slightly unusual (11-30)": np.sum((ensemble_scores > 10) & (ensemble_scores <= 30)),
        "Moderate (31-60)": np.sum((ensemble_scores > 30) & (ensemble_scores <= 60)),
        "Significant (61-90)": np.sum((ensemble_scores > 60) & (ensemble_scores <= 90)),
        "Severe (91-100)": np.sum(ensemble_scores > 90),
    }
    plt.figure(figsize=(7, 7))
    plt.pie(severity_counts.values(), labels=severity_counts.keys(), autopct="%1.1f%%", startangle=140)
    plt.title("Severity Distribution (Ensemble Score)")
    plt.tight_layout()
    plt.savefig("severity_distribution.png")
    plt.close()

    # --- Overlay plot for Top 7 features ---
def plot_top7_features(processor: DataProcessor, ensemble_scores: np.ndarray, timestamp_col: str) -> None:
        """
        Plot top 7 features (by variance) along with ensemble anomaly score.
        """
        numeric_cols = processor.full_df.select_dtypes(include=[np.number])
        top7_features = numeric_cols.var().sort_values(ascending=False).head(7).index.tolist()

        plt.figure(figsize=(14, 6))

        # Plot each feature (scaled to 0-100 for comparison)
        for feature in top7_features:
            values = processor.full_df[feature].values
            scaled_values = np.interp(values, (values.min(), values.max()), (0, 100))
            plt.plot(processor.full_df[timestamp_col], scaled_values, label=feature, alpha=0.7)

        # Overlay ensemble score
        plt.plot(processor.full_df[timestamp_col], ensemble_scores, label="Ensemble Score", color="black", linewidth=2)

        plt.xlabel("Time")
        plt.ylabel("Scaled Value / Anomaly Score (0-100)")
        plt.title("Top 7 Feature Trends vs Ensemble Anomaly Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("top7_features_vs_ensemble.png")
        plt.close()

# --- Bar plot for Top 7 features by variance ---
def plot_top7_features_bar(processor: DataProcessor) -> None:
    """
    Plot top 7 features by variance as a bar chart.
    """
    numeric_cols = processor.full_df.select_dtypes(include=[np.number])
    top7_features = numeric_cols.var().sort_values(ascending=False).head(7)

    plt.figure(figsize=(10, 6))
    plt.bar(top7_features.index, top7_features.values, color="skyblue")
    plt.ylabel("Variance")
    plt.title("Top 7 Features by Variance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("top7_features_bar.png")
    plt.close()



# ---------------- MAIN ---------------- #
def main(input_csv_path: str, output_path: str) -> None:
    """
    Main pipeline for anomaly detection.

    Args:
        input_csv_path (str): Path to input CSV file.
        output_path (str): Path to save results.
    """
    TRAIN_START = "2004-01-01 00:00:00"
    TRAIN_END = "2004-01-05 23:59:00"
    ANALYSIS_END = "2004-01-19 07:59:00"


    try:
        print("Loading data...")
        processor = DataProcessor(input_csv_path, timestamp_col)
        processor.load_data()
        processor.split_data(TRAIN_START, TRAIN_END, ANALYSIS_END)

        print("Training Isolation Forest...")
        iso = IsolationForestDetector()
        iso.fit(processor.train_df, timestamp_col)
        iso_scores = iso.predict(processor.full_df)

        print("Training LSTM Autoencoder...")
        lstm = LSTMAutoencoderDetector()
        lstm.fit(processor.train_df, timestamp_col)
        lstm_scores = lstm.predict(processor.full_df)

        print("Training PCA Reconstruction...")
        pca = PCADetector()
        pca.fit(processor.train_df, timestamp_col)
        pca_scores = pca.predict(processor.full_df)

        # Ensemble
        ensemble_scores = (iso_scores + lstm_scores + pca_scores) / 3

        # Model comparison
        iso_mean, lstm_mean, pca_mean = np.mean(iso_scores), np.mean(lstm_scores), np.mean(pca_scores)
        best_model = max(
            {"IsolationForest": iso_mean, "LSTM": lstm_mean, "PCA": pca_mean},
            key=lambda k: {"IsolationForest": iso_mean, "LSTM": lstm_mean, "PCA": pca_mean}[k],
        )
        print("Best model is:", best_model)
        # Save results
        save_results(processor, ensemble_scores, iso_scores, lstm_scores, pca_scores, best_model, output_path)
        print("All results saved.")

        # Plots
        print("Generating plots...")
        generate_plots(processor, iso_scores, lstm_scores, pca_scores, ensemble_scores, iso_mean, lstm_mean, pca_mean, timestamp_col)
        plot_top7_features(processor, ensemble_scores, timestamp_col)
        plot_top7_features_bar(processor)
        print("Plots saved successfully.")




        plot_top7_features_bar(processor)


    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Input validation for file paths
    input_csv_path = get_valid_path("Enter input CSV path: ", must_exist=True)
    output_path = get_valid_path("Enter output CSV path (folder or file): ", must_exist=False)

    # Load CSV temporarily just to validate timestamp column
    temp_df = pd.read_csv(input_csv_path)
    timestamp_col = get_valid_column(temp_df, "Enter timestamp column name in your dataset: ")

    # Run main

    main(input_csv_path, output_path)
