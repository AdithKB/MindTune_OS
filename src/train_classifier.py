"""Train the main EEG stress classifier for MindTune-OS.

In plain English: this script reads the Kaggle EEG dataset, trains a machine
learning model to tell the difference between calm, relaxed, and stressed
brainwave patterns, and saves the trained model to the models/ directory.
Run this once before starting the main loop — the output files are what the
live system loads at startup.

The script is split into three functions so the high-level steps are obvious:
  load_data()     — read and validate the CSV
  train_model()   — split, scale, add audio features, train, evaluate
  save_results()  — write model files and accuracy log to disk
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, '..', 'data', 'eeg_mental_state.csv')
MODELS_DIR  = os.path.join(BASE, '..', 'models')
MODEL_PATH  = os.path.join(MODELS_DIR, 'classifier.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')

# H-6: create models/ directory if it doesn't exist (e.g. fresh clone, git clean)
os.makedirs(MODELS_DIR, exist_ok=True)

# Label mapping — this dataset uses 0/1/2 numerics; map to strings the rest
# of the system expects. Standard ordering for birdy654/eeg-brainwave-dataset:
LABEL_MAP = {0: 'calm', 1: 'relaxed', 2: 'stressed'}


def load_data():
    """Load the EEG CSV, validate labels, and return (df, feature_cols).

    Prints the first 5 rows and all column names so you can confirm the
    dataset loaded correctly before training begins.
    """
    df = pd.read_csv(DATA_PATH)

    print("=== First 5 rows ===")
    print(df.head())
    print()
    print("=== All column names ===")
    print(list(df.columns))
    print()

    # Map numeric labels to strings
    df['Label'] = df['Label'].map(LABEL_MAP)

    # L-7: verify the mapping actually worked — if the CSV uses string labels
    # ('calm' etc.) instead of numeric labels (0/1/2), map() returns all NaN
    # and train_test_split(..., stratify=y) would fail with a cryptic error.
    null_count = df['Label'].isna().sum()
    if null_count > 0:
        # Try re-mapping with string keys as fallback
        STRING_LABEL_MAP = {'calm': 'calm', 'relaxed': 'relaxed',
                            'concentrating': 'stressed', 'stressed': 'stressed'}
        df['Label'] = pd.read_csv(DATA_PATH)['Label'].map(STRING_LABEL_MAP)
        if df['Label'].isna().sum() > 0:
            raise ValueError(
                f"Label mapping failed: {null_count} NaN labels after mapping. "
                f"Found raw values: {pd.read_csv(DATA_PATH)['Label'].unique()[:10]}. "
                f"Expected numeric {{0,1,2}} or strings {{'calm','relaxed','concentrating'}}."
            )

    print("Label counts after mapping:")
    print(df['Label'].value_counts())
    print()

    # Feature columns = all numeric columns except the label
    feature_cols = [c for c in df.columns
                    if c != 'Label' and pd.api.types.is_numeric_dtype(df[c])]
    return df, feature_cols


def train_model(df, feature_cols):
    """Split data, scale EEG features, append audio placeholders, train, and evaluate.

    Returns (clf, scaler, X_train, y_train, accuracy) for use in save_results().

    Why audio placeholders at 0.5?
    The Kaggle CSV has no Spotify data, so we fill those 5 columns with the
    neutral midpoint (0.5). At runtime, eeg_source.py replaces these with real
    Spotify values — the model has already learned to treat 0.5 as 'unknown audio'.
    """
    X = df[feature_cols]
    y = df['Label']

    # 1. Split FIRST (before any scaling to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Fit scaler on EEG features only, then transform both splits.
    #    Audio features are appended AFTER scaling (see below) — they bypass the
    #    scaler because the Kaggle CSV has no Spotify data.  Including constant 0.5
    #    defaults in the scaler fit would produce zero variance → scale_=0 →
    #    StandardScaler outputs 0 for all live audio values at inference time.
    scaler = StandardScaler()
    X_train_eeg_scaled = scaler.fit_transform(X_train)
    X_test_eeg_scaled  = scaler.transform(X_test)

    # ── Multimodal Fusion: append 5 audio scalars at 0.5 defaults ────────────────
    # These columns mirror the inference path in eeg_source._build_audio_scalars():
    #   [tempo/200, energy, valence, acousticness, instrumentalness]
    # Using 0.5 (neutral midpoint) here means the classifier learns "audio=unknown"
    # during batch training, then receives real Spotify values at runtime.
    _audio_train = np.full((X_train_eeg_scaled.shape[0], 5), 0.5)
    _audio_test  = np.full((X_test_eeg_scaled.shape[0],  5), 0.5)
    X_train_scaled = np.hstack([X_train_eeg_scaled, _audio_train])
    X_test_scaled  = np.hstack([X_test_eeg_scaled,  _audio_test])

    # 3. Train SGDClassifier
    # log_loss gives calibrated predict_proba output (required for Confidence Metrics).
    # partial_fit() is supported natively — no retraining needed for online updates.
    print("Training SGDClassifier (log_loss)...")
    clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # 4. Evaluate
    accuracy = clf.score(X_test_scaled, y_test)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print()

    n_test = X_test_scaled.shape[0]   # return so save_results can log it accurately
    return clf, scaler, X_train, y_train, accuracy, n_test


def save_results(clf, scaler, X_train, y_train, feature_cols, accuracy, n_test):
    """Save the trained model, scaler, accuracy log, and class means to models/.

    Files written:
      models/classifier.joblib  — the trained SGDClassifier
      models/scaler.joblib      — the StandardScaler (EEG features only)
      models/accuracy_log.json  — accuracy metadata for reproducibility
      models/class_means.json   — per-class EEG means (used for band attribution)
    """
    # 4b. Persist accuracy so the figure is verifiable without re-running training
    accuracy_log = {
        'accuracy'        : round(accuracy, 6),
        'accuracy_pct'    : round(accuracy * 100, 2),
        'n_train'         : int(X_train.shape[0]),
        'n_test'          : int(n_test),
        'n_features_eeg'  : len(feature_cols),
        'n_features_total': len(feature_cols) + 5,      # eeg + 5 audio scalars
        'model'           : 'SGDClassifier(loss=log_loss)',
        'test_size'       : 0.2,
        'random_state'    : 42,
        'dataset'         : 'birdy654/eeg-brainwave-dataset-mental-state',
        'trained_at'      : datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    ACCURACY_LOG_PATH = os.path.join(MODELS_DIR, 'accuracy_log.json')
    with open(ACCURACY_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(accuracy_log, f, indent=2)
    print(f"Saved: {ACCURACY_LOG_PATH}")
    print()

    # 5. Save model and scaler
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print()

    # 6. Save per-class means (UNSCALED) for EEG band attribution in main_loop
    #    Only freq_* features (not lag1_*) — used to build brainwave band deviation scores.
    MEANS_PATH = os.path.join(BASE, '..', 'models', 'class_means.json')
    class_means = {}
    for label in ['calm', 'relaxed', 'stressed']:
        mask = y_train == label
        class_means[label] = X_train[mask].mean().to_dict()
    with open(MEANS_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_means, f)
    print(f"Saved: {MEANS_PATH}")
    print()

    # 7. Print feature columns for reference
    print("Feature columns (copy for reference):")
    print(feature_cols)


# ── Run all three steps ───────────────────────────────────────────────────────
df, feature_cols = load_data()
clf, scaler, X_train, y_train, accuracy, n_test = train_model(df, feature_cols)
save_results(clf, scaler, X_train, y_train, feature_cols, accuracy, n_test)
