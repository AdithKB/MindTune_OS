"""Ablation Study — MindTune-OS

In plain English: this script answers the question "does adding Spotify audio
features actually improve the EEG classifier?" It trains the same model three
ways — EEG only, audio only, and both combined — on the same data split, then
compares the accuracy. The result justifies (or challenges) the multimodal
fusion design choice in train_classifier.py and eeg_source.py.

Trains three SGDClassifier configurations on the same 80/20 split to isolate
each feature set's contribution.

Conditions
----------
1. EEG-only      : all numeric CSV features (freq_XXX_C columns + any others)
2. Audio-only    : 5 placeholder audio scalars at 0.5 — expected to be uninformative
3. Multimodal    : EEG + audio (mirrors train_classifier.py exactly)

Scaling follows train_classifier.py:
  - EEG features: StandardScaler fitted on training set only (no leakage)
  - Audio scalars: NOT passed through the scaler (constant 0.5 → zero variance
    would corrupt the scaler; this also matches the inference path in eeg_source.py)

Results are printed as a comparison table and saved to models/ablation_results.json.
"""

import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, '..', 'data', 'eeg_mental_state.csv')
MODELS_DIR = os.path.join(BASE, '..', 'models')
OUT_PATH   = os.path.join(MODELS_DIR, 'ablation_results.json')

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Label mapping — mirrors train_classifier.py exactly ──────────────────────
LABEL_MAP        = {0: 'calm', 1: 'relaxed', 2: 'stressed'}
STRING_LABEL_MAP = {'calm': 'calm', 'relaxed': 'relaxed',
                    'concentrating': 'stressed', 'stressed': 'stressed'}

print("=== MindTune-OS Ablation Study ===")
print(f"Loading: {DATA_PATH}\n")

df = pd.read_csv(DATA_PATH)
df['Label'] = df['Label'].map(LABEL_MAP)
if df['Label'].isna().sum() > 0:
    df['Label'] = pd.read_csv(DATA_PATH)['Label'].map(STRING_LABEL_MAP)
    if df['Label'].isna().sum() > 0:
        raise ValueError(
            "Label mapping failed. Check dataset labels match {{0,1,2}} or "
            "{{'calm','relaxed','concentrating'}}."
        )

# ── Feature extraction — mirrors train_classifier.py ─────────────────────────
feature_cols = [c for c in df.columns
                if c != 'Label' and pd.api.types.is_numeric_dtype(df[c])]
X_eeg = df[feature_cols].values
y     = df['Label']

print(f"Samples  : {len(df)}")
print(f"EEG features : {len(feature_cols)}")
print(f"Label distribution:\n{y.value_counts().to_string()}\n")

# ── Shared 80/20 split — same seed across all three conditions ────────────────
X_tr_eeg, X_te_eeg, y_train, y_test = train_test_split(
    X_eeg, y, test_size=0.2, random_state=42, stratify=y
)

# Audio placeholders: 5 features at 0.5 (mirrors train_classifier.py)
N_AUDIO    = 5
audio_tr   = np.full((X_tr_eeg.shape[0], N_AUDIO), 0.5)
audio_te   = np.full((X_te_eeg.shape[0],  N_AUDIO), 0.5)

# Majority-class baseline (lower bound — a trivial classifier achieves this)
majority_class = y_train.value_counts().idxmax()
majority_acc   = float((y_test == majority_class).mean())

# ── Classifier factory — identical hyperparams across all conditions ───────────
def make_clf():
    return SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

# ── Condition 1: EEG-only ─────────────────────────────────────────────────────
print("Training Condition 1/3: EEG-only ...")
scaler1  = StandardScaler()
X_tr1    = scaler1.fit_transform(X_tr_eeg)
X_te1    = scaler1.transform(X_te_eeg)
clf1     = make_clf()
clf1.fit(X_tr1, y_train)
acc_eeg  = clf1.score(X_te1, y_test)
print(f"  Accuracy: {acc_eeg*100:.2f}%")

# ── Condition 2: Audio-only ───────────────────────────────────────────────────
# All 5 features are constant 0.5 → zero discriminative signal.
# The classifier will converge to predicting the majority class.
# Expected accuracy ≈ majority-class baseline.
print("Training Condition 2/3: Audio-only (5 constant placeholders) ...")
clf2       = make_clf()
clf2.fit(audio_tr, y_train)
acc_audio  = clf2.score(audio_te, y_test)
print(f"  Accuracy: {acc_audio*100:.2f}%")
print(f"  (Majority-class baseline: {majority_acc*100:.2f}% — audio placeholders are uninformative)")

# ── Condition 3: Multimodal ───────────────────────────────────────────────────
# Mirrors train_classifier.py: EEG scaled, audio appended unscaled.
print("Training Condition 3/3: Multimodal (EEG + 5 audio placeholders) ...")
scaler3  = StandardScaler()
X_tr3    = np.hstack([scaler3.fit_transform(X_tr_eeg), audio_tr])
X_te3    = np.hstack([scaler3.transform(X_te_eeg),     audio_te])
clf3     = make_clf()
clf3.fit(X_tr3, y_train)
acc_mm   = clf3.score(X_te3, y_test)
print(f"  Accuracy: {acc_mm*100:.2f}%")

# ── Comparison table ──────────────────────────────────────────────────────────
delta    = acc_mm - acc_eeg
sign     = '+' if delta >= 0 else ''

print()
print("=" * 56)
print(f"{'Ablation Study Results':^56}")
print("=" * 56)
print(f"{'Condition':<30} {'Features':>9} {'Accuracy':>13}")
print("-" * 56)
print(f"{'Majority-class baseline':<30} {'—':>9} {majority_acc*100:>12.2f}%")
print(f"{'Audio-only (placeholders)':<30} {N_AUDIO:>9} {acc_audio*100:>12.2f}%")
print(f"{'EEG-only':<30} {len(feature_cols):>9} {acc_eeg*100:>12.2f}%")
print(f"{'Multimodal (EEG + Audio)':<30} {len(feature_cols)+N_AUDIO:>9} {acc_mm*100:>12.2f}%")
print("=" * 56)
print()
print(f"Audio contribution (Multimodal − EEG-only): {sign}{delta*100:.2f}%")

if abs(delta) < 0.005:
    interp = (
        "Negligible: the 5 placeholder audio scalars add no accuracy over EEG alone. "
        "This is the expected result — all features are constant 0.5 and carry no "
        "discriminative signal. The delta will grow after partial_fit adapts the model "
        "to real Spotify feature distributions at runtime."
    )
elif delta > 0:
    interp = (
        f"Positive (+{delta*100:.2f}%): adding audio features improves accuracy marginally. "
        "Note this is still on constant 0.5 placeholders; the effect may reflect "
        "regularization differences rather than true audio signal."
    )
else:
    interp = (
        f"Negative ({delta*100:.2f}%): the extra audio dimensions slightly hurt accuracy. "
        "This is consistent with adding noise dimensions — the L2 penalty in SGD "
        "has more parameters to regularize against."
    )
print(f"\nInterpretation: {interp}\n")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'run_at'     : datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'dataset'    : 'birdy654/eeg-brainwave-dataset-mental-state',
    'model'      : 'SGDClassifier(loss=log_loss, max_iter=1000, random_state=42)',
    'split'      : {'test_size': 0.2, 'random_state': 42, 'stratified': True},
    'n_train'    : int(X_tr_eeg.shape[0]),
    'n_test'     : int(X_te_eeg.shape[0]),
    'majority_baseline': {
        'class'   : majority_class,
        'accuracy': round(majority_acc, 6),
    },
    'conditions' : {
        'eeg_only': {
            'n_features': len(feature_cols),
            'accuracy'  : round(acc_eeg, 6),
            'accuracy_pct': round(acc_eeg * 100, 2),
        },
        'audio_only': {
            'n_features': N_AUDIO,
            'accuracy'  : round(acc_audio, 6),
            'accuracy_pct': round(acc_audio * 100, 2),
            'note': (
                'All features are constant 0.5 placeholders. '
                'No discriminative signal — accuracy reflects majority-class prediction.'
            ),
        },
        'multimodal': {
            'n_features': len(feature_cols) + N_AUDIO,
            'accuracy'  : round(acc_mm, 6),
            'accuracy_pct': round(acc_mm * 100, 2),
            'note': (
                'EEG features scaled with StandardScaler; audio features appended '
                'unscaled (mirrors train_classifier.py and eeg_source.py inference path).'
            ),
        },
    },
    'audio_contribution': {
        'delta_accuracy'    : round(delta, 6),
        'delta_accuracy_pct': round(delta * 100, 2),
        'interpretation'    : interp,
    },
    'caveats': [
        'Audio scalars are trained on constant 0.5 placeholders — real Spotify values '
        'are only available at runtime via partial_fit. This study measures the structural '
        'contribution of the audio feature dimensions, not real audio content.',
        'EEG features are pre-processed Kaggle spectral columns (freq_XXX_C), not raw '
        'time-domain signals from the Arduino. The StandardScaler bridges distributions '
        'at inference time (proxy-trained, scaler-aligned).',
    ],
}

with open(OUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {OUT_PATH}")
