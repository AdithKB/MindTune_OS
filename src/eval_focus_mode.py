"""Focus Mode Evaluation — θ/β ratio separation analysis.

Replays the EEG CSV through the same band-score logic used at runtime,
computes the theta/beta ratio for every sample, and reports whether the
ratio meaningfully differentiates mental states.  This is the Focus Mode
equivalent of run_ablation_study.py.

No hardware, no Spotify, no live sessions needed — pure offline analysis.
"""

import os, json, datetime
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, '..', 'data', 'eeg_mental_state.csv')
MEANS_PATH = os.path.join(BASE, '..', 'models', 'class_means.json')
OUT_PATH   = os.path.join(BASE, '..', 'models', 'focus_eval_results.json')

# Must match eeg_source._BAND_RANGES exactly
BAND_RANGES = [
    ('Delta',  1,   30),
    ('Theta',  41,  81),
    ('Alpha',  91,  132),
    ('Beta',   142, 304),
    ('Gamma',  314, 999),
]

LABEL_MAP        = {0: 'calm', 1: 'relaxed', 2: 'stressed'}
STRING_LABEL_MAP = {'calm': 'calm', 'relaxed': 'relaxed',
                    'concentrating': 'stressed', 'stressed': 'stressed'}

THRESHOLD  = 2.5   # must match FocusMetrics.INATTENTION_THRESHOLD
WINDOW     = 5     # must match FocusMetrics.HISTORY_LEN

print("=== Focus Mode Evaluation: θ/β Ratio Separation ===\n")

# ── Load dataset and class means ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Label'] = df['Label'].map(LABEL_MAP)
if df['Label'].isna().sum() > 0:
    df['Label'] = pd.read_csv(DATA_PATH)['Label'].map(STRING_LABEL_MAP)

with open(MEANS_PATH, encoding='utf-8') as f:
    class_means = json.load(f)

feature_cols = [c for c in df.columns
                if c != 'Label' and pd.api.types.is_numeric_dtype(df[c])]

# ── Map columns to bands ─────────────────────────────────────────────────────
band_cols = {}
for band, lo, hi in BAND_RANGES:
    cols = []
    for c in feature_cols:
        if not c.startswith('freq_'):
            continue
        parts = c.split('_')
        try:
            val = int(parts[1])
        except (IndexError, ValueError):
            continue
        if lo <= val <= hi:
            cols.append(c)
    band_cols[band] = cols

# ── Compute band scores and θ/β ratio per sample ─────────────────────────────
calm_m     = class_means.get('calm', {})
stressed_m = class_means.get('stressed', {})

def band_score(row, band):
    cols = band_cols[band]
    if not cols:
        return 0.5
    curr  = sum(row[c] for c in cols) / len(cols)
    calm  = sum(calm_m.get(c, 0) for c in cols) / len(cols)
    stres = sum(stressed_m.get(c, 0) for c in cols) / len(cols)
    rng   = stres - calm
    if abs(rng) < 1e-10:
        return 0.5
    return max(0.0, min(1.0, (curr - calm) / rng))

ratios = []
labels = []
for _, row in df.iterrows():
    theta = band_score(row, 'Theta')
    beta  = band_score(row, 'Beta')
    ratio = theta / (beta + 1e-6)
    ratios.append(ratio)
    labels.append(row['Label'])

ratios = np.array(ratios)
labels = np.array(labels)

# ── Per-class statistics ──────────────────────────────────────────────────────
classes = ['calm', 'relaxed', 'stressed']
stats = {}
for cls in classes:
    mask = labels == cls
    vals = ratios[mask]
    stats[cls] = {
        'mean':   round(float(np.mean(vals)), 4),
        'std':    round(float(np.std(vals)), 4),
        'median': round(float(np.median(vals)), 4),
        'n':      int(mask.sum()),
    }

print(f"{'Class':<12} {'N':>6} {'Mean θ/β':>10} {'Std':>8} {'Median':>8}")
print("-" * 48)
for cls in classes:
    s = stats[cls]
    print(f"{cls:<12} {s['n']:>6} {s['mean']:>10.4f} {s['std']:>8.4f} {s['median']:>8.4f}")

# ── Rolling-window inattention trigger simulation ─────────────────────────────
# Simulate FocusMetrics exactly: rolling window of WINDOW readings, count how
# many exceed THRESHOLD.  Trigger when count >= 3.
print(f"\nSimulating FocusMetrics (threshold={THRESHOLD}, window={WINDOW})...\n")

trigger_counts = {cls: 0 for cls in classes}
total_counts   = {cls: 0 for cls in classes}

history = []
for i in range(len(ratios)):
    history = (history + [ratios[i]])[-WINDOW:]
    inattention = sum(1 for r in history if r > THRESHOLD)
    cls = labels[i]
    total_counts[cls] += 1
    if inattention >= 3:
        trigger_counts[cls] += 1

trigger_rates = {}
print(f"{'Class':<12} {'Triggers':>10} {'Total':>8} {'Rate':>8}")
print("-" * 42)
for cls in classes:
    rate = trigger_counts[cls] / max(total_counts[cls], 1)
    trigger_rates[cls] = round(rate, 4)
    print(f"{cls:<12} {trigger_counts[cls]:>10} {total_counts[cls]:>8} {rate:>8.2%}")

# ── Effect size (Cohen's d between calm and stressed) ─────────────────────────
calm_vals    = ratios[labels == 'calm']
stressed_vals = ratios[labels == 'stressed']
pooled_std   = np.sqrt((np.var(calm_vals) + np.var(stressed_vals)) / 2)
cohens_d     = (np.mean(stressed_vals) - np.mean(calm_vals)) / (pooled_std + 1e-10)
print(f"\nCohen's d (stressed vs calm): {cohens_d:.3f}")
if abs(cohens_d) >= 0.8:
    print("  → Large effect size")
elif abs(cohens_d) >= 0.5:
    print("  → Medium effect size")
else:
    print("  → Small effect size")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'run_at':      datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'dataset':     'birdy654/eeg-brainwave-dataset-mental-state',
    'n_samples':   len(df),
    'threshold':   THRESHOLD,
    'window_size': WINDOW,
    'theta_beta_by_class': stats,
    'inattention_trigger_rate_by_class': trigger_rates,
    'cohens_d_stressed_vs_calm': round(float(cohens_d), 4),
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {OUT_PATH}")
