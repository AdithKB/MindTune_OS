"""EEG source abstraction layer.

In plain English: this module is the bridge between raw brainwave data and the
rest of the system. It reads EEG samples, classifies them as calm/relaxed/stressed,
and returns the result in a standard format so main_loop.py doesn't need to know
whether the data came from a CSV file or a live headset.

Provides a unified interface so main_loop.py works identically
whether reading from a pre-recorded CSV dataset or a live BrainFlow device.

Current implementation: CSVReplaySource (no hardware required).

━━━ Adding live EEG hardware (future) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. pip install brainflow
  2. Uncomment the BrainFlowSource class at the bottom of this file.
  3. In main_loop.py, replace:
         source = CSVReplaySource()
     with:
         source = BrainFlowSource(board_id=22)   # 22 = Muse 2
  4. Record a 10-minute calibration session (5 min calm, 5 min stressed)
     and re-run train_classifier.py on YOUR data — the Kaggle feature
     distributions differ from live hardware readings.

Common BrainFlow board IDs:
    -1  Synthetic board       (software simulation — great for testing)
     0  OpenBCI Cyton         (8-channel, 250 Hz)
     1  OpenBCI Ganglion      (4-channel, 200 Hz)
    21  Muse S                (4-channel, 256 Hz)
    22  Muse 2                (4-channel, 256 Hz)
    38  Muse 2016             (4-channel, 220 Hz)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP = {0: 'calm', 1: 'relaxed', 2: 'stressed'}

# Standard EEG frequency band ranges (freq × 10 = column suffix in Kaggle CSV)
_BAND_RANGES = [
    ('Delta',  1,   30),   # 1–3 Hz   — deep sleep / rest
    ('Theta',  41,  81),   # 4–8 Hz   — drowsy / meditative
    ('Alpha',  91,  132),  # 9–13 Hz  — relaxed alertness (suppressed by stress)
    ('Beta',   142, 304),  # 14–30 Hz — active thinking / stress marker
    ('Gamma',  314, 999),  # 31+ Hz   — intense focus / anxiety
]


# ── Multimodal Fusion: Spotify audio feature helper ───────────────────────────

_AUDIO_DEFAULTS = [0.5, 0.5, 0.5, 0.5, 0.5]   # neutral midpoint for all 5 scalars


def _build_audio_scalars(audio_features):
    """Extract and normalise 5 Spotify audio features for multimodal fusion.

    Returns a list of 5 floats, all in [0, 1]:
        [tempo/200, energy, valence, acousticness, instrumentalness]

    CRITICAL GUARD: If audio_features is None (startup, CSV-only mode, or API
    failure) every scalar defaults to 0.5.  A variable-length feature vector
    would change the shape between ticks and immediately crash StandardScaler.

    Audio features bypass the EEG StandardScaler deliberately: they are already
    [0,1]-normalised and the Kaggle training CSV has no Spotify data (constant
    0.5 defaults → zero variance → scaler would zero-out all live values).
    Tempo is divided by 200 to match the [0,1] range; values above 200 BPM
    are clamped to 1.0 (e.g. drum-and-bass at 174 BPM → 0.87, well within range).
    """
    if not audio_features:
        return list(_AUDIO_DEFAULTS)
    return [
        min(float(audio_features.get('tempo',            100.0)) / 200.0, 1.0),
        float(audio_features.get('energy',           0.5)),
        float(audio_features.get('valence',          0.5)),
        float(audio_features.get('acousticness',     0.5)),
        float(audio_features.get('instrumentalness', 0.5)),
    ]


# ── Abstract base ─────────────────────────────────────────────────────────────

class EEGSource:
    """Abstract base class for EEG data sources.

    Subclass this to add new sensor types. main_loop.py only calls
    next_reading() and close(), so the swap is one line.
    """

    def next_reading(self, audio_features=None):
        """Return (prediction: str, band_scores: dict, confidence: float).

        audio_features — optional dict from get_audio_features() with keys:
                         tempo (BPM int), energy, valence, acousticness,
                         instrumentalness (all floats 0-1).
                         Pass None to use neutral 0.5 defaults for all 5 scalars.
        prediction     — one of 'calm', 'relaxed', 'stressed'
        band_scores    — {band_name: float 0-1} where 0=calm-like, 1=stressed-like
        confidence     — max class probability from predict_proba (0.0–1.0);
                         high values indicate a clean, unambiguous signal
        """
        raise NotImplementedError

    def update_model(self, true_label):
        """Incrementally update the classifier with a corrected ground-truth label.

        Called by main_loop.py when the user rejects a track (👎), signalling
        that the stress prediction that triggered the intervention may have been
        a false positive. Subclasses with an SGDClassifier should override this
        to call partial_fit(); the default is a safe no-op.
        """
        pass

    def close(self):
        """Release any resources (serial port, board session, etc.)."""
        pass


# ── CSV Replay (current default) ──────────────────────────────────────────────

class CSVReplaySource(EEGSource):
    """Replays the pre-recorded Kaggle EEG dataset in a continuous loop.

    Behaviour is identical to the original main_loop implementation.
    No headset required — the CSV simulates a 999-second EEG recording.

    Dataset: kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state
    Classifier: SGDClassifier (log_loss) — supports partial_fit for online personalization.
    """

    def __init__(self):
        self.classifier = joblib.load(
            os.path.join(BASE, '..', 'models', 'classifier.joblib'))
        self.scaler = joblib.load(
            os.path.join(BASE, '..', 'models', 'scaler.joblib'))

        df = pd.read_csv(
            os.path.join(BASE, '..', 'data', 'eeg_mental_state.csv'))
        df['Label'] = df['Label'].map(LABEL_MAP)

        self.feature_cols = [
            c for c in df.columns
            if c != 'Label' and pd.api.types.is_numeric_dtype(df[c])
        ]
        # M-1: guard against an empty or header-only CSV file
        if len(df) == 0:
            raise ValueError(
                "EEG dataset is empty — check data/eeg_mental_state.csv. "
                "Download from: kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state"
            )

        self.df  = df
        self.idx = 0   # loops back when dataset ends

        # Sliding window smoothing: rolling median of last N raw rows.
        # Reduces false stress triggers from single-sample noise spikes.
        # Window of 3 readings = 6 s at the 2-second tick rate (PMC11089529).
        self._smooth_buf    = []
        self._smooth_window = 3

        # Online Personalization: cache the last scaled feature vector so
        # update_model() can call partial_fit() without re-computing the row.
        self._last_scaled_row = None
        self._classes = ['calm', 'relaxed', 'stressed']

        # Map band names to their feature columns
        # M-5: guard against non-integer freq column suffixes (e.g. freq_0.5_0)
        # with a try/except so a different dataset format doesn't crash startup.
        self.band_cols = {}
        for band, lo, hi in _BAND_RANGES:
            cols = []
            for c in self.feature_cols:
                if not c.startswith('freq_'):
                    continue
                parts = c.split('_')
                if len(parts) < 2:
                    continue
                try:
                    val = int(parts[1])
                except ValueError:
                    continue
                if lo <= val <= hi:
                    cols.append(c)
            self.band_cols[band] = cols

        # Per-class means for deviation scoring (written by train_classifier.py)
        means_path = os.path.join(BASE, '..', 'models', 'class_means.json')
        try:
            with open(means_path, encoding='utf-8') as f:
                self.class_means = json.load(f)
            print("CSVReplaySource: class means loaded for band attribution.")
        except Exception:
            self.class_means = None
            print("CSVReplaySource: class_means.json not found — "
                  "run train_classifier.py to enable band attribution.")

    def next_reading(self, audio_features=None):
        raw = self.df.iloc[self.idx][self.feature_cols].values  # (n_features,)

        # Sliding window smoothing: buffer raw rows, compute per-feature median.
        # This smooths out transient noise spikes before classification so a
        # single anomalous reading can't trigger a false 'stressed' prediction.
        self._smooth_buf.append(raw.copy())  # .copy() ensures buffer entry is independent
        if len(self._smooth_buf) > self._smooth_window:
            self._smooth_buf.pop(0)
        smoothed = np.median(self._smooth_buf, axis=0)  # shape: (n_features,)

        # ── Multimodal Fusion ─────────────────────────────────────────────────
        # Step 1: scale EEG features only.  Audio features bypass the scaler
        # because: (a) they're already [0,1]-normalised, and (b) the Kaggle
        # training CSV has no Spotify data so the scaler was never fitted on
        # those columns — including them would produce zero-variance → 0 output.
        eeg_scaled    = self.scaler.transform(smoothed.reshape(1, -1))

        # Step 2: build audio scalars (5 values, all [0,1]) and hstack.
        audio_scalars = np.array(_build_audio_scalars(audio_features)).reshape(1, -1)
        row_scaled    = np.hstack([eeg_scaled, audio_scalars])

        self._last_scaled_row = row_scaled   # cached for update_model()
        prediction  = self.classifier.predict(row_scaled)[0]
        proba       = self.classifier.predict_proba(row_scaled)[0]
        confidence  = float(max(proba))   # highest class probability = signal quality

        # Band scores from smoothed values (used by preference model for EEG features)
        smoothed_dict = dict(zip(self.feature_cols, smoothed))
        band_scores   = self._compute_band_scores(smoothed_dict)

        # ── Dataset Loop & Signal Integrity ───────────────────────────────────
        self.idx = (self.idx + 1) % len(self.df)
        if self.idx == 0:
            # Shuffle on wrap to prevent identical repetition in long sessions.
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            print("CSVReplaySource: dataset wrapped — shuffled for variety.")

        # Signal integrity: if confidence > 0.99 for 10+ consecutive readings,
        # the classifier is locked — likely disconnected electrode or ADC saturation.
        if not hasattr(self, '_high_conf_streak'):
            self._high_conf_streak = 0
        
        # Track streak of identical, very high confidence predictions
        if not hasattr(self, '_last_prediction'):
            self._last_prediction = None

        if confidence > 0.99 and prediction == self._last_prediction:
            self._high_conf_streak += 1
            if self._high_conf_streak == 10:
                print("WARNING: EEG signal may be saturated or electrode disconnected "
                      "(confidence > 99% for 10 consecutive identical readings)")
        else:
            self._high_conf_streak = 0
        
        self._last_prediction = prediction

        return prediction, band_scores, confidence

    def _compute_band_scores(self, row_dict):
        """Return {band: float 0-1} — deviation toward 'stressed' class mean."""
        if self.class_means is None:
            return {}
        calm_m     = self.class_means.get('calm', {})
        stressed_m = self.class_means.get('stressed', {})
        scores = {}
        for band, cols in self.band_cols.items():
            curr         = [row_dict[c]       for c in cols if c in row_dict]
            calm_vals    = [calm_m.get(c, 0)  for c in cols if c in row_dict]
            stressed_vals = [stressed_m.get(c, 0) for c in cols if c in row_dict]
            if not curr:
                continue
            current_avg   = sum(curr)          / len(curr)
            calm_mean     = sum(calm_vals)      / len(calm_vals)
            stressed_mean = sum(stressed_vals)  / len(stressed_vals)
            # signal_range: distance between the calm and stressed class means
            # for this band. Used to normalise the score to [0, 1].
            signal_range = stressed_mean - calm_mean
            score = (current_avg - calm_mean) / signal_range if abs(signal_range) > 1e-10 else 0.5
            scores[band] = round(max(0.0, min(1.0, score)), 3)
        return scores

    def update_model(self, true_label):
        """Incrementally update the SGDClassifier with a corrected ground-truth label.

        Uses the feature vector cached during the most recent next_reading() call,
        so no extra computation is needed at feedback time.

        Args:
            true_label: one of 'calm', 'relaxed', 'stressed'.
                On a 👎 skip, main_loop passes 'relaxed' — a gentle correction
                away from 'stressed' without overcorrecting all the way to 'calm'.
                This is the academically defensible midpoint for an ambiguous signal.
        """
        if self._last_scaled_row is None:
            return
        try:
            self.classifier.partial_fit(
                self._last_scaled_row,
                [true_label],
                classes=self._classes,
            )
            print(f"SGD partial_fit: label='{true_label}' applied to last reading")
        except Exception as e:
            print(f"Warning: partial_fit failed ({e})")


# ── Future: BrainFlow live EEG ────────────────────────────────────────────────
# Uncomment the class below when you have a BrainFlow-compatible headset.
# Install: pip install brainflow
# Docs:    brainflow.readthedocs.io
#
# IMPORTANT — calibration note:
#   The classifier was trained on Kaggle CSV features (processed band-power
#   values from a specific recording setup). Live BrainFlow readings will have
#   different amplitude scales and noise profiles. Before deploying on hardware:
#     1. Record 5 min of calm data + 5 min of intentional stress/focus with
#        your headset using BrainFlow's data_collector.py example.
#     2. Format the recording to match the Kaggle CSV schema (freq_XYZ_C cols).
#     3. Re-run: python src/train_classifier.py --data your_recording.csv
#     4. The new classifier.joblib will work with your hardware's signal profile.
#
class BrainFlowSource(EEGSource):
    """Hardware implementation for Muse/OpenBCI (Future Work)."""
    def next_reading(self, audio_features=None):
        raise NotImplementedError("Hardware streaming requires BrainFlow installation.")

