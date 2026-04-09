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
import sys
import json
import time
import queue
import threading
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


# ── Abstract base ─────────────────────────────────────────────────────────────

class EEGSource:
    """Abstract base class for EEG data sources.

    Subclass this to add new sensor types. main_loop.py only calls
    next_reading() and close(), so the swap is one line.
    """

    def next_reading(self):
        """Return (prediction: str, band_scores: dict, confidence: float).

        prediction     — one of 'calm', 'relaxed', 'stressed'
        band_scores    — {band_name: float 0-1} where 0=calm-like, 1=stressed-like
        confidence     — max class probability from predict_proba (0.0–1.0);
                         high values indicate a clean, unambiguous signal
        """
        raise NotImplementedError

    def is_signal_saturated(self):
        """Return True only if confidence is locked for too long."""
        return False

    def update_model(self, true_label):
        """Incrementally update the classifier with a corrected ground-truth label."""
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
        if len(df) == 0:
            raise ValueError("EEG dataset is empty")

        self.df  = df
        self.idx = 0
        self._smooth_buf    = []
        self._smooth_window = 3
        self._last_scaled_row = None
        self._classes = ['calm', 'relaxed', 'stressed']

        self.band_cols = {}
        for band, lo, hi in _BAND_RANGES:
            cols = []
            for c in self.feature_cols:
                if not c.startswith('freq_'): continue
                parts = c.split('_')
                try:
                    val = int(parts[1])
                    if lo <= val <= hi: cols.append(c)
                except: continue
            self.band_cols[band] = cols

        means_path = os.path.join(BASE, '..', 'models', 'class_means.json')
        try:
            with open(means_path, encoding='utf-8') as f:
                self.class_means = json.load(f)
        except:
            self.class_means = None

    def next_reading(self):
        raw = self.df.iloc[self.idx][self.feature_cols].values
        self._smooth_buf.append(raw.copy())
        if len(self._smooth_buf) > self._smooth_window:
            self._smooth_buf.pop(0)
        smoothed = np.median(self._smooth_buf, axis=0)

        # ── EEG Processing (100% Pure EEG) ────────────────────────────────────
        # Apply the pre-fitted StandardScaler (fitted once on training data).
        # partial_fit must NOT be called here — it would drift the scaler's
        # mean/variance at inference time, corrupting the feature distribution.
        row_scaled = self.scaler.transform(smoothed.reshape(1, -1))

        self._last_scaled_row = row_scaled
        prediction  = self.classifier.predict(row_scaled)[0]
        proba       = self.classifier.predict_proba(row_scaled)[0]
        confidence  = float(max(proba))

        smoothed_dict = dict(zip(self.feature_cols, smoothed))
        band_scores   = self._compute_band_scores(smoothed_dict)
        self._last_delta = band_scores.get('Delta', 0.5)

        self.idx = (self.idx + 1) % len(self.df)
        if self.idx == 0:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        if not hasattr(self, '_high_conf_streak'): self._high_conf_streak = 0
        if not hasattr(self, '_last_prediction'): self._last_prediction = None

        if confidence > 0.99 and prediction == self._last_prediction:
            self._high_conf_streak += 1
        else:
            self._high_conf_streak = 0
        self._last_prediction = prediction

        return prediction, band_scores, confidence

    def is_signal_saturated(self):
        high_conf = getattr(self, '_high_conf_streak', 0) >= 25
        return high_conf

    def _compute_band_scores(self, row_dict):
        if self.class_means is None: return {}
        calm_m     = self.class_means.get('calm', {})
        stressed_m = self.class_means.get('stressed', {})
        scores = {}
        for band, cols in self.band_cols.items():
            curr         = [row_dict[c]       for c in cols if c in row_dict]
            calm_vals    = [calm_m.get(c, 0)  for c in cols if c in row_dict]
            stressed_vals = [stressed_m.get(c, 0) for c in cols if c in row_dict]
            if not curr: continue
            current_avg   = sum(curr)          / len(curr)
            calm_mean     = sum(calm_vals)      / len(calm_vals)
            stressed_mean = sum(stressed_vals)  / len(stressed_vals)
            signal_range = stressed_mean - calm_mean
            score = (current_avg - calm_mean) / signal_range if abs(signal_range) > 1e-10 else 0.5
            scores[band] = round(max(0.0, min(1.0, score)), 3)
        return scores

    def update_model(self, true_label):
        if self._last_scaled_row is None: return
        try:
            self.classifier.partial_fit(self._last_scaled_row, [true_label], classes=self._classes)
        except: pass


# ── Arduino Serial EEG Source (Improved with Background Thread) ───────────────

class ArduinoSerialSource(EEGSource):
    """Reads real-time EEG classification from Arduino running mindtune_edge.ino.

    Architecture:
        Starts a background reader thread to drain the serial port instantly.
        Blink events are queued and returned immediately via get_blink_action().
        EEG packets are queued and returned via next_reading().
    """

    _LABEL_MAP  = {0: 'calm', 1: 'relaxed', 2: 'stressed'}
    _BAND_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    _BAUD_RATE  = 115200
    _BOOT_PREFIXES = ('MindTune', 'Format:', 'Classes:', '---', 'CAL')

    def __init__(self, port='auto'):
        try:
            import serial as _serial_mod
            from serial.tools import list_ports as _list_ports
            self._serial_mod = _serial_mod
        except ImportError:
            raise ImportError("pyserial is not installed.")

        if port == 'auto':
            port = self._auto_detect_port()
            if port is None:
                raise RuntimeError("No Arduino detected.")

        self._port = port
        self._ser = _serial_mod.Serial(port, self._BAUD_RATE, timeout=2.0,
                                       rtscts=False, dsrdtr=False)
        print(f"ArduinoSerialSource: connected to {port} @ {self._BAUD_RATE} baud")

        time.sleep(2.0)
        self._ser.reset_input_buffer()

        self._blink_queue = queue.Queue()
        self._eeg_queue   = queue.Queue(maxsize=20)
        self._running     = True
        self._blinks_seen = 0

        self._run_calibration()

        # Start the background thread
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

        self._blink_just_occurred = False
        self._last_prediction     = 'relaxed'
        self._last_delta          = 0.0
        self._high_conf_streak    = 0

    def _reader_thread(self):
        """Continuously reads from serial and populates queues."""
        while self._running:
            try:
                if not self._ser or not self._ser.is_open:
                    time.sleep(0.1)
                    continue
                
                line = self._ser.readline().decode('ascii', errors='ignore').strip()
                if not line: continue

                if line == 'BLINK':
                    print(f"BLINK DETECTED [{time.strftime('%H:%M:%S')}]")
                    self._blink_queue.put('next_track')
                    self._blink_just_occurred = True
                    continue

                if any(line.startswith(p) for p in self._BOOT_PREFIXES):
                    continue

                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        pred_id    = int(parts[0])
                        confidence = float(parts[1])
                        band_scores = {}
                        if len(parts) == 7:
                            powers = [float(p) for p in parts[2:7]]
                            band_scores = self._relative_band_power(powers)
                        
                        try:
                            self._eeg_queue.put_nowait((pred_id, confidence, band_scores))
                        except queue.Full:
                            try: self._eeg_queue.get_nowait()
                            except queue.Empty: pass
                            try: self._eeg_queue.put_nowait((pred_id, confidence, band_scores))
                            except queue.Full: pass   # still full — drop packet, never crash
                    except ValueError:
                        continue
            except Exception as e:
                if self._running:
                    print(f"ArduinoSerialSource: serial error ({e})")
                time.sleep(1)

    def next_reading(self):
        """Returns the latest EEG packet from the queue."""
        try:
            # Check for fresh data (wait up to 1.1s for the 0.5s packet)
            pred_id, confidence, band_scores = self._eeg_queue.get(timeout=1.1)
            
            if self._blink_just_occurred:
                self._blink_just_occurred = False
                return self._last_prediction, band_scores, min(confidence, 0.4)

            prediction = self._LABEL_MAP.get(pred_id, 'relaxed')
            
            if confidence > 0.99 and prediction == self._last_prediction:
                self._high_conf_streak = getattr(self, '_high_conf_streak', 0) + 1
            else:
                self._high_conf_streak = 0
                
            self._last_prediction = prediction
            self._last_delta = band_scores.get('Delta', 0.0)
            return prediction, band_scores, confidence

        except queue.Empty:
            return self._last_prediction, {}, 0.4

    def is_signal_saturated(self):
        return getattr(self, '_high_conf_streak', 0) >= 25

    def _run_calibration(self):
        print("\n" + "=" * 55)
        print("  BLINK CALIBRATION — up to 8s (sit still, then blink)")
        print("=" * 55)
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                line = self._ser.readline().decode('ascii', errors='ignore').strip()
                if line == 'CAL:baseline': print("  Phase 1/2: Sit still...")
                elif line == 'CAL:blink': print("  Phase 2/2: Now BLINK...")
                elif line.startswith('CAL_DONE:'):
                    print(f"  Threshold set.")
                    print("=" * 55 + "\n")
                    return
            except: break
        print("  (Calibration timed out)\n")

    def get_blink_action(self):
        try: return self._blink_queue.get_nowait()
        except queue.Empty: return None

    def inject_blink_spike(self):
        self._blinks_seen += 1
        if self._blinks_seen >= 2:
            self._blink_queue.put('next_track')
            self._blinks_seen = 0

    def simulate_double_blink(self):
        self._blink_queue.put('next_track')

    def _relative_band_power(self, powers):
        total = sum(powers) + 1e-10
        return {name: round(powers[i] / total, 4) for i, name in enumerate(self._BAND_NAMES)}

    def _auto_detect_port(self):
        import serial.tools.list_ports as lp
        for p in lp.comports():
            if any(kw in (p.device or '').lower() for kw in ('usbmodem', 'ttyacm', 'ttyusb')):
                return p.device
        return None

    def close(self):
        self._running = False
        if self._ser: self._ser.close()


# ── Future: BrainFlow ─────────────────────────────────────────────────────────
class BrainFlowSource(EEGSource):
    def next_reading(self):
        raise NotImplementedError("Hardware streaming requires BrainFlow.")
