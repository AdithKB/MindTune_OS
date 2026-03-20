"""neuro_apps.py — MindTune-OS Experimental Neural Applications

In plain English: this module adds two BCI features on top of the existing
Calm Mode, using only a single EEG electrode and no extra sensors.

  FOCUS MODE (FocusMetrics)
  ─────────────────────────
  Monitors the θ/β (theta/beta) brainwave ratio — a well-studied attention
  biomarker. When theta rises above beta (mind wandering), Focus Mode
  triggers instrumental/lo-fi music via Spotify to aid concentration.

  BLINK REMOTE (BlinkDetector)
  ────────────────────────────
  Detects deliberate double-blinks from the raw EOG voltage spike on a single
  frontal electrode (Fp1). Two blinks within 1.0 second → skip current track.
  Single blinks are ignored (involuntary blinks average ~15/min so a single
  blink is almost certainly not deliberate). No arming step needed.

Classes
───────
  FocusMetrics    — θ/β ratio helper. Uses band_scores from eeg_source;
                    no new hardware needed.
  BlinkDetector   — EOG double-blink detector for track skipping via Arduino serial.
                    Dependency: pip install pyserial
                    (or run without hardware using port=None for demo mode)

Quick-start
───────────
  1. pip install pyserial
  2. Find your Arduino port: /dev/cu.usbmodem* on macOS, COM3 on Windows,
     /dev/ttyUSB0 on Linux.
  3. See main_loop_integration_guide() at the bottom of this file for
     the exact lines to add to main_loop.py.
"""

import math
import queue
import threading
import time

# ── Optional imports ───────────────────────────────────────────────────────────
# We import these at the top but catch errors gracefully so that the file can
# be imported even if the optional packages are not yet installed.

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    import serial as _serial_mod
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False


# =============================================================================
# FocusMetrics — θ/β (theta/beta) ratio for attention monitoring
# =============================================================================

class FocusMetrics:
    """Tracks the θ/β ratio to detect when attention is wandering.

    WHY θ/β?
    ────────
    Theta waves (4–8 Hz) are produced by the hippocampus during mind-wandering
    and daydreaming. Beta waves (14–30 Hz) reflect active, alert thinking.
    When theta power exceeds beta power (ratio > ~2.5), the brain is in a
    'default mode' state — attention has slipped away from the task.

    This is the same biomarker used in commercial neurofeedback systems like
    Myndlift and NeuroPeak for ADHD training (Monastra et al., 2005).

    NOTE ON VALUES
    ──────────────
    The band_scores dict from eeg_source.next_reading() contains 0–1 deviation
    scores (distance from calm/stressed class means), not raw microvolt power.
    The θ/β ratio still works as a relative attention proxy because the two
    bands move in opposite directions under inattention (theta up, beta down),
    so the ratio amplifies the signal.

    USAGE (mirrors stress_count pattern in main_loop.py)
    ─────
        focus_metrics = FocusMetrics()

        # In the main loop tick, after eeg_source.next_reading():
        focus_metrics.update(band_scores)
        if focus_metrics.inattention_count() >= 3 and not focus_mode_active:
            # activate focus mode
    """

    INATTENTION_THRESHOLD = 2.5   # θ/β ratio above this → attention is wandering
    HISTORY_LEN           = 5     # rolling window size (matches stress_count window)

    def __init__(self):
        self._history = []   # rolling list of recent θ/β ratios

    def update(self, band_scores):
        """Add the current θ/β ratio to the rolling history.

        Call this once per main-loop tick, right after eeg_source.next_reading().

        Args:
            band_scores: dict like {'Theta': 0.6, 'Beta': 0.2, ...}
                         returned by CSVReplaySource.next_reading()
        """
        ratio = self.theta_beta_ratio(band_scores)
        self._history = (self._history + [ratio])[-self.HISTORY_LEN:]

    @staticmethod
    def theta_beta_ratio(band_scores):
        """Return theta / (beta + epsilon) from a band_scores dict.

        Args:
            band_scores: dict with at minimum 'Theta' and 'Beta' keys.
                         Missing keys default to 0.5 (neutral midpoint).

        Returns:
            float — the θ/β ratio. Values above INATTENTION_THRESHOLD
            indicate attention is drifting.
        """
        theta = band_scores.get('Theta', 0.5)
        beta  = band_scores.get('Beta',  0.5)
        return theta / (beta + 1e-6)   # 1e-6 prevents division by zero

    def inattention_count(self):
        """Return how many of the last HISTORY_LEN readings exceeded the threshold.

        Use this exactly like stress_count in main_loop.py:
            if focus_metrics.inattention_count() >= 3 and not focus_mode_active:
                _enter_focus_mode()   # triggers Spotify instrumental music
                focus_mode_active = True
        """
        return sum(1 for r in self._history if r > self.INATTENTION_THRESHOLD)

    def current_ratio(self):
        """Return the most recent θ/β ratio, or 1.0 if no history yet."""
        return self._history[-1] if self._history else 1.0


# =============================================================================
# BlinkDetector — command-mode EOG blink remote
# =============================================================================

class BlinkDetector:
    """Translates deliberate eye blinks into Spotify actions via Arduino serial.

    HOW EOG BLINK DETECTION WORKS
    ──────────────────────────────
    The cornea of the eye carries a ~100–300 μV electrical potential relative
    to the retina. When you blink, the eyelid sweeps across the cornea, causing
    a large voltage spike visible at electrodes near the eye (Fp1/Fp2 positions
    in the 10-20 EEG system — just above the eyebrows).

    At 256 Hz, a voluntary blink creates 13–100 consecutive samples above a
    threshold (auto-calibrated at startup for BioAmp EXG Pill on R4 Minima 14-bit ADC).

    WHY COMMAND MODE?
    ─────────────────
    People blink involuntarily ~15 times per minute. Reacting to every blink
    would trigger Spotify 15× per minute. The solution: require two blinks
    within 2.5 seconds (a deliberate double-blink is rare involuntarily).
    A single blink is ignored; only the double-blink pattern skips the track.

    BLINK PATTERN
    ─────────────
        2 blinks within 2.5 s  →  next track (skip)

    ARDUINO SETUP
    ─────────────
    The default mindtune_edge.ino outputs predictions ("2,0.87\\n"), not raw ADC.
    Blink detection needs raw ADC values. Upload this minimal sketch to a
    separate Arduino with the Fp1 electrode on pin A0:

        void setup() { Serial.begin(115200); }
        void loop()  { Serial.println(analogRead(A0)); delay(4); }  // ~250 Hz

    DEMO MODE (port=None)
    ─────────────────────
    Pass port=None to run without hardware. A background thread simulates
    a double-blink skip every 30–60 s so you can test the Spotify integration
    without an Arduino.

    USAGE
    ─────
        detector = BlinkDetector(port='/dev/cu.usbmodem1401')
        detector.start()

        # Once per main-loop tick:
        action = detector.get_action()
        if action == 'next_track' and music_active:
            skip_requested = True

        detector.close()   # at session end
    """

    # ── Blink detection thresholds (calibrated for BioAmp EXG Pill at 256 Hz) ──
    # Default for 14-bit ADC (0–16383), 5V reference on R4 Minima.
    # EXG Pill powered from 5V per Upside Down Labs specs.
    # auto_calibrate() overrides this with a measured value at startup.
    ADC_THRESHOLD_DEFAULT = 11200
    ADC_THRESHOLD     = 11200  # runtime value — updated by auto_calibrate()
    BLINK_MIN_SAMPLES = 13     # minimum samples — shorter events are EMG noise (~50 ms)
    BLINK_MAX_SAMPLES = 102    # maximum for a valid blink (~400 ms @ 256 Hz); longer = squint, ignore
    PATTERN_WINDOW_S  = 1.0    # seconds in which a second blink must arrive to trigger skip
    # 1.0 s covers deliberate double-blinks (IBI 200–600 ms + blink2 ~400 ms = ≤1 s).

    BAUD_RATE = 115200
    CALIBRATION_SECONDS = 4    # seconds of baseline + blink data to collect

    def __init__(self, port='auto'):
        """
        Args:
            port: 'auto'  — auto-detect Arduino serial port (plug-and-play).
                  '/dev/cu.usbmodem1401' — explicit port path.
                  None    — demo/simulation mode (no Arduino needed).
        """
        self._port     = port
        self._sim_mode = (port is None)
        self._serial   = None
        self._thread   = None
        self._running  = False

        # ── Blink state machine variables ────────────────────────────────────
        self._in_blink          = False
        self._blink_count       = 0
        self._blinks_seen       = 0
        self._pattern_start     = 0.0
        self._state_lock        = threading.Lock()

        self._action_queue = queue.Queue()

        if self._sim_mode:
            print("BlinkDetector: DEMO MODE (port=None) — "
                  "simulated actions will fire every 20–40 s for testing.")

    # ── Auto-detect Arduino serial port ──────────────────────────────────────

    @staticmethod
    def _auto_detect_port():
        """Scan for an Arduino serial port. Returns port path or None.

        Looks for common Arduino USB identifiers across macOS, Linux, Windows.
        Prefers ports with 'usbmodem' or 'ttyACM' (Arduino R4/R3) in the name.
        """
        if not _SERIAL_OK:
            return None
        try:
            from serial.tools import list_ports
            candidates = []
            for p in list_ports.comports():
                desc = (p.description or '').lower()
                hwid = (p.hwid or '').lower()
                name = (p.device or '').lower()
                # Arduino R4 Minima shows as "USB Serial Device" or has Renesas VID
                is_arduino = any(kw in desc for kw in ('arduino', 'serial', 'usbmodem', 'ttyacm'))
                is_arduino = is_arduino or any(kw in name for kw in ('usbmodem', 'ttyacm', 'ttyusb'))
                is_arduino = is_arduino or '2341' in hwid  # Arduino VID
                is_arduino = is_arduino or '1a86' in hwid  # CH340 (common clone)
                if is_arduino:
                    candidates.append(p.device)
            return candidates[0] if candidates else None
        except Exception:
            return None

    # ── Auto-calibration ─────────────────────────────────────────────────────

    def _auto_calibrate(self):
        """Read ADC samples to learn baseline and set threshold automatically.

        Phase 1 (2s): Sit still → measure baseline noise ceiling (max value).
        Phase 2 (2s): Blink several times → measure blink peak.
        Threshold = midpoint between baseline ceiling and blink peak.

        Falls back to ADC_THRESHOLD_DEFAULT if calibration fails.
        """
        import sys
        hz = 250  # approximate sample rate from blink_detector.ino

        print("\n" + "="*55)
        print("  BLINK CALIBRATION — takes 4 seconds")
        print("="*55)

        # Phase 1: baseline
        print("  Phase 1/2: Sit still, eyes open, DON'T blink...")
        sys.stdout.flush()
        baseline_samples = self._read_adc_samples(self.CALIBRATION_SECONDS // 2, hz)
        if len(baseline_samples) < 50:
            print("  Calibration: not enough samples — using default threshold.")
            return

        baseline_mean = sum(baseline_samples) / len(baseline_samples)
        baseline_max  = max(baseline_samples)

        # Phase 2: blinks
        print("  Phase 2/2: Now BLINK 3-4 times deliberately...")
        sys.stdout.flush()
        blink_samples = self._read_adc_samples(self.CALIBRATION_SECONDS // 2, hz)
        if len(blink_samples) < 50:
            print("  Calibration: not enough samples — using default threshold.")
            return

        blink_peak = max(blink_samples)

        # Compute threshold = midpoint between baseline noise ceiling and blink peak
        if blink_peak <= baseline_max * 1.2:
            # Blinks didn't produce a clear spike — fall back to default
            print(f"  Calibration: blink peak ({blink_peak}) not clearly above "
                  f"baseline ({baseline_max}) — using default threshold {self.ADC_THRESHOLD_DEFAULT}.")
            BlinkDetector.ADC_THRESHOLD = self.ADC_THRESHOLD_DEFAULT
            return

        threshold = int((baseline_max + blink_peak) / 2)
        BlinkDetector.ADC_THRESHOLD = threshold
        self.ADC_THRESHOLD = threshold

        print(f"  Baseline mean={baseline_mean:.0f}, max={baseline_max}")
        print(f"  Blink peak={blink_peak}")
        print(f"  → Threshold set to {threshold}")
        print("="*55 + "\n")

    def _read_adc_samples(self, seconds, approx_hz):
        """Read raw ADC integers from serial for a given duration."""
        samples = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                line = self._serial.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                val = int(line)
                samples.append(val)
            except (ValueError, UnicodeDecodeError):
                continue
            except Exception:
                break
        return samples

    def start(self):
        """Open the serial port (if live) and start the background reader thread.

        If port='auto', scans for Arduino. If found, runs a 4-second calibration
        to set the blink threshold from your actual hardware signal.
        """
        if self._running:
            return

        # ── Auto-detect port if requested ────────────────────────────────────
        if self._port == 'auto':
            if not _SERIAL_OK:
                print("BlinkDetector: pyserial not installed — falling back to DEMO MODE.")
                self._sim_mode = True
                self._port = None
            else:
                detected = self._auto_detect_port()
                if detected:
                    print(f"BlinkDetector: auto-detected Arduino on {detected}")
                    self._port = detected
                    self._sim_mode = False
                else:
                    print("BlinkDetector: no Arduino detected — falling back to DEMO MODE.")
                    print("  (Plug in your Arduino and restart to use live hardware.)")
                    self._sim_mode = True
                    self._port = None

        self._running = True

        if not self._sim_mode:
            if not _SERIAL_OK:
                raise ImportError(
                    "pyserial is not installed.\n"
                    "Run: pip install pyserial"
                )
            self._serial = _serial_mod.Serial(
                self._port, self.BAUD_RATE, timeout=1.0)
            print(f"BlinkDetector: connected to {self._port} @ {self.BAUD_RATE} baud")

            # Wait for Arduino boot (R4 resets on serial open)
            time.sleep(2.0)
            self._serial.reset_input_buffer()

            # Run auto-calibration
            self._auto_calibrate()

            self._thread = threading.Thread(
                target=self._reader_thread,
                daemon=True,
                name='BlinkDetector-reader',
            )
        else:
            self._thread = threading.Thread(
                target=self._sim_thread,
                daemon=True,
                name='BlinkDetector-sim',
            )

        self._thread.start()

    def close(self):
        """Stop the reader thread and release the serial port."""
        self._running = False
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("BlinkDetector: closed")

    def get_action(self):
        """Return the next blink action, or None.

        Call once per main-loop tick. Non-blocking — returns immediately.

        Returns:
            'next_track'  — two blinks detected within PATTERN_WINDOW_S seconds
            None          — no action pending this tick
        """
        try:
            return self._action_queue.get_nowait()
        except queue.Empty:
            return None

    def inject_blink_spike(self):
        """Simulate a single raw EEG voltage spike (EOG).

        Values scaled for 14-bit ADC (0–16383): high=15200 (~93%), low=1600 (~10%).
        """
        for _ in range(25): self._process_sample(15200)
        for _ in range(15): self._process_sample(1600)
        return True

    def simulate_double_blink(self):
        """Simulate a perfect intentional double-blink sequence.
        
        This handles the timing internally to ensure the detection engine
        receives two distinct blinks regardless of network latency.
        """
        print("BlinkDetector: Running automated double-blink simulation...")
        self.inject_blink_spike()
        # 300ms gap — scientifically typical for an intentional double-blink
        time.sleep(0.3)
        self.inject_blink_spike()
        return True
    # ── Private: live hardware reader thread ───────────────────────────────────

    def _reader_thread(self):
        """Reads raw ADC integers from serial, one per line, at ~256 Hz.

        Expected Arduino output format: "8192\\n", "7800\\n", "15000\\n", etc. (14-bit ADC)
        Non-integer lines (e.g. Arduino boot message "MindTune-OS...") are ignored.
        """
        while self._running:
            try:
                line = self._serial.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                try:
                    adc_value = int(line)
                except ValueError:
                    continue   # skip non-integer lines silently
                self._process_sample(adc_value)
            except Exception:
                if self._running:
                    print("BlinkDetector: serial error — reader stopping")
                self._running = False

    def _process_sample(self, adc_value):
        """Run one ADC sample through the blink detection state machine.

        Called ~256 times per second from the reader thread. Fast by design —
        no I/O, no sleeps, no allocations inside the hot path.

        State machine:
            IDLE       → sample > ADC_THRESHOLD  → IN_BLINK (start counting)
            IN_BLINK   → sample > ADC_THRESHOLD  → IN_BLINK (keep counting)
            IN_BLINK   → sample ≤ ADC_THRESHOLD  → IDLE     (classify + reset)
        """
        above = adc_value > self.ADC_THRESHOLD
        call_classify = False
        duration = 0

        with self._state_lock:
            if above and not self._in_blink:
                self._in_blink  = True
                self._blink_count = 1

            elif above and self._in_blink:
                self._blink_count += 1

            elif not above and self._in_blink:
                duration = self._blink_count
                self._in_blink    = False
                self._blink_count = 0
                # Release lock before calling _classify_blink (which re-acquires it)
                call_classify = True

        if call_classify:
            self._classify_blink(duration)

    def _classify_blink(self, duration_samples):
        """Classify a completed blink event by its duration.
        
        Triggers 'next_track' IMMEDIATELY on the second blink within the window.
        """
        now = time.time()

        # Reject noise or squints
        if duration_samples < self.BLINK_MIN_SAMPLES or duration_samples > self.BLINK_MAX_SAMPLES:
            return

        with self._state_lock:
            # If this is the first blink, or the window has expired, reset
            if self._blinks_seen == 0 or self._blinks_seen >= 2 or (now - self._pattern_start > self.PATTERN_WINDOW_S):
                self._blinks_seen = 1
                self._pattern_start = now
            else:
                self._blinks_seen += 1
            
            blinks_so_far = self._blinks_seen

        print(f"BlinkDetector: blink #{blinks_so_far} "
              f"({duration_samples} samples = {duration_samples/256*1000:.0f} ms)")

        if blinks_so_far == 2:
            print("BlinkDetector: ACTION → 'next_track' (double-blink)")
            self._action_queue.put('next_track')
            # M-8: don't reset to 0 instantly, stay at 2 for a moment so 
            # the dashboard's 500ms poll can catch the success state.
            # It will reset on the next tick or after the pattern window.
            # with self._state_lock: self._blinks_seen = 0 (removed instant reset)

    def _commit_pattern(self, pattern_start_at_schedule):
        """No longer used for triggering, but kept for interface compatibility."""
        pass

    # ── Private: demo / simulation thread ─────────────────────────────────────

    def _sim_thread(self):
        """Simulates a double-blink skip every 30–60 s for demo testing.

        Lets you test the skip flow without attaching an Arduino or electrode.
        """
        import random
        while self._running:
            pause = random.uniform(30, 60)
            time.sleep(pause)
            if not self._running:
                break
            print("BlinkDetector [DEMO]: simulated double-blink → 'next_track'")
            self._action_queue.put('next_track')


# =============================================================================
# main_loop.py integration guide
# =============================================================================

def main_loop_integration_guide():
    """Print the minimal changes needed to wire neuro_apps into main_loop.py.

    This is documentation-as-code — run this function to see the integration
    steps printed to the terminal, or read it in the source below.
    """
    guide = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║         main_loop.py Integration Guide — neuro_apps.py          ║
    ╚══════════════════════════════════════════════════════════════════╝

    STEP 1 — Import the classes (add near the top of main_loop.py)
    ──────────────────────────────────────────────────────────────
        from neuro_apps import FocusMetrics, BlinkDetector

    STEP 2 — Instantiate the objects (after sp = get_spotify_client())
    ──────────────────────────────────────────────────────────────────
        focus_metrics  = FocusMetrics()

        # port='auto' → auto-detects Arduino; falls back to demo if not found
        blink_detector = BlinkDetector(port='auto')
        blink_detector.start()

    STEP 3 — Add a focus_mode_active global (near other runtime state)
    ──────────────────────────────────────────────────────────────────
        focus_mode_active = False   # True while Focus Mode music is playing

    STEP 4 — Add to the main loop tick body (after band_scores is set)
    ──────────────────────────────────────────────────────────────────
        # ── Focus Mode: θ/β ratio detection ─────────────────────────
        focus_metrics.update(band_scores)
        inattention = focus_metrics.inattention_count()

        if inattention >= 3 and not focus_mode_active and not music_active:
            # Attention is slipping — trigger instrumental music via Spotify.
            _enter_focus_mode()
            focus_mode_active = True
            print(f"FOCUS MODE ON  | θ/β={focus_metrics.current_ratio():.2f}")

        elif inattention <= 1 and focus_mode_active:
            # Attention restored — exit Focus Mode.
            _exit_focus_mode(reason='improved')
            focus_mode_active = False
            print(f"FOCUS MODE OFF | θ/β={focus_metrics.current_ratio():.2f}")

        # ── Blink Remote: double-blink → skip track ──────────────────
        blink_action = blink_detector.get_action()
        if blink_action == 'next_track' and music_active:
            skip_requested = True
            print("BLINK: double-blink detected — skipping track")

    STEP 5 — Add focus_mode_active to the state snapshot (system_state dict)
    ─────────────────────────────────────────────────────────────────────────
        "focus_mode_active": focus_mode_active,
        "theta_beta_ratio":  round(focus_metrics.current_ratio(), 2),

    STEP 6 — Clean up on exit (in the except KeyboardInterrupt block)
    ──────────────────────────────────────────────────────────────────
        blink_detector.close()
    """
    print(guide)


if __name__ == '__main__':
    main_loop_integration_guide()
