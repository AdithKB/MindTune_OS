"""neuro_apps.py — MindTune-OS Experimental Neural Applications

In plain English: this module adds two new BCI modes on top of the existing
Calm Mode, using only a single EEG electrode and no extra sensors.

  FOCUS MODE (FocusMetrics + BinauralEngine)
  ───────────────────────────────────────────
  Monitors the θ/β (theta/beta) brainwave ratio — a well-studied attention
  biomarker. When theta rises above beta (mind wandering), Spotify is paused
  and 40 Hz binaural beats play instead. 40 Hz is the gamma band frequency
  associated with focused attention and working memory.

  BLINK REMOTE (BlinkDetector)
  ────────────────────────────
  Detects deliberate double-blinks from the raw EOG voltage spike on a single
  frontal electrode (Fp1). Two blinks within 2.5 seconds → skip current track.
  Single blinks are ignored (involuntary blinks average ~15/min so a single
  blink is almost certainly not deliberate). No arming step needed.

Classes
───────
  FocusMetrics    — θ/β ratio helper. Uses band_scores from eeg_source;
                    no new hardware needed.
  BinauralEngine  — 40 Hz stereo binaural beats via sounddevice.
                    Dependency: pip install sounddevice
  BlinkDetector   — EOG double-blink detector for track skipping via Arduino serial.
                    Dependency: pip install pyserial
                    (or run without hardware using port=None for demo mode)

Quick-start
───────────
  1. pip install sounddevice pyserial
  2. Connect stereo headphones (binaural beats need separate L/R channels).
  3. Find your Arduino port: /dev/cu.usbmodem* on macOS, COM3 on Windows,
     /dev/ttyUSB0 on Linux.
  4. See main_loop_integration_guide() at the bottom of this file for
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
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

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
                binaural.start(beat_freq=40)
                focus_mode_active = True
        """
        return sum(1 for r in self._history if r > self.INATTENTION_THRESHOLD)

    def current_ratio(self):
        """Return the most recent θ/β ratio, or 1.0 if no history yet."""
        return self._history[-1] if self._history else 1.0


# =============================================================================
# BinauralEngine — 40 Hz stereo binaural beats
# =============================================================================

class BinauralEngine:
    """Generates and streams 40 Hz binaural beats via sounddevice.

    HOW BINAURAL BEATS WORK
    ────────────────────────
    Each ear hears a slightly different pure tone:
        Left ear:  BASE_FREQ Hz              (default: 200 Hz)
        Right ear: BASE_FREQ + beat_freq Hz  (default: 240 Hz)

    The brain cannot physically hear 40 Hz through air (below the ~20 Hz
    detection threshold), but it 'perceives' the 40 Hz difference between
    the two ears as an internal beat. This is the 'frequency following
    response' — brainwaves tend to synchronise toward the beat frequency.

    Beat frequency guide:
        4–8 Hz  (Theta) → calm, meditative
        8–13 Hz (Alpha) → relaxed alertness
        40 Hz   (Gamma) → focus, working memory (default for Focus Mode)

    ⚠️ REQUIREMENT: Stereo headphones are ESSENTIAL. Speakers mix both
    channels before reaching your ears, so the brain never receives two
    separate frequencies and no beat is perceived.

    USAGE
    ─────
        engine = BinauralEngine()
        engine.start(beat_freq=40)   # begin 40 Hz focus beats
        # ... later ...
        engine.stop()                # stop beats (engine ready to restart)
        engine.close()               # release sounddevice at session end

    DEPENDENCIES
    ────────────
        pip install sounddevice numpy
    """

    SAMPLE_RATE  = 44100   # Hz — standard audio CD quality
    BASE_FREQ    = 200     # Hz — carrier tone (comfortable mid frequency)
    CHUNK_FRAMES = 1024    # samples per sounddevice callback (~23 ms at 44100 Hz)
    VOLUME       = 0.15    # amplitude 0.0–1.0; 0.15 is gentle background level

    def __init__(self):
        if not _NUMPY_OK or not _SD_OK:
            raise ImportError(
                "BinauralEngine requires numpy and sounddevice.\n"
                "Run: pip install numpy sounddevice"
            )
        self._stream      = None    # sd.OutputStream, created fresh on each start()
        self._playing     = False
        self._beat_freq   = 40      # Hz; set by start()
        self._phase_L     = 0.0    # continuous phase accumulator, left channel
        self._phase_R     = 0.0    # continuous phase accumulator, right channel
        self._lock        = threading.Lock()

    def start(self, beat_freq=40):
        """Begin streaming binaural beats.

        Calling start() while already playing is a safe no-op.

        Args:
            beat_freq: Hz difference between left and right ears.
                       40 → gamma (focus), 10 → alpha (calm), 4 → theta (sleep)
        """
        with self._lock:
            if self._playing:
                return
            self._beat_freq = beat_freq
            self._phase_L   = 0.0
            self._phase_R   = 0.0
            self._stream = sd.OutputStream(
                samplerate = self.SAMPLE_RATE,
                channels   = 2,          # stereo: index 0 = left, index 1 = right
                dtype      = 'float32',
                blocksize  = self.CHUNK_FRAMES,
                callback   = self._audio_callback,
            )
            self._stream.start()
            self._playing = True
        print(f"BinauralEngine: started  "
              f"L={self.BASE_FREQ} Hz | R={self.BASE_FREQ + beat_freq} Hz | "
              f"beat={beat_freq} Hz | vol={self.VOLUME}")

    def stop(self):
        """Stop streaming. The engine can be started again with start()."""
        with self._lock:
            if not self._playing:
                return
            self._stream.stop()
            self._stream.close()
            self._stream  = None
            self._playing = False
        print("BinauralEngine: stopped")

    def close(self):
        """Stop and release all resources. Call once at session end."""
        self.stop()

    @property
    def is_playing(self):
        """True while binaural beats are streaming."""
        return self._playing

    # ── Audio callback (runs on sounddevice's internal audio thread) ───────────

    def _audio_callback(self, outdata, frames, time_info, status):
        """sounddevice calls this function every CHUNK_FRAMES samples (~23 ms).

        It fills `outdata` with the next `frames` stereo samples and returns.
        Using phase accumulators (not index-based sin) keeps the tone perfectly
        continuous across chunk boundaries — no audible 'clicks' at seams.

        This function is called from sounddevice's audio thread, NOT the main
        thread. It must never block, allocate memory slowly, or call Python I/O.
        """
        # Angular step per sample = 2π × frequency / sample_rate
        step_L = 2.0 * math.pi * self.BASE_FREQ                   / self.SAMPLE_RATE
        step_R = 2.0 * math.pi * (self.BASE_FREQ + self._beat_freq) / self.SAMPLE_RATE

        # Build sample arrays for both channels.
        # np.arange gives [0, 1, ..., frames-1] so each sample gets the correct
        # phase offset from the running phase accumulator.
        t = np.arange(frames, dtype=np.float32)
        left_wave  = np.sin(self._phase_L + t * step_L).astype(np.float32) * self.VOLUME
        right_wave = np.sin(self._phase_R + t * step_R).astype(np.float32) * self.VOLUME

        outdata[:, 0] = left_wave
        outdata[:, 1] = right_wave

        # Advance phase accumulators and wrap to [0, 2π] to prevent float
        # precision degradation during very long sessions (> several hours).
        self._phase_L = (self._phase_L + frames * step_L) % (2.0 * math.pi)
        self._phase_R = (self._phase_R + frames * step_R) % (2.0 * math.pi)


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
    threshold (~700/1023 ADC counts for a typical BioAmp EXG Pill setup).

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
    ADC_THRESHOLD     = 700    # raw ADC counts above which a sample is 'in a blink'
    BLINK_MIN_SAMPLES = 13     # minimum samples — shorter events are EMG noise (~50 ms)
    BLINK_MAX_SAMPLES = 102    # maximum for a valid blink (~400 ms @ 256 Hz); longer = squint, ignore
    PATTERN_WINDOW_S  = 1.0    # seconds in which a second blink must arrive to trigger skip
    # 1.0 s covers deliberate double-blinks (IBI 200–600 ms + blink2 ~400 ms = ≤1 s).
    # The old value of 2.5 s was for the arming-based design and is too wide here:
    # at 15 involuntary blinks/min (Poisson), a 2.5 s window gives ~46% false-trigger
    # probability per blink; 1.0 s reduces that to ~22 %, which is workable.

    BAUD_RATE = 115200

    def __init__(self, port=None):
        """
        Args:
            port: serial port path, e.g. '/dev/cu.usbmodem1401' (macOS),
                  'COM3' (Windows), '/dev/ttyUSB0' (Linux).
                  Pass None to run in demo/simulation mode (no Arduino needed).
        """
        self._port     = port
        self._sim_mode = (port is None)
        self._serial   = None
        self._thread   = None
        self._running  = False

        # ── Blink state machine variables ────────────────────────────────────
        # These are shared between the reader thread and the timer threads.
        # _state_lock protects all of them from concurrent modification.
        self._in_blink          = False  # True while a blink event is ongoing
        self._blink_count       = 0      # samples seen in the current blink event
        self._blinks_seen       = 0      # valid blinks counted in current pattern window
        self._pattern_start     = 0.0   # time.time() when the first blink in the pattern arrived
        self._state_lock        = threading.Lock()

        # Detected actions (strings) go into this queue.
        # get_action() pops one per tick — safe to call from the main thread.
        self._action_queue = queue.Queue()

        if self._sim_mode:
            print("BlinkDetector: DEMO MODE (port=None) — "
                  "simulated actions will fire every 20–40 s for testing.")

    def start(self):
        """Open the serial port (if live) and start the background reader thread."""
        if self._running:
            return
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
        """Simulate a single raw EEG voltage spike (EOG)."""
        for _ in range(25): self._process_sample(950)
        for _ in range(15): self._process_sample(100)
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

        Expected Arduino output format: "512\\n", "487\\n", "1021\\n", etc.
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
        from neuro_apps import FocusMetrics, BinauralEngine, BlinkDetector

    STEP 2 — Instantiate the objects (after sp = get_spotify_client())
    ──────────────────────────────────────────────────────────────────
        focus_metrics  = FocusMetrics()
        binaural       = BinauralEngine()

        # port=None → demo mode (no Arduino needed for testing)
        blink_detector = BlinkDetector(port=None)
        blink_detector.start()

    STEP 3 — Add a focus_mode_active global (near other runtime state)
    ──────────────────────────────────────────────────────────────────
        focus_mode_active = False   # True while binaural beats are playing

    STEP 4 — Add to the main loop tick body (after band_scores is set)
    ──────────────────────────────────────────────────────────────────
        # ── Focus Mode: θ/β ratio detection ─────────────────────────
        focus_metrics.update(band_scores)
        inattention = focus_metrics.inattention_count()

        if inattention >= 3 and not focus_mode_active and not music_active:
            # Attention is slipping — pause Spotify and play binaural beats.
            # music_active guard prevents Focus Mode competing with Calm Mode.
            try:
                sp.pause_playback()
            except Exception:
                pass   # no active playback — that's fine
            binaural.start(beat_freq=40)
            focus_mode_active = True
            status_message    = f"Focus Mode: 40 Hz binaural beats (θ/β={focus_metrics.current_ratio():.1f})"
            print(f"FOCUS MODE ON  | θ/β={focus_metrics.current_ratio():.2f} | inattention={inattention}/5")

        elif inattention <= 1 and focus_mode_active:
            # Attention restored — resume Spotify, stop binaural beats.
            binaural.stop()
            focus_mode_active = False
            status_message    = "Focus restored — resuming music..."
            try:
                sp.start_playback()
            except Exception:
                pass
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
        binaural.close()
        blink_detector.close()
    """
    print(guide)


if __name__ == '__main__':
    main_loop_integration_guide()
