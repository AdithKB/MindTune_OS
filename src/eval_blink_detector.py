"""Blink Detector Evaluation — synthetic test vector validation.

Feeds synthetic ADC waveforms through BlinkDetector._process_sample()
and verifies that the state machine correctly accepts valid double-blinks
and rejects noise, squints, and single blinks.

No hardware needed — all waveforms are generated in code.
"""

import os, sys, json, time, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuro_apps import BlinkDetector

BASE     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, '..', 'models', 'blink_eval_results.json')

# Suppress demo-mode print noise
import io, contextlib

print("=== Blink Detector Evaluation: Synthetic Test Vectors ===\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_detector():
    """Create a fresh BlinkDetector in port=None mode without starting sim thread."""
    with contextlib.redirect_stdout(io.StringIO()):
        det = BlinkDetector(port=None)
    # Do NOT call det.start() — we feed samples manually
    return det


def feed_samples(det, adc_values):
    """Feed a list of ADC integers through the state machine."""
    for v in adc_values:
        det._process_sample(v)


def drain_actions(det):
    """Return all queued actions as a list."""
    actions = []
    while True:
        a = det.get_action()
        if a is None:
            break
        actions.append(a)
    return actions


def blink_wave(duration_samples, high=15200, low=1600):
    """Generate a single blink: high samples followed by low cooldown.

    Values scaled for 14-bit ADC (0–16383) on Arduino UNO R4 Minima.
    high=15200 (~93% of range), low=1600 (~10% of range).
    """
    return [high] * duration_samples + [low] * 30


def gap_samples(seconds, low=1600):
    """Generate a gap of silence (below-threshold samples) for a given duration.

    At 256 Hz, 1 second = 256 samples.
    low=1600 is well below ADC_THRESHOLD (11200) on 14-bit ADC.
    """
    return [low] * int(256 * seconds)


# ── Test Vectors ──────────────────────────────────────────────────────────────

results = []


def run_test(name, description, expected_actions):
    """Decorator-style test runner."""
    def decorator(fn):
        det = make_detector()
        with contextlib.redirect_stdout(io.StringIO()):
            fn(det)
        actual = drain_actions(det)
        passed = actual == expected_actions
        result = {
            'name':        name,
            'description': description,
            'expected':    expected_actions,
            'actual':      actual,
            'pass':        passed,
        }
        results.append(result)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {description}")
        if not passed:
            print(f"         expected={expected_actions}, actual={actual}")
        return fn
    return decorator


# Test 1: Valid double-blink → next_track
@run_test("valid_double_blink",
          "Two valid blinks (25 samples each, 300ms gap) → next_track",
          ["next_track"])
def _(det):
    feed_samples(det, blink_wave(25))
    feed_samples(det, gap_samples(0.3))
    feed_samples(det, blink_wave(25))
    feed_samples(det, gap_samples(0.1))  # flush


# Test 2: Too-short pulse (noise) → rejected, no action
@run_test("noise_rejection",
          "5-sample pulse (too short, <13) → rejected",
          [])
def _(det):
    feed_samples(det, blink_wave(5))
    feed_samples(det, gap_samples(0.3))
    feed_samples(det, blink_wave(5))
    feed_samples(det, gap_samples(0.1))


# Test 3: Too-long pulse (squint) → rejected, no action
@run_test("squint_rejection",
          "150-sample pulse (too long, >102) → rejected",
          [])
def _(det):
    feed_samples(det, blink_wave(150))
    feed_samples(det, gap_samples(0.3))
    feed_samples(det, blink_wave(150))
    feed_samples(det, gap_samples(0.1))


# Test 4: Single blink only → no action (window expires)
@run_test("single_blink_no_trigger",
          "One valid blink, then 1.5s silence (>1.0s window) → no action",
          [])
def _(det):
    feed_samples(det, blink_wave(25))
    feed_samples(det, gap_samples(1.5))


# Test 5: Mixed — one valid + one noise → no action
@run_test("mixed_valid_noise",
          "One valid blink (25 samples) + one noise (5 samples) → no action",
          [])
def _(det):
    feed_samples(det, blink_wave(25))
    feed_samples(det, gap_samples(0.3))
    feed_samples(det, blink_wave(5))   # noise — rejected
    feed_samples(det, gap_samples(0.1))


# Test 6: Boundary — minimum valid blink duration (13 samples)
@run_test("boundary_minimum_duration",
          "Two blinks at minimum valid duration (13 samples each) → next_track",
          ["next_track"])
def _(det):
    feed_samples(det, blink_wave(13))
    feed_samples(det, gap_samples(0.3))
    feed_samples(det, blink_wave(13))
    feed_samples(det, gap_samples(0.1))


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r['pass'])
failed = sum(1 for r in results if not r['pass'])
total  = len(results)

print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed, {failed} failed")
print(f"{'='*50}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'run_at':     datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'tests':      results,
    'passed':     passed,
    'failed':     failed,
    'total':      total,
    'parameters': {
        'adc_threshold': BlinkDetector.ADC_THRESHOLD,
        'min_samples':   BlinkDetector.BLINK_MIN_SAMPLES,
        'max_samples':   BlinkDetector.BLINK_MAX_SAMPLES,
        'window_s':      BlinkDetector.PATTERN_WINDOW_S,
    },
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {OUT_PATH}")
