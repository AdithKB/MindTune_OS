# MindTune-OS Hardware Setup — Plug and Play

**Kit:** Upside Down Labs DIY Neuroscience Kit Pro
**Board:** Arduino UNO R4 Minima
**Sensor:** BioAmp EXG Pill

---

## Quick Start (3 steps)

### Step 1: Wire it up

```
BioAmp EXG Pill          Arduino UNO R4 Minima
───────────────          ─────────────────────
   OUT  ────────────→    A0
   GND  ────────────→    GND
   VCC  ────────────→    5V
```

This matches the [official Upside Down Labs wiring](https://docs.upsidedownlabs.tech/hardware/bioamp/bioamp-exg-pill/index.html).

Electrodes:
- **Signal** → forehead, above left eyebrow (Fp1)
- **Reference** → left earlobe
- **Ground** → behind left ear (mastoid bone)

Clean skin with alcohol wipe before applying electrodes.

### Step 2: Upload the sketch

1. Open Arduino IDE ([arduino.cc/en/software](https://www.arduino.cc/en/software))
2. Install board: **Tools → Boards Manager → "Arduino Renesas UNO R4 Boards"**
3. Select board: **Tools → Board → Arduino UNO R4 Minima**
4. Select port: **Tools → Port** → the one that appeared when you plugged in
5. Open `eeg-music-system/arduino/blink_detector.ino`
6. Click **Upload**

### Step 3: Run MindTune-OS

```bash
cd eeg-music-system
pip install -r requirements.txt
python src/launch.py
```

**That's it.** The system will:
1. Auto-detect your Arduino (no port configuration needed)
2. Auto-calibrate the blink threshold (4-second guided process)
3. Start detecting double-blinks → Spotify track skips

---

## What Happens at Startup

When you run `launch.py` with the Arduino plugged in, you'll see:

```
BlinkDetector: auto-detected Arduino on /dev/cu.usbmodem14101
BlinkDetector: connected to /dev/cu.usbmodem14101 @ 115200 baud

=======================================================
  BLINK CALIBRATION — takes 4 seconds
=======================================================
  Phase 1/2: Sit still, eyes open, DON'T blink...
  Phase 2/2: Now BLINK 3-4 times deliberately...
  Baseline mean=8234, max=8890
  Blink peak=14200
  → Threshold set to 11545
=======================================================
```

If **no Arduino is plugged in**, it falls back to demo mode automatically:

```
BlinkDetector: no Arduino detected — falling back to DEMO MODE.
  (Plug in your Arduino and restart to use live hardware.)
```

No code changes needed either way.

---

## How It Works

```
You blink once       →  "blink #1 (25 samples = 98 ms)"
You blink again      →  "blink #2 (22 samples = 86 ms)"
  within 1 second    →  ACTION: 'next_track' → Spotify skips

You blink once       →  "blink #1"
  wait > 1 second    →  nothing happens (single blinks ignored)
```

**Built-in safety filters:**
- Blinks shorter than 50 ms → rejected (muscle noise)
- Blinks longer than 400 ms → rejected (squint / eye closure)
- Single blinks → ignored (you blink ~15x/min involuntarily)

---

## Edge Inference Sketch (Advanced)

`mindtune_edge.ino` runs the full stress classifier **on the Arduino** — FFT,
band-power extraction, and classification, outputting `"2,0.87\n"` (stressed, 87%
confidence) over Serial.

### Extra setup for this sketch

1. Install **arduinoFFT** library: Sketch → Include Library → Manage Libraries → search "arduinoFFT" by kosme → install v1.9.x
2. Open `eeg-music-system/arduino/mindtune_edge.ino`
3. Upload

### Limitation

The classifier was trained on the Kaggle EEG dataset, not your hardware. Predictions
will work but accuracy requires recording your own calibration data and retraining.
This is documented as a limitation in the report.

---

## Troubleshooting

### Arduino not detected by auto-detect

- Try a different USB-C cable (some are charge-only)
- Check Arduino IDE can see the port under Tools → Port
- On Linux, you may need: `sudo usermod -a -G dialout $USER` (then re-login)

### Calibration says "blink peak not clearly above baseline"

- Electrode contact is poor — re-seat the forehead electrode
- Make sure you blinked **hard and deliberately** during Phase 2
- Try pressing the forehead electrode firmly for better contact

### Signal is all 0 or all 16383

- **All 0:** EXG Pill not powered — check the 5V wire
- **All 16383:** Electrode has no skin contact (open circuit → rail high), or a short

### Too many false triggers during use

- Noisy signal — check earlobe reference electrode contact
- Move electrode cables away from laptop charger
- Unplug charger if possible (eliminates 50/60 Hz mains hum)

### Blinks not detected during use

- Calibration threshold may be too high — restart the app to re-calibrate
- Try blinking more forcefully (exaggerated blinks work better than subtle ones)
