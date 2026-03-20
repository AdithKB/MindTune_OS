/*
 * blink_detector.ino — Raw ADC streamer for EOG blink detection
 * Part of MindTune-OS (github.com/...)
 *
 * PURPOSE
 * -------
 * Reads the raw analog voltage from a single frontal EEG/EOG electrode
 * (Fp1 position, just above the left eyebrow) and streams it over Serial
 * at ~250 Hz.  The Python-side BlinkDetector (src/neuro_apps.py) processes
 * these values to detect intentional double-blinks for track skipping.
 *
 * HARDWARE
 * --------
 *   - Arduino Uno R4 Minima (Renesas RA4M1, Cortex-M4F, 48 MHz, 32 KB SRAM)
 *   - BioAmp EXG Pill (or similar analog EEG frontend)
 *   - Single Ag/AgCl electrode at Fp1 + reference at earlobe
 *   - Connect EXG Pill output → Arduino A0
 *   - Power EXG Pill from the 5V pin (matches Upside Down Labs default wiring)
 *
 * SERIAL FORMAT
 * -------------
 *   One integer per line: the raw 14-bit ADC reading (0–16383).
 *   Example output:  "8192\n", "7800\n", "15000\n"
 *   Non-integer lines (like this boot message) are ignored by the Python parser.
 *
 * BAUD RATE
 * ---------
 *   115200 bps — must match BlinkDetector.BAUD_RATE in neuro_apps.py.
 *
 * SAMPLE RATE
 * -----------
 *   delay(4) gives ~250 Hz (4 ms per sample).  This is close to the 256 Hz
 *   assumed by the blink detection thresholds (13–102 samples = 50–400 ms).
 *   delay() on R4 is clock-cycle calibrated and behaves identically to R3.
 */

void setup() {
    analogReadResolution(14);  // R4 Minima: use full 14-bit ADC (0–16383)
    Serial.begin(115200);
    Serial.println("MindTune-OS Blink Detector v2.0 (R4 Minima, 14-bit ADC)");
}

void loop() {
    Serial.println(analogRead(A0));
    delay(4);  // ~250 Hz sample rate
}
