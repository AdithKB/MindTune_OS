# MindTune-OS — Demo Script
**MSc Foundations of AI | NCI | April 2026**

---

## Pre-Demo Setup

1. **Start the System:**
   ```bash
   cd eeg-music-system
   make run
   ```
2. **Enable Debug Mode (CRITICAL):**
   The browser will open automatically to the standard dashboard. **You must manually append `?debug` to the URL** in the address bar (e.g., `http://127.0.0.1:5050?debug`) and hit Enter.
   *   **Why?** The `🔥 Force Stress` button is a developer tool hidden from regular users. Appending this flag reveals the button so Student 3 can trigger the intervention on command.
3. **Spotify Check:**
   Open Spotify in a second window. Ensure a song is active (playing or paused) on your device. If the dashboard says "Waiting for Spotify," play a song for 2 seconds to "wake up" the API.
4. **History Safety:**
   **DO NOT run `make clean`** — it wipes the pre-seeded feedback history and the 400+ entries in the wins log.

---

## Student Assignments

| Order | Student | Section | Time |
|---|---|---|---|
| 1st | Student 1 | The Signal | ~90s |
| 2nd | Student 2 | The Brain | ~90s |
| 3rd | Student 3 | The Trigger | ~120s |
| 4th | Student 4 | The Loop | ~90s |

---

## Student 1 — The Signal

### Script
> "MindTune-OS is a closed-loop system that reads brainwaves and adapts music in real time. At its core it uses **Electroencephalography** — EEG — which measures the electrical activity produced by neurons firing in the brain as microvolt-level voltages on the scalp. In our hardware setup, these signals are amplified by the **BioAmp EXG Pill** analog front-end and digitised by the **Arduino UNO R4**, which acts as an **Analog-to-Digital Converter**.
>
> The digitised signal is decomposed into five physiological frequency bands — you can see them live in this Band Strip. **Delta**, 0.5 to 4 Hz, associated with deep recovery and sleep. **Theta**, 4 to 8 Hz, linked to drowsiness and creative states. **Alpha**, 8 to 13 Hz, the signature of relaxed wakefulness — this is the band that drops when you're stressed. **Beta**, 13 to 30 Hz, which rises during active thinking and stress. And **Gamma**, above 30 Hz, present during high cognitive load.
>
> Right now we're in **CSV Replay Mode** — because we don't have the headset connected, the system is streaming pre-recorded data from a labelled mental state dataset through the exact same processing pipeline as live hardware. The architecture doesn't change; only the data source does. And even in replay mode, the system is interactive — I can tap the **⚡ Skip** button to manually override the AI, simulating our **Blink Remote**: a deliberate double-blink on the real headset is detected by a threshold classifier and triggers the same skip action."

### What to point at
| Element | What it is |
|---|---|
| Band Strip (Delta → Gamma) | Normalised band power, updated every 500 ms |
| EEG chart | Rolling brainwave trace |
| ⚡ Skip button (topbar) | Manual override / blink remote simulation |

### Technical terms
- **Electroencephalography (EEG)** — scalp-recorded electrical field potentials from neural activity
- **Microvolt amplification** — BioAmp EXG Pill boosts µV signals to a readable ADC range
- **ADC** — Arduino UNO R4 digitises the analog signal at discrete time intervals
- **Frequency band decomposition** — FFT-based power extraction into physiological bands
- **CSVReplaySource** — identical pipeline to `ArduinoSerialSource`, deterministic for testing
- **Duration-based heuristic (Blink Remote)** — 13-sample min (rejects EMG noise spikes), 102-sample max (rejects sustained squints), 1.0s inter-blink window; signal processing, not ML

### Professor Q&A
**"How do you validate that CSV replay matches hardware behaviour?"**
> Both `CSVReplaySource` and `ArduinoSerialSource` implement the same Python interface — they return the same 5-band normalised dictionary every tick. Every downstream component (classifier, scaler, intervention logic) is identical regardless of source. The difference is only where the numbers come from, not how they are processed.

**"What about noise and EMG artifacts in the EEG?"**
> Single-channel prefrontal EEG is susceptible to EMG (muscle) and EOG (eye) artifacts. We don't apply explicit artifact rejection — instead the stress count uses a 5-tick rolling window, so one noisy reading cannot trigger an intervention on its own. For the blink remote, the detector enforces a 13-sample minimum and 102-sample maximum duration, rejecting noise spikes and sustained squints. This was validated with 6/6 unit tests in `blink_eval_results.json`.

**"Why is TinyML only 74% accurate when your Python model is 92.94%?"**
> The TinyML model uses only 5 features — the five band-power scalars — because the Arduino has limited flash and no floating-point matrix libraries at scale. The full Python classifier uses 988 pre-computed spectral features from the Kaggle dataset. Fewer features means lower accuracy. 74% is sufficient for on-device pre-filtering and runs in microseconds without any USB round-trip.

**"What is the sampling rate of the hardware?"**
> The BioAmp EXG Pill outputs an analog signal; the Arduino samples it and streams numeric readings over USB serial. The on-device FFT in `mindtune_edge.ino` uses a 128-point window. The Python layer processes one classification tick per second — the EEG pipeline is designed around 1 Hz state updates, not raw sample-level event locking.

---

## Student 2 — The Brain

### Script
> "Every second, those five normalised band scores form a **feature vector** that feeds our classifier. But before any numbers enter the model, they pass through a **StandardScaler** — z-score normalisation: subtract the mean, divide by the standard deviation. This is not optional. Two of our features are ratio-based: α/β and θ/β. When Beta is nearly zero, those ratios can reach values of 100 or more. Without normalisation, the SGD coefficients for those features balloon to ±600 and completely override the five band scores the model is supposed to learn from. The comment in `preference_model.py` at line 342 documents this explicitly.
>
> The base classifier is an **SGDClassifier** — Stochastic Gradient Descent with Log-loss, trained on 1,983 samples from a labelled EEG mental state dataset, achieving **92.94% accuracy** on a 20% held-out test set. We also ran a full **ablation study**: EEG-only reaches 91.94%, while audio features alone score 33.47% — barely above the 33.5% majority-class baseline, which means audio features at training time carry zero discriminative signal. That result directly validates our **Pure EEG design**.
>
> The personalisation model runs in two phases. **Phase 1** uses **Laplace-smoothed** win-rate counting: score = (positive + 1) / (positive + negative + 2). The +1 and +2 prevent the model from being overconfident on a single data point — a query seen once with a positive result would score 1.0 without smoothing. Phase 1 runs from the very first feedback entry with no training needed.
>
> Once we hit 10 entries, the system transitions to **Phase 2** — which you can see active now [point to badge]. The feature vector expands to 7 EEG features: the 5 bands plus α/β and θ/β ratio — both validated stress biomarkers from PMC9749579 — plus a one-hot vector of Last.fm genre tags. The model is `SGDClassifier(loss='log_loss')`, equivalent to Logistic Regression but supporting `partial_fit` for O(1) incremental updates. After each feedback press, if the tag vocabulary hasn't changed, the model does a single `partial_fit` call on just the new entry — milliseconds, not seconds.
>
> [Open Patterns Learned dropdown.] From 10 prior entries the model has learned that Ambient and Classical genres received positive feedback for this user, while Lo-Fi and Hip Hop did not. The EEG weight panel shows which bands the model considers most predictive for this specific person."

### What to point at
| Element | What it is |
|---|---|
| Phase 2 badge | Confirms SGD model is active (`MIN_SAMPLES = 10` reached in `preference_model.py`) |
| Stress Gauge (0–5) | Count of 'stressed' in the last 5 ticks — `recent_predictions.count('stressed')` |
| Patterns Learned dropdown | Tag-weighted preference map + EEG band weights from `coef_` |
| CLI / ADI metrics | θ/β (DASM cognitive load proxy) and α/(β+γ) (Davidson 1988) |

### Technical terms
- **Feature vector** — `[Delta, Theta, Alpha, Beta, Gamma, α/β, θ/β, tag_0 … tag_n]`
- **z-score normalisation (StandardScaler)** — zero mean, unit variance; critical for SGD with ratio features
- **SGDClassifier / Log-loss** — logistic regression trained via stochastic gradient descent
- **Stochastic optimisation** — `partial_fit` is a direct implementation of SGD, the foundational optimisation algorithm behind most modern AI
- **92.94% accuracy** — 1,983 train / 496 test, stratified split, `random_state=42`
- **Ablation study** — EEG-only 91.94% vs Audio-only 33.47% (= majority baseline); validates Pure EEG
- **Laplace smoothing** — `(pos+1)/(pos+neg+2)`; prevents zero-probability estimates on sparse data
- **`partial_fit`** — O(1) incremental SGD update; only full-refit when tag vocabulary grows
- **Phase 1 → Phase 2 threshold** — `MIN_SAMPLES = 10` in `preference_model.py` line 72
- **CLI (θ/β)** — Cognitive Load Index, established DASM proxy for cognitive workload
- **ADI (α/(β+γ))** — Alpha Dominance Index, Davidson 1988 approach motivation proxy

### Professor Q&A
**"What are the 988 features in your main classifier?"**
> The Kaggle dataset provides pre-computed spectral columns (`freq_XXX_C`) — already-processed frequency-domain measurements across many sub-bands. The Python classifier is trained on those 988 columns. At inference time on the Arduino, 5 band-power scalars are computed via FFT, and a StandardScaler bridges the two distributions. This mismatch is an acknowledged limitation documented in the README as "proxy-trained, scaler-aligned."

**"Why SGD and not a Random Forest or neural network?"**
> `SGDClassifier` supports `partial_fit` — O(1) incremental weight update per feedback event. A Random Forest or neural network requires full retraining on every 👍/👎 press, which is O(n). Online learning means the model updates in milliseconds per feedback without touching historical data.

**"What if the user only ever gives positive feedback?"**
> Line 337–339 of `preference_model.py`: `if len(set(y)) < 2: return`. The model refuses to initialise until both classes — positive (1) and negative (0) feedback — are present. Until that condition is met it stays in Phase 1. The pre-seeded log has 6 positive and 4 negative entries, so this is already satisfied.

**"Why 7 EEG features in Phase 2 and not just the 5 bands?"**
> The feature vector appends two ratio features: α/β (low under stress) and θ/β (high under stress), both validated biomarkers from PMC9749579. They're clamped to `[0, 100]` to prevent Beta ≈ 0 producing ratios of 1e6. Placing them after the 5 bands means `coef_[:5]` still indexes bands cleanly in the insights panel.

**"What does 33.47% audio-only accuracy actually mean?"**
> The majority class (relaxed) represents 33.5% of the dataset. A classifier that always predicts "relaxed" scores 33.5%. Audio-only scores 33.47% — statistically indistinguishable from that baseline. The five Spotify audio scalars are all 0.5 in training because the Kaggle dataset has no Spotify metadata. The ablation proves they add nothing, which is why the system is Pure EEG.

**"How do you handle class imbalance?"**
> Two layers. In Phase 1, Laplace smoothing handles sparse data — every class starts with a prior of 1 so no tag or query is assigned a probability of zero from limited observations. In Phase 2, `SGDClassifier` with `log_loss` outputs calibrated probabilities rather than raw vote counts, which is more robust to imbalance than accuracy-maximising classifiers. Additionally, Phase 2 won't initialise until both positive and negative examples exist (`len(set(y)) < 2` guard, line 337 of `preference_model.py`), so the model never trains on a single-class dataset.

### Transition to Student 3
> "Now that the brain is decoded and the model is personalised — Student 3 will show what the system actually *does* when it detects stress."

---

## Student 3 — The Trigger

### Pre-condition check
> ⚠️ Before clicking anything — look at the **Goal Strip**:
> - **"Calm Mode: Monitoring for stress signals..."** → clear to proceed
> - **"Playing: [track name]"** → music already active; click 👎 first, wait 2 seconds, then proceed

### Script
> "Let me show the system responding in real time. The Goal Strip confirms we're monitoring. I'll simulate a stress spike using our debug tool. [Click 🔥 Force Stress.]
>
> What just happened in the code: `main_loop.py` received a `force_stress` feedback signal and set `recent_predictions = ['stressed'] * 5` — overwriting all five slots in the rolling window. The stress count is `recent_predictions.count('stressed')`, so it jumps to 5/5 instantly. The trigger condition is `stress_count >= 3`, satisfied immediately.
>
> Watch the Goal Strip — it's just flipped to **'Stress Peak Detected — selecting music...'** The system is now executing the intervention chain. First it tries **Strategy 1**: Last.fm's similarity API, seeded with artist names from historical wins to find related tracks. If that returns nothing, it escalates to **Strategy 2**: the Groq API running **Llama-3.3-70b-versatile**.
>
> The Groq prompt passes the current EEG readings, the last 5 wins, the last 3 failures, and the last 10 queries already tried this session. The model is told to respond in exactly two lines — `QUERY:` and `REASON:` — and the parser strips any markdown formatting the LLM adds. If the response doesn't match the format, the system falls back to a rotating list of 10 clinically-informed queries including 'Weightless' by Marconi Union — a track used in peer-reviewed stress-reduction trials.
>
> [Point to AI Reasoning panel.] That text is the LLM's live output — its justification for why it chose this query. It's performing **zero-shot contextual reasoning** over 500+ logged entries — 190+ confirmed wins — to explain what worked for this brain before and why this new choice follows that pattern."

### What to point at
| Element | What it is |
|---|---|
| Goal Strip | Flips from "Calm Mode: Monitoring..." to "Stress Peak Detected — selecting music..." |
| Stress Gauge | Hits 5/5 immediately after Force Stress |
| AI Reasoning panel | Groq LLM output — open by default |
| Spotify window | Track changes within a few seconds |

### Technical terms
- **Rolling prediction window** — `recent_predictions[-5:]`; `stress_count = recent_predictions.count('stressed')`
- **Trigger condition** — `stress_count >= 3` (`main_loop.py` line 590)
- **Strategy 1** — Last.fm similar-artist API seeded from `wins_log.json` win artists
- **Strategy 2** — Groq API, Llama-3.3-70b-versatile, structured prompt, max 100 tokens
- **Zero-shot contextual reasoning** — no fine-tuning; reliability comes from structured prompt + win/fail history
- **Fallback hierarchy** — Last.fm → LLM query → `_FALLBACK_QUERIES` in `agent.py` (10 rotating entries)
- **Rate limit gate** — `_spotify_blocked_until` timestamp; 429 responses return instantly with no sleep
- **500+ entries / 190+ wins** — `wins_log.json` written atomically by `save_track_to_wins()`

### Professor Q&A
**"What does the Groq prompt look like?"**
> Constructed in `agent.py` lines 76–127. Five sections: (1) mode declaration (CALM or FOCUS), (2) current EEG reading as a list of prediction strings, (3) queries already tried this session (last 10, to prevent repeats), (4) past wins summary (last 5 from `wins_log.json`), (5) past fails (last 3). Output is constrained to `QUERY: <3–6 words>` and `REASON: <one sentence>`. The parser uses case-insensitive regex and strips `**` markdown bold that the LLM sometimes adds.

**"Isn't zero-shot unreliable? What if it hallucinates a track name?"**
> The LLM output is a **Spotify search query**, not a track name — so hallucination doesn't cause a crash. "ambient piano stress relief" produces real search results from Spotify's catalogue regardless of whether those exact words came from training data or were invented. The structural output validation (QUERY: / REASON: format check) and the 10-query fallback list mean the system always produces a playable result.

**"Why Llama-3.3-70b specifically?"**
> Groq provides free inference for `llama-3.3-70b-versatile` with 100k tokens/day. The prompt is approximately 400 tokens and the response is capped at 100 tokens — the entire call completes in under 1 second. This meets the latency requirement for a 1Hz control loop without any API cost during development or demo.

**"What if both Spotify and Groq are down?"**
> `_fallback_query()` in `agent.py` cycles through 10 preset queries — ambient piano, nature sounds, binaural beats, Weightless by Marconi Union, etc. — skipping any already tried this session. Even with both external APIs unavailable, the system continues searching Spotify using these hardcoded queries and never hangs.

---

## Student 4 — The Loop

### Script
> "The music is playing and the Stress Gauge is beginning to drop. I'll give the system explicit feedback now. [Click 👍 Helps.]
>
> What just happened: `_handle_feedback('up', ...)` was called in `main_loop.py`. It called `pref_model.record()`, which appended this interaction — the track, artist, genre tags, EEG band scores at the time, and feedback value 1 — to `feedback_log.json` using an **atomic write**: `tempfile.NamedTemporaryFile` followed by `os.replace`. The file is never in a partially-written state. Then `partial_fit` ran immediately on just this new entry — O(1), milliseconds. The **Total Feedbacks** counter just incremented in the Personalisation panel.
>
> Clicking 👍 trains the model — but it does **not** record a Win. Wins are tracked automatically. When `stress_count` drops to 1 or below while music is playing, `main_loop.py` sets a `pending_win` flag. When the track ends or the user gives explicit feedback, `save_track_to_wins()` writes the entry to `wins_log.json`. That is how this system accumulated **190+ wins** across 500+ total intervention attempts with no human logging. The system measured its own success rate.
>
> One more dimension worth noting: this is not purely reactive. The system implements **Ultradian rhythm detection** — after 90 minutes of session time, if stress is at just 2/5, it fires a proactive intervention before the user consciously feels stressed. This is timed to the natural 90-minute human cognitive cycle identified in chronobiology research.
>
> [Toggle Focus Mode.] Same biofeedback loop, separate model instance — `FocusPreferenceModel` — different objective: attention protection rather than relaxation. Mid-tempo instrumental tracks, targeting the θ/β ratio below a threshold rather than waiting for a stress spike. Two personalised models, one architecture.
>
> On the engineering side, the dashboard updates every 500 ms by polling `state.json`. Before writing to any DOM element, it checks a `lastState` cache object — it only writes to the DOM if the value actually changed. At 500 ms that prevents over 60 redundant DOM writes per second and keeps the UI fluid even on low-powered hardware."

### What to point at
| Element | What it is |
|---|---|
| 👍 Helps button | Triggers `partial_fit` update + atomic log write |
| Total Feedbacks counter | Increments immediately on 👍 or 👎 |
| Stress Gauge dropping | Natural stress recovery while music plays |
| Focus Mode toggle | Activates separate `FocusPreferenceModel` SGD instance |
| Session History & Patterns | Visual summary of wins, fails, and tag learning |

### Technical terms
- **Atomic write** — `tempfile.NamedTemporaryFile` + `os.replace`; file is never partially written on crash
- **`partial_fit`** — O(1) SGD weight update triggered on every feedback event, reuses stored `StandardScaler`
- **`pending_win` flag** — automatic win detection: `stress_count <= 1` while `music_active` (`main_loop.py` line 651)
- **`save_track_to_wins()`** — writes to `wins_log.json` with full EEG context; no user action required
- **Ultradian rhythm detection** — proactive intervention at 90+ minutes if `stress_count == 2`
- **Dirty-checking (`lastState`)** — skips DOM writes when value unchanged; documented in `dashboard.js` line 12
- **`FocusPreferenceModel`** — separate SGD instance in `focus_model.py`; same architecture, independent weights
- **Closed-loop biofeedback** — sensor → classify → intervene → feedback → retrain → repeat

### Professor Q&A
**"Why two separate processes for the main loop and dashboard?"**
> The main loop runs a tight 1-second tick cycle for EEG classification and intervention logic. Embedding a Flask web server in the same thread would block on HTTP requests and delay EEG readings. Separating them lets the dashboard be slow (browser polling, HTTP latency) without affecting the real-time control loop. They share state through `state.json` written atomically by the main loop every tick.

**"Why state.json and not a proper database or message queue?"**
> Simplicity and zero dependencies. This is a research prototype running on a single laptop — SQLite or Redis would add setup friction with no benefit. `state.json` with atomic writes provides crash-safe IPC. The dashboard polls it every 500 ms, which is more than sufficient for a 1 Hz control loop. If the system scaled to multiple users, a message queue would be the right next step.

**"What prevents the system from triggering interventions in a loop?"**
> Several guards in `main_loop.py`: (1) `music_active` flag — once music starts, no new intervention fires until 30 seconds have passed or the music stops. (2) `interventions_tried` list — passed to the agent on every call so it never repeats a query. (3) `explicit_feedback_given` flag — prevents double-recording for the same track. (4) `_spotify_blocked_until` rate limit gate — 429 responses are caught and the gate closes for the duration of the Retry-After header, preventing rapid-fire calls.

**"What data leaves the device and where does it go?"**
> Only two external API calls: Spotify (search query string and playback commands) and Groq (the structured prompt which includes EEG band-score strings and win/fail history summaries). No raw EEG signal is transmitted. No personally identifiable information is included — EEG readings are normalised numeric scores. All logs (`feedback_log.json`, `wins_log.json`, `state.json`) are written locally.

**"What are the limitations of single-channel EEG for this use case?"**
> No spatial mapping, no Frontal Alpha Asymmetry (FAA), no ICA artifact rejection — all require multiple channels. However, a single prefrontal channel (Fp1/Fp2) is the established standard for monitoring global arousal states: Alpha/Beta balance for stress, Theta/Beta for focus. The 2024–2025 literature cited in the README confirms single prefrontal EEG is sufficient for these two specific biomarkers. The trade-off is documented explicitly under Known Limitations.

**"Why does the main loop run at 1 Hz? Why not faster?"**
> Three reasons. First, the EEG stress count uses a 5-tick rolling window — processing at 10 Hz would just re-classify the same window ten times with negligible new information. Second, each tick involves a Spotify API call (`get_now_playing`) every 5 ticks, which has network latency; running faster risks blocking the control loop on a slow response. Third, 1 Hz is sufficient for the timescale of stress interventions — stress is a physiological state that changes over seconds, not milliseconds.

**"Couldn't a user game the system by always clicking 👍 regardless of their stress?"**
> Yes — this is a real limitation. Explicit 👍 feedback calls `save_track_to_wins()` immediately with no stress check (`main_loop.py` lines 468–472). A user who clicks 👍 at stress 5/5 will train the model on that track as a win. The *automatic* win path (via `pending_win`) does require stress to drop to ≤1, so passive wins are honest. This is an acknowledged reward hacking vulnerability — the honest mitigation is that in a real deployment, the explicit feedback button would be removed and wins would only be recorded automatically by the system.

**"Could this system be improved with reinforcement learning?"**
> Yes — and the code documents the upgrade path. `preference_model.py` lines 25–35 describe an RLHF upgrade: replace the SGDClassifier with a small MLP trained via pairwise preference comparisons (Bradley-Terry model, same as InstructGPT's reward model). The `feedback_log.json` schema is already compatible — just add a `comparison_winner` field. The references are Christiano et al. (2017) and Ziegler et al. (2019).

---

## Key Numbers

| Stat | Value | Source |
|---|---|---|
| Feedback entries (pre-seeded) | 10 | `feedback_log.json` |
| Phase 2 threshold | 10 | `MIN_SAMPLES`, `preference_model.py` line 72 |
| Total intervention entries | 500+ | `wins_log.json` |
| Confirmed wins | 190+ | `wins_log.json` (status = 'win') |
| Failed interventions | 300+ | `wins_log.json` (status = 'failed') |
| Main classifier accuracy | **92.94%** | `research_and_dev/models/accuracy_log.json` |
| TinyML accuracy | **74.0%** | `research_and_dev/arduino/arduino_inference.h` |
| Ablation — EEG only | 91.94% | `research_and_dev/models/ablation_results.json` |
| Ablation — Audio only | 33.47% | `research_and_dev/models/ablation_results.json` |
| Majority-class baseline | 33.47% | Same — audio = random chance |
| Blink detector pass rate | 6/6 (100%) | `research_and_dev/models/blink_eval_results.json` |
| Stress trigger threshold | ≥ 3/5 | `main_loop.py` line 590 |
| Focus Mode θ/β trigger | > 2.5 for 3/5 ticks | `research_and_dev/models/focus_eval_results.json` |
| Poll rate | 500 ms | state.json IPC |
| Ultradian threshold | 90 minutes | `ULTRADIAN_MINS`, `main_loop.py` |

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| Music already playing when Student 3 starts | Click 👎, wait 2s, then hit 🔥 Force Stress |
| Goal Strip doesn't flip after Force Stress | Wait 1–2 seconds — state.json updates on next 500 ms tick |
| Spotify doesn't change track | Mention Last.fm fallback; if that fails, "Weightless" fires from `_FALLBACK_QUERIES` |
| Phase 2 badge shows Phase 1 | Someone ran `make clean` — pivot to explaining what Phase 2 would show |
| `make run` fails immediately | Check `.env` has `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `GROQ_API_KEY` |
| Port 5050 already in use | `make stop`, then `make run` |

---

## Commands

```bash
make run      # Start system + open browser
make stop     # Stop all processes
make logs     # Tail live logs (both processes)
make status   # Check process state + credentials
make clean    # ⚠️  Wipes ALL session history — do not run before demo
```
