# EEG-Adaptive Music System

**MSc Foundations of AI | National College of Ireland | H9FAI | March 2026**

A real-time system that replays EEG brainwave data, classifies mental state with a trained SVM, and — when sustained stress is detected — uses an AI agent to search Spotify for calming music and play it automatically. Every outcome (win or failure) is logged, giving the agent an increasingly personalised model of what music works for each user.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        main_loop.py                           │
│  EEG CSV ──► ML Classifier ──► Stress Score ──► Logic        │
└──────┬────────────────────────────────────┬──────────────────┘
       │ stress ≥ 3/5                       │ write every 2 s
       ▼                                    ▼
┌─────────────┐   search query    ┌──────────────────┐
│  agent.py   │ ────────────────► │spotify_controller│
│  Groq LLM   │                   │  search + play   │
└─────────────┘                   └──────────────────┘
                                            │ win/fail
                                            ▼
                                   ┌──────────────┐       ┌──────────────┐
                                   │  state.json  │ ────► │ dashboard.py │
                                   │wins_log.json │       │  Flask :5050 │
                                   └──────────────┘       └──────────────┘
```

| Component | File | Role |
|-----------|------|------|
| ML Classifier | `src/train_classifier.py` | Trains SVM on EEG data, saves `models/` |
| Spotify Controller | `src/spotify_controller.py` | Auth, device, search, playback, memory log |
| AI Agent | `src/agent.py` | Groq LLM generates search query from stress history + memory |
| Main Loop | `src/main_loop.py` | Replays EEG row by row, runs intervention logic |
| Dashboard | `src/dashboard.py` + `templates/dashboard.html` | Live browser UI on port 5050 |

---

## Prerequisites

- **Python 3.10+**
- **Spotify Premium** account with the Spotify desktop/mobile app open on at least one device
- **Groq API key** — free at [console.groq.com](https://console.groq.com) (100 000 tokens/day)
- **EEG dataset** — see Step 2 below
- **make** — pre-installed on macOS and Linux; Windows users: no make needed, use the `.bat` scripts below

---

## Quick Start

> **Windows user?** Skip to the [Windows Quick Start](#windows-quick-start) section below.

### Step 1 — Clone and install

```bash
git clone <repo-url>
cd eeg-music-system
make setup
```

`make setup` installs all Python dependencies and creates a `.env` file from the template.

---

### Step 2 — Configure credentials

Open `.env` and fill in every field:

| Variable | Where to get it |
|----------|----------------|
| `SPOTIFY_CLIENT_ID` | [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) → your app → Settings |
| `SPOTIFY_CLIENT_SECRET` | Same place |
| `SPOTIFY_REDIRECT_URI` | Set to exactly `http://127.0.0.1:8888/callback` in your Spotify app settings **and** in `.env` |
| `SPOTIFY_WINS_PLAYLIST_ID` | Open any playlist in Spotify → share → copy link → the ID is the string after `/playlist/` |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys → Create |

**Spotify app setup (one time):**

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Click **Create app**
3. Fill in any name/description
4. Set **Redirect URI** to: `http://127.0.0.1:8888/callback`
   - This exact URI is required — `localhost` has been banned since April 2025
5. Save. Copy the **Client ID** and **Client Secret** into `.env`.

---

### Step 3 — Train the classifier

```bash
make train
```

This runs the SGDClassifier training pipeline on the EEG dataset (completes in seconds). Expect output like:

```
Training SGDClassifier (log_loss)...
Accuracy: 91.94%
Saved: models/classifier.joblib
Saved: models/scaler.joblib
Saved: models/class_means.json
```

Training only needs to run once. The next `make train` will skip if models already exist.

---

### Step 4 — Spotify OAuth (first run only)

On the very first `make run`, spotipy needs to authenticate with Spotify:

1. A browser window opens to the Spotify login page
2. Log in and click **Agree**
3. The browser redirects to `http://127.0.0.1:8888/callback?code=...` — this page will show an error (that's expected)
4. **Copy the full URL** from the browser address bar
5. **Paste it** into the terminal and press Enter

spotipy saves a `.cache` token file and handles silent token refresh from then on. You will not be asked again unless `.cache` is deleted.

---

### Step 5 — Run

```bash
make run
```

This starts both processes, opens `http://127.0.0.1:5050` in your browser automatically, and tails the log output. Press **Ctrl+C** to stop both cleanly.

---

---

## Windows Quick Start

No WSL, no Make, no terminal expertise required. Everything runs from double-clickable `.bat` files.

### What you need before starting

- **Python 3.10 or later** — download from [python.org/downloads](https://www.python.org/downloads/)
  - During installation, check **"Add Python to PATH"** — this is required
- **Spotify Premium** account with the Spotify desktop app open and playing on your laptop
- **Groq API key** — free at [console.groq.com](https://console.groq.com) (100,000 tokens/day)

---

### Step 1 — Download or clone the project

**Option A — Download ZIP (no Git needed):**
1. Click the green **Code** button on the GitHub page
2. Click **Download ZIP**
3. Right-click the ZIP → **Extract All** → choose a folder (e.g. `C:\Users\YourName\eeg-music-system`)

**Option B — Git clone:**
```
git clone <repo-url>
cd eeg-music-system
```

---

### Step 2 — First-time setup

Double-click **`setup.bat`** inside the project folder.

It will:
- Install all Python dependencies automatically
- Create a `.env` file from the template for you to fill in

If Windows asks _"Do you want to allow this app to make changes?"_, click **Yes**.

---

### Step 3 — Configure your API keys

Open the `.env` file in Notepad (or any text editor) and fill in every field:

| Variable | Where to get it |
|----------|----------------|
| `SPOTIFY_CLIENT_ID` | [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) → your app → Settings |
| `SPOTIFY_CLIENT_SECRET` | Same page as above |
| `SPOTIFY_REDIRECT_URI` | Set to exactly `http://127.0.0.1:8888/callback` |
| `SPOTIFY_WINS_PLAYLIST_ID` | Open any playlist in Spotify → Share → Copy link → the ID is the string after `/playlist/` |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys → Create |

**Spotify app setup (one time):**

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Click **Create app**
3. Fill in any name/description
4. Under **Redirect URIs**, add exactly: `http://127.0.0.1:8888/callback`
   - Use `127.0.0.1`, not `localhost` — Spotify banned `localhost` in April 2025
5. Save. Copy the **Client ID** and **Client Secret** into `.env`.

---

### Step 4 — Train the classifier

Double-click **`train.bat`** inside the project folder.

Expected output:
```
Training SGDClassifier (log_loss)...
Accuracy: 91.94%
Saved: models/classifier.joblib
```

This only needs to run once. The models are saved and reused on every subsequent launch.

---

### Step 5 — Spotify login (first run only)

On the very first launch, Spotify needs to verify your identity:

1. A browser tab opens to the Spotify login page — log in and click **Agree**
2. The browser redirects to a page that shows an error or blank screen — that is expected
3. **Copy the full URL** from the browser address bar (it starts with `http://127.0.0.1:8888/callback?code=...`)
4. Switch back to the Command Prompt window and **paste the URL**, then press **Enter**

Spotify saves your login token to a `.cache` file. You will not be asked again unless you delete `.cache`.

---

### Step 6 — Start the system

Double-click **`run.bat`**.

The system will:
1. Check your `.env` credentials
2. Start the dashboard server and the main EEG loop
3. Open your browser to `http://127.0.0.1:5050` automatically

**To stop:** press **Ctrl+C** in the window that opened, or double-click **`stop.bat`**.

---

### Windows Troubleshooting

**"Python is not recognized as an internal or external command"**
Python is not on your PATH. Re-run the Python installer, click **Modify**, and check **"Add Python to environment variables"**. Then close and reopen Command Prompt.

**"No module named joblib" or similar import error**
Run `setup.bat` again, or open Command Prompt and run:
```
pip install -r requirements.txt
```

**The browser opens but the dashboard is blank or spinning forever**
The main loop hasn't started yet. Wait 5–10 seconds and refresh. If it stays blank, check `logs\dashboard.log` and `logs\main_loop.log` inside the project folder for error details.

**"No active device" / Spotify won't play**
Open the Spotify desktop app on your laptop and play any track manually first. The Spotify API requires at least one active device. Then relaunch with `run.bat`.

**Spotify OAuth page — "INVALID_CLIENT: Redirect URI mismatch"**
The Redirect URI in your Spotify Developer app settings does not match `.env`. Make sure both are set to exactly `http://127.0.0.1:8888/callback` (no trailing slash, no `localhost`).

**Windows Defender / antivirus blocks run.bat**
`.bat` files can trigger Windows Defender Smart Screen. Click **More info** → **Run anyway**. The scripts only start Python processes — no system changes are made.

**"Port 5050 already in use"**
A previous session is still running. Double-click `stop.bat`, wait a few seconds, then try `run.bat` again.

---

## Dashboard

The live dashboard at `http://127.0.0.1:5050` updates every 1 second.

```
┌────────────────────────────────────────────────────────────────┐
│  Brain State   │  Stress Timeline                              │
│  δ Delta  ████ │  5 ┤                  ╭──╮                   │
│  θ Theta  ██   │  3 ┤     ╭────────╮  │  │                   │
│  α Alpha  █    │  0 ┤─────╯        ╰──╯  ╰─────────────────  │
│  β Beta   ████ │                                               │
│  γ Gamma  ████ │  — Calm/Relaxed    — Stressed (≥ 3/5)       │
├──────────────┬─┴──────────────────┬───────────────────────────┤
│  AI Agent    │  Now Playing        │  Learning Progress        │
│  Reasoning   │  Track / Artist     │  Session wins            │
│  ...         │  Calm Audio Profile │  Memory log              │
├──────────────┴────────────────────┴───────────────────────────┤
│  Memory Log — last 20 wins and failures                       │
└────────────────────────────────────────────────────────────────┘
```

**If the dashboard does not update:** hard-refresh with **Cmd+Shift+R** (macOS) or **Ctrl+Shift+R** (Linux/Windows).

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies, create `.env` from template |
| `make train` | Train SGDClassifier (skip if models already exist) |
| `make ablation` | Run ablation study — saves results to `models/ablation_results.json` |
| `make check` | Validate all required files are present |
| `make run` | Start main loop + dashboard, open browser |
| `make stop` | Stop all running processes |
| `make logs` | Tail live logs from both processes |
| `make status` | Show process state, credentials check, memory stats |
| `make clean` | Remove caches, `state.json`, log files |
| `make reset-memory` | Wipe `wins_log.json` (requires typing YES) |

---

## Troubleshooting

**"No active device" / playback fails**
Open the Spotify app on your laptop or phone and play any track. The API requires at least one active device. Then run again.

**"Groq error 429 — rate limit"**
The free Groq tier allows 100 000 tokens per day, resetting at midnight UTC. The system automatically falls back to a rotating list of proven stress-relief queries — the agent is non-functional but the rest of the system keeps running.

**"classifier.joblib missing"**
Run `make train`. If the dataset is also missing, download it first (Step 2).

**"Port 5050 already in use"**
Run `make stop` to clear any stale processes, then `make run`.

**Dashboard not reflecting code changes**
Flask serves templates with caching disabled, but your browser may cache the page. Hard-refresh: **Cmd+Shift+R** (macOS) or **Ctrl+Shift+R** (Linux/Windows).

**Spotify OAuth token expired**
Delete `.cache` in the project root, then run `make run` to go through the OAuth flow again.

**"Warning: nothing playing — cannot save win"**
The Spotify session timed out or the track ended before the win was logged. Open Spotify, play something, and the next win will log correctly.

---

## Project Structure

```
eeg-music-system/
├── Makefile                  ← build / run / stop targets (macOS/Linux)
├── run.sh                    ← starts both processes, traps Ctrl+C (macOS/Linux)
├── stop.sh                   ← kills processes from .pids (macOS/Linux)
├── launch.py                 ← cross-platform launcher (used by run.bat)
├── stop.py                   ← cross-platform stopper (used by stop.bat)
├── run.bat                   ← Windows: double-click to start
├── stop.bat                  ← Windows: double-click to stop
├── setup.bat                 ← Windows: first-time setup (pip install + .env)
├── train.bat                 ← Windows: train the classifier
├── requirements.txt          ← Python dependencies
├── .env.example              ← credential template (copy → .env)
├── .env                      ← your credentials (gitignored)
├── .cache                    ← spotipy OAuth token (gitignored)
├── state.json                ← live state written by main loop
├── wins_log.json             ← persistent memory log
├── data/
│   └── eeg_mental_state.csv  ← EEG dataset (download separately)
├── models/
│   ├── classifier.joblib     ← trained SGDClassifier (created by make train)
│   ├── scaler.joblib         ← feature scaler
│   ├── class_means.json      ← per-class feature means for band scoring
│   └── accuracy_log.json     ← persisted test accuracy + run metadata
├── src/
│   ├── train_classifier.py
│   ├── run_ablation_study.py ← ablation: EEG-only vs Audio-only vs Multimodal
│   ├── spotify_controller.py
│   ├── agent.py
│   ├── main_loop.py
│   └── dashboard.py
└── templates/
    └── dashboard.html
```

---

## Notes on Spotify API Restrictions

Spotify's **Extended Access** policy (November 2024) restricts Development Mode apps from:

- Modifying playlists (`playlist-modify-*`)
- Modifying the library (`user-library-modify`)
- Reading audio features (`/audio-features`)

As a result, winning tracks are logged to `wins_log.json` only (not added to the Spotify playlist), and audio feature learning is silently skipped. Both the agent and dashboard read from `wins_log.json` as the sole source of truth. All core functionality — EEG classification, stress detection, music playback, and memory — works without Extended Access.

---

## Dataset

Veeramalla, A. (2019). *EEG Brainwave Dataset: Mental State* [Dataset]. Kaggle.
https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state

---

## Scientific Methodology

MindTune-OS uses two distinct neurological signatures to trigger interventions. These thresholds are derived from clinical BCI literature (e.g., Monastra et al., 2005) and calibrated for the 256Hz single-channel prefrontal signal.

### 1. Calm Mode: Stressed ➔ Calm
*   **Biomarker:** ML-Classified Mental State (988 features).
*   **The Trigger:** A rolling window of 5 ticks (10 seconds). Intervention fires if **≥ 3/5** readings are labeled `stressed`.
*   **The Logic:** Stress is identified by a specific signature of suppressed Alpha (8–12 Hz) and elevated Beta (13–30 Hz).
*   **Efficacy:** The transition is considered successful when the ML prediction returns to `calm` or `relaxed`.

### 2. Focus Mode: Unfocused ➔ Focused
*   **Biomarker:** $\theta / \beta$ (Theta/Beta) Ratio.
*   **The Trigger:** Intervention fires if the ratio exceeds **2.5** for 3 out of 5 ticks.
*   **The Logic:** High $\theta$ (4–8 Hz) relative to $\beta$ (13–30 Hz) is a primary marker for mind-wandering and daydreaming. 
*   **Efficacy:** The transition is considered successful when the $\theta / \beta$ ratio drops below **1.0** (Beta dominance).

### 3. Hardware Interfacing (BioAmp EXG Pill)
The system extracts 5 primary frequency bands from the raw ADC stream using a 128-point FFT. These bands are converted into **Deviation Scores [0, 1]** by comparing the current signal to the mean values of the `calm` and `stressed` classes in the training dataset.

---

## Literature Review & Research Basis

MindTune-OS is built upon current neuroscientific research (2024–2025) regarding EEG biomarkers and the regulatory role of music in arousal management.

### 1. Neurological Biomarkers
*   **Calm State (Alpha/Beta Balance):** Alpha (8–12 Hz) represents "relaxed alertness." Research confirms that a sudden drop in the Alpha/Beta ratio is a reliable precursor to physiological stress (Shan & Qi, 2025). Targeted **High-Alpha (11–13 Hz)** is increasingly recognized as the primary sub-band for emotional recovery.
*   **Focus State (Theta/Beta Ratio):** Frontal Theta (4–8 Hz) power is a robust marker for attentional selection. A high **Theta/Beta Ratio (TBR)** is a clinical indicator of mind-wandering or under-arousal (ADHD biomarker). 2024 research identifies **SMR/Low-Beta (12–15 Hz)** as the "sweet spot" for calm, sustainable concentration (Monastra et al., 2005).

### 2. Music as a Neurological Regulator
Music is utilized not just for subjective relief, but as a "precision tuning tool" for brain arousal (Yerkes-Dodson Law, 2025 Refinement).
*   **For Stress Relief:** 
    *   **The 24-Minute Rule:** A daily "dose" of 24 minutes of slow, predictable music is required to measurably lower cortisol and shift the body into a parasympathetic state (Song et al., 2024).
    *   **Entrainment:** Slow tempos (60–80 BPM) matching a resting heart rate facilitate "physiological entrainment," slowing respiration and heart rate.
*   **For Cognitive Focus:**
    *   **Dopaminergic Modulation:** Music triggers dopamine release in the prefrontal cortex, aiding "stay-on-task" endurance.
    *   **Lyrics & Working Memory:** 2024 studies confirm that lyrics compete for working memory resources. MindTune-OS explicitly targets **instrumental tracks** for Focus Mode to prevent this cognitive interference.
    *   **Arousal Scaffolding:** Mid-to-high energy beats (120–140 BPM) provide a rhythmic scaffold that prevents "Theta-driven" daydreaming without inducing anxiety.

---

## Known Limitations

**1. Kaggle proxy training — not raw EEG**
The classifier is trained on pre-processed spectral features from the Kaggle CSV, not on raw time-domain EEG signals. The Kaggle columns (`freq_XXX_C`) are already-computed frequency-domain measurements from a different recording setup. The TinyML model (Arduino) computes fresh FFT magnitudes from a physical BioAmp EXG Pill, and a StandardScaler bridges the two distributions. This is an acknowledged approximation documented as "proxy-trained, scaler-aligned" — it works in practice but means the training and live-inference feature spaces are not identical.

**2. Audio features trained on neutral placeholders**
The Spotify audio scalars (tempo, energy, valence, acousticness, instrumentalness) are all set to `0.5` during batch training because the Kaggle EEG dataset contains no Spotify metadata. At runtime the classifier receives real Spotify values, but it was never trained on the actual distribution of those features. The audio scalars therefore contribute minimal discriminative signal in the initial model; they become more meaningful only after enough `partial_fit` updates with real feedback.

**3. Single-channel EEG constraint**
The BioAmp EXG Pill provides one differential channel. Multi-channel techniques (ICA, spatial filtering, inter-channel coherence) are not possible with this hardware.
*   **The Limitation:** No Frontal Alpha Asymmetry (FAA) or spatial mapping.
*   **The Scientific Win:** 2024-2025 research confirms that a single prefrontal channel (Fp1/Fp2) is the "gold standard" for monitoring **global arousal states** (Stress via Alpha/Beta) and **executive function** (Focus via Theta/Beta). Furthermore, the proximity to the eyes makes a single frontal channel superior for **EOG-based intentional control** (Double-blinks), as the signal-to-noise ratio for eye artifacts is maximized at this location.

**4. No sub-millisecond event locking**
USB serial latency and Python `time.sleep` scheduling jitter prevent reliable event-locked ERP analysis. ErrP detection and SSVEP paradigms require sub-millisecond synchronisation and are not feasible with this stack.

**5. Accuracy reproducibility requires the dataset**
Test accuracy is persisted to `models/accuracy_log.json` after each training run. The file is gitignored (it is generated, not source). To reproduce the reported figure, run `make train` (the dataset is included in the repository).

---

## Acknowledgements

- EEG dataset by [@birdy654](https://www.kaggle.com/birdy654) on Kaggle
- Groq for free LLM inference ([console.groq.com](https://console.groq.com))
- spotipy for the Spotify Web API Python wrapper

