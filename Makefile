SHELL   := /bin/bash
# Prefer the Python that actually has the project's packages installed.
# On Anaconda/conda systems "python" is the managed interpreter; on plain
# Linux/macOS systems "python3" is correct.
PYTHON := $(shell if python -c "import joblib" 2>/dev/null; then echo python; elif python3 -c "import joblib" 2>/dev/null; then echo python3; else echo python3; fi)

.PHONY: help setup train ablation check run stop logs status clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  EEG-Adaptive Music System"
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  First-time setup"
	@echo "    setup          Install dependencies and create .env template"
	@echo "    train          Train the EEG classifier (SGDClassifier, ~seconds)"
	@echo "    ablation       Run ablation study (EEG-only vs Audio-only vs Multimodal)"
	@echo "    check          Validate all required files are present"
	@echo ""
	@echo "  Daily use"
	@echo "    run            Start main loop + dashboard (opens browser)"
	@echo "    stop           Stop all running processes"
	@echo "    logs           Tail live logs from both processes"
	@echo "    status         Show process state, credentials, memory stats"
	@echo ""
	@echo "  Maintenance"
	@echo "    clean          Stop processes + wipe all session history (called by run)"
	@echo ""

# ── First-time setup ──────────────────────────────────────────────────────────
setup:
	@echo "→ Installing Python dependencies..."
	@$(PYTHON) -m pip install -r requirements.txt -q
	@mkdir -p data models logs
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "→ Created .env from template — open it and fill in your credentials"; \
	else \
		echo "→ .env already exists — skipping"; \
	fi
	@echo ""
	@echo "✓ Setup complete."
	@echo "  Next steps:"
	@echo "    1. Edit .env with your Spotify and Groq credentials"
	@echo "    2. Download the dataset → make train"
	@echo "    3. Start the system    → make run"

# ── Train classifier ──────────────────────────────────────────────────────────
train:
	@if [ ! -f data/eeg_mental_state.csv ]; then \
		echo ""; \
		echo "ERROR: data/eeg_mental_state.csv not found."; \
		echo ""; \
		echo "  Download the dataset from Kaggle:"; \
		echo "  https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state"; \
		echo ""; \
		echo "  Save the file as:  data/eeg_mental_state.csv"; \
		echo ""; \
		exit 1; \
	fi
	@if [ -f models/classifier.joblib ] && [ -f models/scaler.joblib ]; then \
		echo "→ Models already exist — skipping training."; \
		echo "  Delete models/*.joblib and models/class_means.json to force a retrain."; \
	else \
		echo "→ Training EEG classifier..."; \
		$(PYTHON) src/train_classifier.py; \
		echo "✓ Training complete. Models saved to models/"; \
	fi

# ── Ablation study ────────────────────────────────────────────────────────────
ablation:
	@if [ ! -f data/eeg_mental_state.csv ]; then \
		echo "ERROR: data/eeg_mental_state.csv not found — run: make train first"; \
		exit 1; \
	fi
	@echo "→ Running ablation study..."
	@$(PYTHON) src/run_ablation_study.py
	@echo "✓ Results saved to models/ablation_results.json"

# ── Validate environment ──────────────────────────────────────────────────────
check:
	@$(PYTHON) src/main_loop.py --check

# ── Run ───────────────────────────────────────────────────────────────────────
run: clean
	@if [ ! -f .env ]; then \
		echo "ERROR: .env not found — run: make setup"; \
		exit 1; \
	fi
	@if [ ! -f models/classifier.joblib ]; then \
		echo "ERROR: models/classifier.joblib missing — run: make train"; \
		exit 1; \
	fi
	@bash run.sh

# ── Stop ──────────────────────────────────────────────────────────────────────
stop:
	@bash stop.sh

# ── Logs ──────────────────────────────────────────────────────────────────────
logs:
	@mkdir -p logs
	@if [ ! -f logs/main_loop.log ] && [ ! -f logs/dashboard.log ]; then \
		echo "No log files yet — run: make run"; \
	else \
		tail -f logs/main_loop.log logs/dashboard.log; \
	fi

# ── Status ────────────────────────────────────────────────────────────────────
status:
	@echo ""
	@echo "=== Processes ==="
	@if [ -f .pids ]; then \
		while IFS='=' read -r name pid; do \
			[ -z "$$pid" ] && continue; \
			if kill -0 "$$pid" 2>/dev/null; then \
				echo "  $$name (PID $$pid): RUNNING"; \
			else \
				echo "  $$name (PID $$pid): STOPPED (stale .pids entry)"; \
			fi; \
		done < .pids; \
	else \
		echo "  Not running"; \
	fi
	@echo ""
	@echo "=== Credentials (.env) ==="
	@if [ ! -f .env ]; then \
		echo "  .env: NOT FOUND — run: make setup"; \
	else \
		for var in SPOTIFY_CLIENT_ID SPOTIFY_CLIENT_SECRET GROQ_API_KEY; do \
			val=$$(grep "^$$var=" .env | cut -d= -f2- | tr -d '"'"'"' '); \
			if [ -z "$$val" ]; then \
				echo "  $$var: NOT SET"; \
			else \
				echo "  $$var: set"; \
			fi; \
		done; \
	fi
	@echo ""
	@echo "=== Models ==="
	@for f in models/classifier.joblib models/scaler.joblib models/class_means.json; do \
		if [ -f "$$f" ]; then \
			echo "  $$f: present"; \
		else \
			echo "  $$f: MISSING — run: make train"; \
		fi; \
	done
	@echo ""
	@echo "=== Memory ==="
	@$(PYTHON) -c "\
import json, os; \
path = 'wins_log.json'; \
log = json.load(open(path)) if os.path.exists(path) else []; \
wins = sum(1 for e in log if e.get('status') == 'win'); \
fails = sum(1 for e in log if e.get('status') == 'failed'); \
print(f'  Entries: {len(log)}  |  Wins: {wins}  |  Failed: {fails}') \
" 2>/dev/null || echo "  wins_log.json: not found"
	@echo ""

# ── Clean — stops processes and wipes all session history ─────────────────────
clean:
	@bash stop.sh 2>/dev/null; true
	@echo "[]" > wins_log.json
	@echo "[]" > feedback_log.json
	@rm -f state.json feedback_signal.json
	@echo "✓ Session history cleared"
