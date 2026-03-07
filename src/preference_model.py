"""Personalized music preference model.

In plain English: this module remembers which songs helped reduce your stress.
After enough sessions it learns to predict — from your current brainwave state —
what music is likely to calm YOU specifically, not just what works on average.
It starts with simple win-rate counting (Phase 1) and upgrades automatically
to a logistic regression model once 10 feedback entries have been collected (Phase 2).

Learns which music features work best for THIS user's EEG state
from explicit ↑ / ↓ keyboard feedback collected during sessions.

━━━ Learning phases ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Phase 1 (0 – MIN_SAMPLES-1 feedback entries)
    Frequency-based scoring with Laplace smoothing.
    Queries and Last.fm tags that received ↑ are preferred; ↓ are down-weighted.
    No ML training — runs immediately from the first session.

  Phase 2 (MIN_SAMPLES+ feedback entries)
    Logistic regression on a joint feature vector:
      [delta, theta, alpha, beta, gamma,          ← EEG band scores (0-1)
       ambient, piano, instrumental, ...]          ← Last.fm tag presence (0/1)
    Predicts P(user gives ↑ | current EEG state + candidate music tags).
    Model is retrained automatically after each new feedback entry.

━━━ RLHF upgrade path (future) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Replace Phase 2 logistic regression with a neural reward model trained via
  pairwise preference comparisons (Bradley-Terry / Elo style):
    - Present two candidate tracks simultaneously to the user.
    - User picks the one that feels more calming for their current state.
    - Train a reward model R(EEG, music_features) on these binary choices.
    - Use R to rank candidates before playing — same interface as Phase 2.
  Reference: Christiano et al. (2017) "Deep RL from Human Preferences"
             Ziegler et al. (2019) "Fine-Tuning Language Models from Human Feedback"
  The feedback_log.json schema is already compatible with this upgrade:
  just add a 'comparison_winner' field alongside 'feedback'.

━━━ Feature schema (feedback_log.json) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    "timestamp":      "2026-02-28T14:22:11",
    "track":          "Weightless",
    "artist":         "Marconi Union",
    "query":          "artist:\"Marconi Union\"",
    "tags":           ["ambient", "relaxing", "sleep", "instrumental"],
    "eeg_band_scores": {"Delta": 0.2, "Theta": 0.4, "Alpha": 0.1,
                        "Beta": 0.85, "Gamma": 0.72},
    "feedback":       1          // 1 = ↑ helped,  0 = ↓ didn't help
  }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
import datetime
import tempfile
from collections import defaultdict

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_LOG_PATH = os.path.join(BASE, '..', 'feedback_log.json')

EEG_BANDS         = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
# Ratio features validated as strongest stress biomarkers (PMC9749579):
#   α/β decreases under stress, θ/β increases under stress.
# Appended after the 5 band scores in the feature vector.
EEG_RATIO_FEATURES = ['α/β ratio', 'θ/β ratio']
EEG_FEATURE_LABELS = EEG_BANDS + EEG_RATIO_FEATURES  # used in get_insights()
N_EEG_FEATURES     = len(EEG_FEATURE_LABELS)          # = 7

MIN_SAMPLES = 10   # entries needed before Phase 2 (ML) activates


class PreferenceModel:
    """Personalized EEG-state × music-tag preference model.

    Instantiate once at startup; call record() after each ↑/↓ press.
    Use score_candidate() to rank music choices before playing them.
    """

    def __init__(self, log_path=FEEDBACK_LOG_PATH):
        self.log_path         = log_path
        self.log              = self._load_log()
        self._ml_model        = None
        self._feature_scaler  = None    # StandardScaler — fit during full fit, reused in partial_fit
        self._sgd_initialized = False   # True after first full SGD fit
        self._all_tags        = []
        self._query_counts  = defaultdict(lambda: {'pos': 0, 'neg': 0})
        self._tag_counts    = defaultdict(lambda: {'pos': 0, 'neg': 0})
        self._recompute_frequencies()
        if len(self.log) >= MIN_SAMPLES:
            self._train_ml_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, track, artist, query, tags, eeg_band_scores, feedback):
        """Save a feedback entry and retrain if enough data exists.

        Args:
            track:           Track name (str)
            artist:          Artist name (str)
            query:           Spotify search query used (str)
            tags:            Last.fm tag list for this track (list[str])
            eeg_band_scores: Dict of band→float from eeg_source (dict)
            feedback:        1 = ↑ (helped reduce stress), 0 = ↓ (didn't help)
        """
        entry = {
            'timestamp':       datetime.datetime.now().isoformat(),
            'track':           track,
            'artist':          artist,
            'query':           query,
            'tags':            tags or [],
            'eeg_band_scores': eeg_band_scores or {},
            'feedback':        int(feedback),
        }
        self.log.append(entry)
        self._save_log()
        self._recompute_frequencies()
        if len(self.log) >= MIN_SAMPLES:
            self._train_ml_model(new_entry=entry)
        label = '↑ POSITIVE' if feedback else '↓ NEGATIVE'
        print(f"Preference recorded [{label}]: '{track}' (total: {len(self.log)})")

    def score_candidate(self, query, tags, eeg_band_scores=None):
        """Return a 0-1 preference score for a candidate track.

        Uses the ML model when trained (Phase 2), otherwise falls back
        to frequency-based scoring (Phase 1).

        Args:
            query:           Spotify query string for this candidate
            tags:            Last.fm tags for this candidate (list[str])
            eeg_band_scores: Current EEG state (dict) — used in Phase 2 only

        Returns:
            float in [0, 1] — higher means more likely to be preferred
        """
        if self._ml_model is not None and eeg_band_scores:
            score = self._predict_ml(eeg_band_scores, tags)
            if score is not None:
                return score
        # Phase 1 fallback: average of query score + tag score
        return 0.5 * self._score_query(query) + 0.5 * self._score_tags(tags)

    def get_insights(self):
        """Return dict of human-readable personalization insights for the dashboard."""
        n_pos = sum(1 for e in self.log if e.get('feedback') == 1)
        n_neg = sum(1 for e in self.log if e.get('feedback') == 0)

        # Tag scores with minimum support of 2 entries (Phase 1 frequency-based)
        tag_scores = []
        for tag, counts in self._tag_counts.items():
            support = counts['pos'] + counts['neg']
            if support >= 2:
                score = (counts['pos'] + 1) / (support + 2)
                tag_scores.append({'tag': tag, 'score': round(score, 2),
                                   'support': support})
        tag_scores.sort(key=lambda x: x['score'], reverse=True)

        # Phase 2: extract what the ML model actually learned from coef_
        # coef_ shape: (1, n_features) where features = [Delta..Gamma, tag0, tag1, ...]
        # Positive coef → this feature predicts music will help.
        # Negative coef → this feature predicts music won't help.
        eeg_weights  = []
        ml_top_tags  = []
        ml_worst_tags = []
        if self._ml_model is not None:
            try:
                coefs = self._ml_model.coef_[0]
                # ── EEG band + ratio weights ──────────────────────────────────
                # Feature vector has N_EEG_FEATURES = 7:
                #   [Delta, Theta, Alpha, Beta, Gamma, α/β ratio, θ/β ratio]
                eeg_coefs = coefs[:N_EEG_FEATURES]
                max_abs   = max(abs(c) for c in eeg_coefs) or 1.0
                eeg_weights = sorted([
                    {
                        'band':     EEG_FEATURE_LABELS[i],
                        'weight':   round(float(eeg_coefs[i]), 3),
                        # strength: 0–1 normalized absolute importance
                        'strength': round(abs(float(eeg_coefs[i])) / max_abs, 2),
                    }
                    for i in range(N_EEG_FEATURES)
                ], key=lambda x: x['weight'], reverse=True)

                # ── Tag weights from ML (more precise than win-rate) ──────────
                if self._all_tags:
                    tag_coefs    = coefs[N_EEG_FEATURES:]
                    max_abs_t    = max(abs(c) for c in tag_coefs) or 1.0
                    tag_w_list   = sorted([
                        {
                            'tag':      self._all_tags[i],
                            'weight':   round(float(tag_coefs[i]), 3),
                            'strength': round(abs(float(tag_coefs[i])) / max_abs_t, 2),
                        }
                        for i in range(len(tag_coefs))
                    ], key=lambda x: x['weight'], reverse=True)
                    ml_top_tags   = tag_w_list[:5]
                    ml_worst_tags = [t for t in reversed(tag_w_list) if t['weight'] < 0][:5]
            except Exception:
                pass  # coef_ not available yet — stay with empty lists

        return {
            'total':        len(self.log),
            'positive':     n_pos,
            'negative':     n_neg,
            'phase':        2 if self._ml_model is not None else 1,
            'top_tags':     tag_scores[:5],
            'worst_tags':   tag_scores[-5:][::-1] if len(tag_scores) >= 5 else [],
            # Phase 2 extras — empty lists when model not yet trained
            'eeg_weights':  eeg_weights,   # [{band, weight, strength}, ...] sorted by weight
            'ml_top_tags':  ml_top_tags,   # tags with highest positive coef (helps most)
            'ml_worst_tags': ml_worst_tags, # tags with most negative coef (least helpful)
        }

    @property
    def sample_count(self):
        return len(self.log)

    @property
    def ml_active(self):
        return self._ml_model is not None

    # ── Internal: frequency scoring (Phase 1) ─────────────────────────────────

    def _recompute_frequencies(self):
        self._query_counts = defaultdict(lambda: {'pos': 0, 'neg': 0})
        self._tag_counts   = defaultdict(lambda: {'pos': 0, 'neg': 0})
        for entry in self.log:
            feedback_value = entry.get('feedback')
            key = 'pos' if feedback_value == 1 else 'neg' if feedback_value == 0 else None
            if key is None:
                continue
            q = entry.get('query', '')
            if q:
                self._query_counts[q][key] += 1
            for tag in entry.get('tags', []):
                self._tag_counts[tag][key] += 1

    def _score_query(self, query):
        """Laplace-smoothed win rate for this exact query string."""
        c = self._query_counts.get(query, {'pos': 0, 'neg': 0})
        return (c['pos'] + 1) / (c['pos'] + c['neg'] + 2)

    def _score_tags(self, tags):
        """Average Laplace-smoothed win rate across all provided tags."""
        if not tags:
            return 0.5
        scores = []
        for tag in tags:
            c = self._tag_counts.get(tag, {'pos': 0, 'neg': 0})
            scores.append((c['pos'] + 1) / (c['pos'] + c['neg'] + 2))
        return sum(scores) / len(scores)

    # ── Internal: ML model (Phase 2) ──────────────────────────────────────────

    def _build_feature_vector(self, eeg_band_scores, tags):
        eeg_vec = [eeg_band_scores.get(b, 0.5) for b in EEG_BANDS]
        # Ratio features: validated stress biomarkers (PMC9749579)
        alpha = eeg_band_scores.get('Alpha', 0.5)
        beta  = eeg_band_scores.get('Beta',  0.5)
        theta = eeg_band_scores.get('Theta', 0.5)
        # Clamp to [0, 100] to prevent extreme values when Beta ≈ 0.
        # Without clamping, beta=0 → ratio ≈ 1e6, which breaks the scaler.
        # Upper bound of 100 is orders of magnitude above any real EEG ratio.
        eeg_vec.append(min(100.0, alpha / (beta + 1e-6)))   # α/β: low under stress
        eeg_vec.append(min(100.0, theta / (beta + 1e-6)))   # θ/β: high under stress
        tag_vec = [1 if t in tags else 0 for t in self._all_tags]
        return eeg_vec + tag_vec

    def _train_ml_model(self, new_entry=None):
        """Train/update the SGD preference model incrementally.

        Uses SGDClassifier(loss='log_loss') which is a probabilistic linear
        classifier equivalent to logistic regression but supports partial_fit()
        for fast incremental updates without full retraining (IJRITCC 2024).

        Strategy:
          - First activation or tag vocabulary change → full fit on all data.
          - Subsequent entries (stable vocabulary) → partial_fit on new entry only.
            This is O(1) per feedback event instead of O(n).

        ── RLHF upgrade note ────────────────────────────────────────────────
        To upgrade to a neural reward model:
          1. Replace SGDClassifier with a small MLP (torch.nn.Sequential):
             layers: input_dim → 64 → 32 → 1 (sigmoid output)
          2. Convert binary feedback to pairwise comparisons:
             for each pair (entry_A, entry_B), label = 1 if A.feedback > B.feedback
          3. Train with Binary Cross Entropy on pairwise logits
             (Bradley-Terry model — same as InstructGPT reward model)
          4. Replace self._predict_ml() with the MLP's forward pass
        The feature vector schema and record() interface stay the same.
        ─────────────────────────────────────────────────────────────────────
        """
        try:
            # Rebuild tag vocabulary — needed to detect changes
            new_all_tags = sorted({
                tag for entry in self.log for tag in entry.get('tags', [])
            })
            vocab_changed = new_all_tags != self._all_tags
            self._all_tags = new_all_tags

            # Can we do a cheap incremental update?
            can_partial = (
                self._sgd_initialized and
                not vocab_changed and
                new_entry is not None and
                new_entry.get('feedback') in (0, 1)
            )

            if can_partial:
                # Fast path: update on just the one new entry.
                # Re-use the existing scaler fitted during the last full fit.
                # SGDClassifier is very sensitive to feature scale, so scaling
                # is mandatory — partial_fit uses the stored scaler without
                # refitting it (refitting on new data would break partial_fit).
                eeg  = new_entry.get('eeg_band_scores', {})
                tags = new_entry.get('tags', [])
                X_new = np.array([self._build_feature_vector(eeg, tags)])
                X_new_scaled = self._feature_scaler.transform(X_new)
                y_new = [int(new_entry['feedback'])]
                self._ml_model.partial_fit(X_new_scaled, y_new, classes=[0, 1])
                print(f"Preference model: incremental update "
                      f"({len(self.log)} samples total)")
            else:
                # Full fit: first activation, or tag vocab grew.
                X, y = [], []
                for entry in self.log:
                    fb = entry.get('feedback')
                    if fb not in (0, 1):
                        continue
                    eeg  = entry.get('eeg_band_scores', {})
                    tags = entry.get('tags', [])
                    X.append(self._build_feature_vector(eeg, tags))
                    y.append(fb)

                if len(set(y)) < 2:
                    # Need both positive and negative examples to train
                    return

                X_arr = np.array(X)
                # StandardScaler normalises each feature to zero mean and unit
                # variance.  This is CRITICAL for SGDClassifier — without it,
                # the ratio features (α/β, θ/β) can reach values of 100+ when
                # Beta≈0, swamping the model with coefficients of ±600 while
                # the meaningful band-score features are ignored.
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_arr)

                model = SGDClassifier(loss='log_loss', random_state=42,
                                      max_iter=1000, tol=1e-3)
                model.fit(X_scaled, y)
                self._ml_model        = model
                self._feature_scaler  = scaler
                self._sgd_initialized = True
                reason = 'vocab changed' if vocab_changed else 'initial fit'
                print(f"Preference model (Phase 2) trained [{reason}]: "
                      f"{len(X)} samples, "
                      f"{sum(y)} positive / {len(y)-sum(y)} negative")

        except Exception as ex:
            print(f"Preference model training skipped: {ex}")

    def _predict_ml(self, eeg_band_scores, tags):
        """Return P(positive feedback) from ML model, or None on error."""
        try:
            vec    = self._build_feature_vector(eeg_band_scores, tags)
            scaled = self._feature_scaler.transform(np.array([vec]))
            prob   = self._ml_model.predict_proba(scaled)[0][1]
            return float(prob)
        except Exception:
            return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_log(self):
        # M-3: validate that the loaded value is a list — a corrupt or hand-edited
        # file may contain a JSON object {}, causing AttributeError on .append().
        try:
            with open(self.log_path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_log(self):
        # C-3: atomic write via tempfile + os.replace — prevents corrupt JSON
        # if the process is killed (SIGKILL, OOM, power loss) mid-write.
        tmp_name = None
        try:
            dir_path = os.path.dirname(os.path.abspath(self.log_path))
            with tempfile.NamedTemporaryFile('w', dir=dir_path,
                                             delete=False, suffix='.tmp') as tmp:
                tmp_name = tmp.name
                json.dump(self.log, tmp, indent=2)
            os.replace(tmp_name, self.log_path)
        except Exception as e:
            print(f"Warning: could not write feedback_log.json: {e}")
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except Exception:
                    pass
