"""Main control loop for MindTune-OS — the BCI adaptive music system.

This is the heart of the application. It runs continuously, doing three things
on every 1-second tick:

  1. Reads the next EEG sample (and Spotify audio features).
  2. Classifies the mental state (calm / relaxed / stressed).
  3. Acts: if stress is sustained, trigger an AI intervention via Spotify.

User feedback (👍 / 👎 buttons on the dashboard) is communicated through
feedback_signal.json — a tiny file this loop reads and deletes each tick.
"""

import sys, os, warnings, json, time, tempfile, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from dotenv import load_dotenv
from eeg_source import CSVReplaySource, ArduinoSerialSource
from preference_model import PreferenceModel
from focus_model import FocusPreferenceModel
from spotify_controller import (get_spotify_client, get_active_device,
    get_now_playing, save_track_to_wins, save_to_log,
    get_memory, search_and_play, get_wins_count,
    get_win_artists, get_similar_artists, get_track_tags, get_artist_tags,
    is_spotify_rate_limited)
from agent import get_music_suggestion
from neuro_apps import FocusMetrics, BlinkDetector
from utils import atomic_write_json

load_dotenv()
WINS_ID    = os.getenv('SPOTIFY_WINS_PLAYLIST_ID')
LASTFM_KEY = os.getenv('LASTFM_API_KEY')

BASE                 = os.path.dirname(os.path.abspath(__file__))
STATE_PATH           = os.path.join(BASE, '..', 'state.json')
LOG_PATH             = os.path.join(BASE, '..', 'wins_log.json')
FEEDBACK_SIGNAL_PATH = os.path.join(BASE, '..', 'feedback_signal.json')

# ── Ultradian Rhythm Awareness ────────────────────────────────────────────────
# Humans have a 90–120 minute cycle of cognitive engagement. After 90 mins,
# stress sensitivity naturally rises.
SESSION_START  = time.time()
ULTRADIAN_MINS = 90

# Maximum loop iterations per session (~277 hours at 1 s/tick — effectively unlimited).
MAX_TICKS = 999_999

# ── Sanity check mode ─────────────────────────────────────────────────────────
if '--check' in sys.argv:
    print("Imports OK — all files found — ready to run")
    sys.exit(0)


def _read_feedback_signal():
    """Return the action string from feedback_signal.json, or None. Deletes the file after reading.

    Possible values: 'up', 'down', 'focus_on', 'focus_off', 'focus_up', 'focus_down'.
    """
    if not os.path.exists(FEEDBACK_SIGNAL_PATH):
        return None
    try:
        with open(FEEDBACK_SIGNAL_PATH, encoding='utf-8') as f:
            data = json.load(f)
        os.remove(FEEDBACK_SIGNAL_PATH)
        return data.get('action')
    except Exception:
        # Corrupt or partially-written file — discard silently
        try:
            os.remove(FEEDBACK_SIGNAL_PATH)
        except Exception:
            pass
        return None


# ── EEG source ────────────────────────────────────────────────────────────────
# Auto-detect: try Arduino (mindtune_edge.ino) first; fall back to CSV replay.
# If Arduino is detected, BlinkDetector is disabled below (same serial port).
_arduino_active = False
try:
    eeg_source     = ArduinoSerialSource(port='auto')
    _arduino_active = True
    print("EEG SOURCE: Arduino hardware (mindtune_edge.ino) — 74% TinyML accuracy")
except Exception as _eeg_hw_err:
    eeg_source = CSVReplaySource()
    print(f"EEG SOURCE: CSV replay (no Arduino: {_eeg_hw_err})")

# ── Personalization models ────────────────────────────────────────────────────
pref_model  = PreferenceModel(log_path=os.path.join(BASE, '..', 'feedback_log.json'))
focus_model = FocusPreferenceModel()

# If this is a fresh start, show 0 instead of waiting for first win
if pref_model.sample_count > 0:
    print(f"Memory loaded: {pref_model.sample_count} prior feedback entries.")
else:
    print("No prior feedback found — system starting in Phase 1 (Standard).")


# ── Spotify + memory ──────────────────────────────────────────────────────────
sp = get_spotify_client()
try:
    sp.current_user()   # cheap auth probe — fails fast if token is dead
except Exception as _e:
    print(f"Spotify auth check failed ({_e}). Re-run to re-authenticate.")
    sys.exit(1)

memory        = get_memory(LOG_PATH)
wins_count    = get_wins_count(LOG_PATH)
win_artists   = get_win_artists(LOG_PATH)


# ── Neural apps: Focus Mode + Blink Remote ────────────────────────────────────
focus_metrics  = FocusMetrics()

# Blink remote — handled by ArduinoSerialSource when Arduino is active
# (mindtune_edge.ino does blink detection on-device, alongside EEG inference).
# Falls back to BlinkDetector with blink_detector.ino in CSV mode.
# _blink_ctrl is the single object used everywhere blink actions are needed.
if _arduino_active:
    blink_detector = None          # not used; eeg_source handles blinks directly
    _blink_ctrl    = eeg_source    # ArduinoSerialSource has the same blink API
else:
    try:
        blink_detector = BlinkDetector(port='auto')
        blink_detector.start()
        _blink_ctrl = blink_detector
    except Exception as _e:
        blink_detector = None
        _blink_ctrl    = None
        print(f"neuro_apps: BlinkDetector failed to start ({_e}) — Blink Remote disabled.")


def _build_lastfm_pool(artists, key):
    """Return [(similar_artist, source_win_artist), ...] — capped at 20, deduplicated."""
    if not key or not artists:
        return []
    pool     = []
    seen_sim = set()
    for source in artists[-5:]:
        similars = get_similar_artists(source, key, limit=5)
        for s in similars:
            if s.lower() not in seen_sim:
                pool.append((s, source))
                seen_sim.add(s.lower())
    return pool[:20]


lastfm_pool = _build_lastfm_pool(win_artists, LASTFM_KEY)


# ── Runtime state ─────────────────────────────────────────────────────────────
music_active             = False         # True while an intervention is playing
last_intervention_time   = None          # Unix timestamp of last Spotify play
intervention_start_stress = 0            # Stress count when music started
current_track_info       = None          # Spotify metadata {track, artist, uri}
current_track_tags       = []            # Genre/mood tags for current song
current_query            = None          # The search query that found the song
current_reason           = None          # Groq's explanation for the query
explicit_feedback_given  = False         # Prevents saving two wins for one song
skip_requested           = False         # Stress dropped but song still playing — wait for it to end
pending_win              = False         # stress dropped but song still playing — wait for it to end
pending_win_response_s   = 0            # seconds from intervention start to stress drop
agent_reasoning          = "No intervention yet"
status_message           = "Monitoring..."
band_scores              = {}            # H-1: initialised at module level
focus_mode_active        = False  # True while Focus Mode music is playing
last_blink_ts            = 0.0    # Unix timestamp of the last confirmed double-blink event
last_debug_msg           = "System initialized"
current_mode             = 'calm' # 'calm' or 'focus' — set at intervention start, persists through pending_win

timeline_points      = collections.deque(maxlen=150)
timeline_markers     = []
tick                 = 0
debug_clear_tick     = 0
recent_predictions   = []
interventions_tried  = []
now_playing_info     = None  # cached; refreshed every 5 ticks to avoid blocking the loop

# Derive session number from existing log (new session = last recorded + 1)
try:
    with open(LOG_PATH, encoding='utf-8') as _f:
        _log = json.load(_f)
    session_number = max((e.get('session_number', 1) for e in _log), default=0) + 1
except Exception:
    session_number = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_music_state(status_msg="Monitoring..."):
    """Reset all music-tracking globals to their idle state.
    
    This function ONLY handles track-level metadata. It does NOT touch
    system modes (current_mode, focus_mode_active) to prevent accidental
    mode resets after a successful intervention.
    """
    global music_active, pending_win, pending_win_response_s
    global last_intervention_time, current_query, current_reason
    global current_track_info, current_track_tags
    global intervention_start_stress, explicit_feedback_given, skip_requested
    global status_message
    music_active              = False
    pending_win               = False
    pending_win_response_s    = 0
    last_intervention_time    = None
    current_query             = None
    current_reason            = None
    current_track_info        = None
    current_track_tags        = []
    intervention_start_stress = 0
    explicit_feedback_given   = False
    skip_requested            = False
    status_message            = status_msg


def _build_win_context(stress_count, response_seconds):
    """Build the context dict passed to save_track_to_wins()."""
    stress_before = intervention_start_stress
    stress_after  = stress_count
    # efficacy: neurological stress reduction per second
    # capped at 0 to avoid negative floats if 👍 pressed during a spike
    efficacy = round(max(0, stress_before - stress_after) / max(response_seconds, 1), 4)
    
    return {
        'query':            current_query or 'unknown',
        'reason':           current_reason or 'unknown',
        'stress_before':    stress_before,
        'stress_after':     stress_after,
        'response_seconds': response_seconds,
        'efficacy':         efficacy,
        'mode':             current_mode,
        'session_number':   session_number,
    }


def _fetch_tags_for_track(track_info):
    """Return genre tags from Last.fm, falling back to artist tags."""
    if not LASTFM_KEY: return []
    tags = get_track_tags(LASTFM_KEY, track_info['artist'], track_info['track'])
    if len(tags) < 3:
        tags += get_artist_tags(LASTFM_KEY, track_info['artist'])
    return list(set(tags))[:10]


def _handle_feedback(fb_type, now, current_band_scores):
    """Record 👍 or 👎 to the appropriate personalization model."""
    global explicit_feedback_given, skip_requested, last_intervention_time
    global memory, wins_count, win_artists, lastfm_pool

    if not music_active or not current_track_info:
        return   # nothing to give feedback on

    feedback_val = 1 if fb_type == 'up' else 0
    active_model = focus_model if current_mode == 'focus' else pref_model
    
    active_model.record(
        track             = current_track_info['track'],
        artist            = current_track_info['artist'],
        query             = current_query or 'unknown',
        tags              = current_track_tags,
        eeg_band_scores   = current_band_scores,
        feedback          = feedback_val
    )
    explicit_feedback_given = True

    if fb_type == 'down':
        skip_requested = True
        # Force the 30s no-improvement retry to fire on the NEXT tick
        last_intervention_time = now - 35
    
    # Reload local memory snapshot
    memory        = get_memory(LOG_PATH)
    wins_count    = get_wins_count(LOG_PATH)
    win_artists   = get_win_artists(LOG_PATH)
    lastfm_pool   = _build_lastfm_pool(win_artists, LASTFM_KEY)


# M-6: _run_intervention defined at module level (outside the loop) to avoid
# creating 999,999 function objects over a long session.
_TAG_PREFETCH_LIMIT = 2   # Reduced from 3 to save 1-2s of Last.fm latency
tried_artists = set()     # Unordered set for O(1) lookup


def _run_intervention(severity='mild', mode='calm'):
    """Select and play a track using the three-strategy hierarchy.

    Args:
        severity: 'acute' (5/5 stressed) or 'mild' (3-4/5). Used by Strategy 0
                  to set a lower energy target for extreme stress.
        mode:     'calm' or 'focus'. Focus mode skips Strategy 0 (calm acoustic
                  recommendations) and goes straight to Strategy 1 + Groq with
                  a focus-oriented prompt (instrumental, lo-fi, mid-tempo).

    Reads from module-level globals: lastfm_pool, tried_artists, band_scores,
    sp, recent_predictions, interventions_tried, memory.
    Returns (result_dict, query_str, reason_str) or (None, None, None).

    Strategy 0 — Spotify recommendations
    Strategy 1 — Last.fm similar-artist pool, preference-model ranked
    Strategy 2 — Groq LLM agent
    """
    result = query = reason = None
    active_model = focus_model if mode == 'focus' else pref_model

    # Strategy 1 — Last.fm similar-artist pool, preference-model ranked
    # (Strategy 0 was Spotify /recommendations — deprecated 2024-11-27, removed)
    if not result and lastfm_pool and not is_spotify_rate_limited():
        # Collect eligible candidates (not yet tried this session)
        candidates = []
        for sim_artist, source_win in lastfm_pool:
            if sim_artist.lower() not in tried_artists:
                # Laplace score based on historical wins for this query/style
                # (Simple placeholder for ranking logic)
                score = active_model.score_candidate(sim_artist, [], band_scores)
                candidates.append((sim_artist, source_win, score))
        
        # Sort by ML score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        for sim_artist, source_win, _ in candidates[:_TAG_PREFETCH_LIMIT]:
            result = search_and_play(sp, sim_artist)
            if result:
                tried_artists.add(sim_artist.lower())
                query  = sim_artist
                reason = f'Last.fm similarity: "{sim_artist}" is musically related to your past win by "{source_win}"'
                break

    # Strategy 2 — Groq LLM agent
    if not result and not is_spotify_rate_limited():
        mode_val = 'focus' if mode == 'focus' else 'calm'
        query, reason = get_music_suggestion(recent_predictions, interventions_tried,
                                             memory, mode=mode_val)
        ml_conf = active_model.score_candidate(query, [], band_scores)
        if active_model.ml_active:
            reason = f'{reason} · ML: {ml_conf:.0%} EEG match'
            print(f"Groq suggestion ML confidence: {ml_conf:.0%} for '{query[:40]}'")
            if ml_conf < 0.3:
                print(f"  (low confidence — Phase 1 history suggests different style)")
        result = search_and_play(sp, query)
        # Add the played artist to tried_artists so the same artist won't be
        # suggested again this session (Groq has no awareness of tried_artists).
        if result:
            artist_key = result['artist'].lower()
            if artist_key != 'unknown artist':
                tried_artists.add(artist_key)
                # Cap tried_artists to 150 to keep Strategy 1 (Last.fm) viable in long sessions.
                if len(tried_artists) > 150:
                    # Remove the 50 oldest artists
                    oldest_to_keep = list(tried_artists)[-100:]
                    tried_artists.clear()
                    tried_artists.update(oldest_to_keep)

    return result, query, reason


def _enter_focus_mode():
    """Start a Focus Mode music episode (manual trigger only).

    Calls _run_intervention(mode='focus') which routes the Groq agent to suggest
    instrumental/lo-fi tracks. The 30-second retry and pending_win logic handle
    everything from here — no separate focus state machine needed.
    """
    global focus_mode_active, music_active, agent_reasoning, status_message
    global last_intervention_time, intervention_start_stress, current_query
    global current_reason, current_track_info, current_track_tags
    global explicit_feedback_given, skip_requested
    global pending_win, pending_win_response_s, interventions_tried, current_mode

    if focus_mode_active or music_active:
        return   # already in a mode — don't double-trigger

    result, query, reason = _run_intervention(severity='mild', mode='focus')
    interventions_tried = (interventions_tried + [query or ''])[-50:]

    if result:
        focus_mode_active         = True
        current_mode              = 'focus'
        music_active              = True
        last_intervention_time    = time.time()
        intervention_start_stress = stress_count
        current_query             = query
        current_reason            = reason
        current_track_info        = result
        current_track_tags        = _fetch_tags_for_track(result)
        explicit_feedback_given   = False
        skip_requested            = False
        pending_win               = False
        pending_win_response_s    = 0
        agent_reasoning           = reason
        status_message            = f"Focus: {result['track']}"
        print(f"FOCUS MODE ON  | playing: {result['track']}")


def _exit_focus_mode():
    """Stop Focus Mode and return to Calm Mode (manual only)."""
    global focus_mode_active, status_message, current_mode

    if not focus_mode_active:
        return
    focus_mode_active = False
    current_mode = 'calm'
    if music_active:
        try:
            sp.pause_playback()
        except Exception:
            pass
        _reset_music_state("Focus Mode off — monitoring...")
    print("FOCUS MODE OFF")


# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    for _tick in range(MAX_TICKS):

        prediction, band_scores, confidence = eeg_source.next_reading()

        recent_predictions = (recent_predictions + [prediction])[-5:]
        stress_count = recent_predictions.count('stressed')

        if len(timeline_markers) > 200:
            timeline_markers[:] = timeline_markers[-200:]

        timeline_points.append({'t': tick, 'stress': stress_count})
        tick += 1
        
        # Auto-clear simulation debug messages
        if debug_clear_tick > 0 and tick >= debug_clear_tick:
            last_debug_msg = ""
            debug_clear_tick = 0

        # ── Dashboard button feedback ─────────────────────────────────────────
        while True:
            fb = _read_feedback_signal()
            if not fb:
                break
            
            now = time.time()
            now_playing_info = get_now_playing(sp)
            
            if fb in ('up', 'down'):
                _handle_feedback(fb, now, band_scores)
            elif fb == 'focus_on':
                if music_active and not focus_mode_active:
                    try:
                        sp.pause_playback()
                    except Exception:
                        pass
                    _reset_music_state("Switching to Focus Mode...")
                _enter_focus_mode()
            elif fb == 'focus_off':
                _exit_focus_mode()
            elif fb == 'focus_up':
                _handle_feedback('up', now, band_scores)
            elif fb == 'focus_down':
                _handle_feedback('down', now, band_scores)
            elif fb == 'double_blink' and _blink_ctrl:
                last_debug_msg = "Simulating double-blink..."
                _blink_ctrl.simulate_double_blink()
            elif fb == 'force_stress':
                recent_predictions = ['stressed'] * 5
                last_debug_msg = "⚡ FORCING STRESS (Debug Mode)"
                debug_clear_tick = tick + 3
            elif fb == 'recalibrate' and hasattr(eeg_source, 'recalibrate'):
                last_debug_msg = "🔄 Recalibrating sensor..."
                debug_clear_tick = tick + 15
                eeg_source.recalibrate()

            if fb in ('up', 'focus_up') and music_active and current_track_info:
                response_seconds = int(now - last_intervention_time) if last_intervention_time else 0
                save_track_to_wins(sp, WINS_ID,
                                   context=_build_win_context(stress_count, response_seconds),
                                   track_info=current_track_info)
                timeline_markers.append({
                    't': tick - 1, 'type': 'win',
                    'label': f'✓ {current_track_info["track"][:25]}'
                })
                memory        = get_memory(LOG_PATH)
                wins_count    = get_wins_count(LOG_PATH)
                win_artists   = get_win_artists(LOG_PATH)
                lastfm_pool   = _build_lastfm_pool(win_artists, LASTFM_KEY)
                print(f"WIN saved immediately on 👍: '{current_track_info['track']}'")
                _reset_music_state("👍 Saved to Wins! Monitoring...")

        # ── Heartbeat status refresh ──────────────────────────────────────────
        # get_now_playing() is a Spotify network call — only refresh every 5 ticks
        # (5 s) to prevent a slow/hung response from blocking the EEG control loop.
        now = time.time()
        if _tick % 5 == 0:
            now_playing_info = get_now_playing(sp)

        # ── Focus Mode: update θ/β metric for dashboard display ─────────────
        focus_metrics.update(band_scores)

        # ── Blink Remote: double-blink → skip current track ───────────────────
        if _blink_ctrl:
            blink_action = _blink_ctrl.get_blink_action()
            if blink_action == 'next_track':
                last_debug_msg = "✓ DOUBLE-BLINK! Skipping..."
                debug_clear_tick = tick + 3 
                status_message = "Skipping via Blink — finding new track..."
                print("BLINK REMOTE: double-blink detected — triggering skip...")
                last_blink_ts  = time.time()
                _handle_feedback('down', now, band_scores)
                
                if music_active and current_track_info:
                    fail_context = {
                        'query':          current_query or 'unknown',
                        'reason':         current_reason or 'unknown',
                        'stress_before':  intervention_start_stress,
                        'stress_after':   stress_count,
                        'seconds_played': int(now - last_intervention_time) if last_intervention_time else 0,
                        'session_number': session_number,
                        'mode':           current_mode,
                    }
                    save_to_log(current_track_info, fail_context, 'failed', LOG_PATH)
                    timeline_markers.append({
                        't': tick - 1, 'type': 'failed',
                        'label': f'↓ {current_track_info["track"][:25]} (Blink)'
                    })
                    memory = get_memory(LOG_PATH)
                    wins_count = get_wins_count(LOG_PATH)
                
                severity = 'acute' if stress_count >= 5 else 'mild'
                result, query, reason = _run_intervention(severity, mode=current_mode)
                interventions_tried = (interventions_tried + [query or ''])[-50:]
                
                if result:
                    music_active           = True
                    last_intervention_time = now
                    current_query          = query
                    current_reason         = reason
                    current_track_info     = result
                    current_track_tags     = _fetch_tags_for_track(result)
                    status_message         = f"Skipped via Blink: {result['track']}"
                    print(f"BLINK REMOTE: Switch successful -> {result['track']}")
                else:
                    _reset_music_state("Skip failed (no tracks found) — monitoring...")
                
                skip_requested = False

        # ── Pending WIN: save once the song actually finishes ─────────────────
        if pending_win and current_track_info:
            song_ended = (
                skip_requested or
                now_playing_info is None or
                now_playing_info.get('uri') != current_track_info.get('uri')
            )
            if song_ended:
                if not skip_requested:
                    save_track_to_wins(sp, WINS_ID,
                                       context=_build_win_context(stress_count, pending_win_response_s),
                                       track_info=current_track_info)
                    timeline_markers.append({
                        't': tick - 1, 'type': 'win',
                        'label': f'✓ {current_track_info["track"][:25]}'
                    })
                    memory        = get_memory(LOG_PATH)
                    wins_count    = get_wins_count(LOG_PATH)
                    win_artists   = get_win_artists(LOG_PATH)
                    lastfm_pool   = _build_lastfm_pool(win_artists, LASTFM_KEY)
                    print(f"WIN saved: '{current_track_info['track']}' (song ended)")
                    _reset_music_state("Song finished — track saved to Wins!")
                else:
                    timeline_markers.append({
                        't': tick - 1, 'type': 'failed',
                        'label': f'↓ {current_track_info["track"][:25]}'
                    })
                    print(f"Song ended but skip was requested — not saved as win")
                    _reset_music_state("Skipped — monitoring...")

        # ── Implicit Feedback Tracking ────────────────────────────────────────
        # Gated implicit win: only record if user listens for >60s (or >90% of song)
        # AND the EEG signal actually indicates low stress (stress_count <= 2).
        # This prevents training on passive noise if the hardware is saturated.
        if music_active and current_track_info and not explicit_feedback_given and not pending_win:
            elapsed = now - last_intervention_time
            
            # Check 90% completion via Spotify progress data
            progress_ratio = 0.0
            if now_playing_info and now_playing_info.get('duration_ms', 0) > 0:
                progress_ratio = now_playing_info['progress_ms'] / now_playing_info['duration_ms']
            
            if (elapsed >= 60 or progress_ratio >= 0.90) and stress_count <= 2:
                print(f"IMPLICIT WIN detected: User listened to '{current_track_info['track']}' for {int(elapsed)}s with low stress.")
                _handle_feedback('up', now, band_scores)
                explicit_feedback_given = True  # don't record multiple times for same track

        # ── Calm Mode intervention logic ──────────────────────────────────────
        minutes_elapsed = (now - SESSION_START) / 60
        triggered = stress_count >= 3 or (minutes_elapsed > ULTRADIAN_MINS and stress_count == 2)

        if triggered and not music_active and not focus_mode_active:
            severity = 'acute' if stress_count >= 5 else 'mild'
            if minutes_elapsed > ULTRADIAN_MINS and stress_count == 2:
                status_message = f"Ultradian trough detected ({minutes_elapsed:.0f} min) — proactive intervention"
                print(f"Proactive intervention triggered at {minutes_elapsed:.1f} mins")
            
            result, query, reason = _run_intervention(severity, mode='calm')
            interventions_tried = (interventions_tried + [query or ''])[-50:]

            if result:
                music_active              = True
                last_intervention_time    = now
                intervention_start_stress = stress_count
                current_query             = query
                current_reason            = reason
                current_track_info        = result
                current_track_tags        = _fetch_tags_for_track(result)
                explicit_feedback_given   = False
                skip_requested            = False
                pending_win               = False
                pending_win_response_s    = 0
                agent_reasoning           = reason
                status_message            = f"Playing: {result['track']}"
                print(f"INTERVENTION fired | stress={stress_count}/5 | query='{query}'")

        elif triggered and music_active and last_intervention_time is not None and not pending_win:
            if now - last_intervention_time >= 30:
                if current_track_info:
                    fail_context = {
                        'query':          current_query or 'unknown',
                        'reason':         current_reason or 'unknown',
                        'stress_before':  intervention_start_stress,
                        'stress_after':   stress_count,
                        'seconds_played': 30,
                        'session_number': session_number,
                        'mode':           current_mode,
                    }
                    save_to_log(current_track_info, fail_context, 'failed', LOG_PATH)
                    timeline_markers.append({
                        't': tick - 1, 'type': 'failed',
                        'label': f'✗ {current_track_info["track"][:25]}'
                    })
                    memory = get_memory(LOG_PATH)
                    current_track_info = None

                severity = 'acute' if stress_count >= 5 else 'mild'
                result, query, reason = _run_intervention(severity, mode='calm')
                interventions_tried = (interventions_tried + [query or ''])[-50:]
                last_intervention_time = now

                if result:
                    music_active           = True
                    current_query          = query
                    current_reason         = reason
                    current_track_info     = result
                    current_track_tags     = _fetch_tags_for_track(result)
                    status_message         = f"Switching: {result['track']}"
                    print(f"RETRY successful | stress={stress_count}/5 | new query='{query}'")

        elif stress_count <= 1 and music_active and not pending_win and current_mode == 'calm':
            pending_win            = True
            pending_win_response_s = int(now - last_intervention_time) if last_intervention_time else 0
            last_intervention_time = now
            track_name = current_track_info['track'][:30] if current_track_info else '?'
            status_message = f"Stress down — finish listening to '{track_name}' (👎 to skip)"
            print(f"Stress reduced — waiting for song to finish before saving win")

        # ── Atomic state snapshot ─────────────────────────────────────────────
        pref_insights  = pref_model.get_insights()
        focus_insights = focus_model.get_insights()

        # Alpha Dominance Index: corrolate of approach motivation/valence
        # (Davidson, 1988) — single-channel proxy for positive approach.
        alpha_val = band_scores.get('Alpha', 0.0)
        beta_val  = band_scores.get('Beta', 0.0)
        gamma_val = band_scores.get('Gamma', 0.0)
        alpha_dominance = round(alpha_val / (beta_val + gamma_val + 1e-10), 2)

        system_state = {
            "prediction":          str(prediction),
            "recent_predictions":  recent_predictions,
            "stress_count":        stress_count,
            "music_active":        music_active,
            "now_playing":         now_playing_info
                                   or (current_track_info if music_active else None)
                                   or {"track": "Nothing playing", "artist": ""},
            "agent_reasoning":     agent_reasoning,
            "interventions_tried": interventions_tried,
            "wins_count":          wins_count,
            "session_number":      session_number,
            "status_message":      status_message,
            "timeline":            list(timeline_points),
            "markers":             timeline_markers,
            "band_scores":         band_scores,
            "confidence":          round(confidence, 3),
            "focus_mode_active":   focus_mode_active,
            "current_mode":        current_mode,
            "pending_win":         pending_win,
            "last_blink_ts":       last_blink_ts,
            "debug_msg":           last_debug_msg,
            "cognitive_load":      round(focus_metrics.current_ratio(), 2),
            "alpha_dominance":     alpha_dominance,
            "focus_ml_phase":      focus_insights['phase'],
            "focus_ml_entries":    focus_insights['total'],
            "feedback_count":      pref_insights['total'],
            "feedback_positive":   pref_insights['positive'],
            "feedback_negative":   pref_insights['negative'],
            "pref_phase":          pref_insights['phase'],
            "top_tags":            pref_insights['top_tags'],
            "eeg_weights":         pref_insights['eeg_weights'],
            "ml_top_tags":         pref_insights['ml_top_tags'],
            "ml_worst_tags":       pref_insights['ml_worst_tags'],
            "signal_saturated":    eeg_source.is_signal_saturated(),
        }

        atomic_write_json(system_state, STATE_PATH)

        print(f"[{_tick % 9999:4d}] {prediction:10s} | stress={stress_count}/5 "
              f"| music={'ON' if music_active else 'off'} "
              f"| fb={pref_insights['total']} "
              f"| {now_playing_info['track'] if now_playing_info else '-'}")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nShutting down cleanly...")
    eeg_source.close()
    if blink_detector:   # only set in CSV mode
        blink_detector.close()

    try:
        with open(LOG_PATH, encoding='utf-8') as f:
            full_log = json.load(f)
        wins_this_session = [e for e in full_log
                             if e.get('session_number') == session_number
                             and e.get('status') == 'win']
        if wins_this_session:
            avg_recovery = sum(e.get('response_seconds', 0)
                               for e in wins_this_session) / len(wins_this_session)
            best = min(wins_this_session, key=lambda e: e.get('response_seconds', 999))
            
            print(f"\n── Session {session_number} summary ──────────────────")
            print(f"  Wins         : {len(wins_this_session)}")
            print(f"  Avg recovery : {avg_recovery:.1f}s")
            print(f"  Best track   : '{best['track']}' ({best['response_seconds']}s)")
            if focus_mode_active:
                print(f"  Focus mode was active at exit")
            print("────────────────────────────────────────\n")
    except Exception:
        pass

    sys.exit(0)
