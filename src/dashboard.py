"""Flask dashboard server for MindTune-OS.

In plain English: this module serves the web dashboard that shows the live EEG
state, current music, wins history, and personalization insights. It runs on
port 5050 in a separate terminal from the main loop. The two processes
communicate through two JSON files: state.json (main loop → dashboard, updated
every 2 s) and feedback_signal.json (dashboard → main loop, written on button press).
"""

from flask import Flask, jsonify, render_template, request, url_for
from collections import defaultdict
import json
import os
import datetime
from utils import atomic_write_json

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Cache-bust static files by appending their mtime as ?v=<timestamp>.
# This means the browser always fetches fresh CSS/JS after any edit.
@app.context_processor
def inject_static_version():
    def dated_url_for(endpoint, **values):
        if endpoint == 'static':
            filename = values.get('filename')
            if filename:
                file_path = os.path.join(app.static_folder, filename)
                try:
                    values['v'] = int(os.stat(file_path).st_mtime)
                except OSError:
                    pass
        return url_for(endpoint, **values)
    return dict(url_for=dated_url_for)

BASE                  = os.path.dirname(os.path.abspath(__file__))
STATE_PATH            = os.path.join(BASE, '..', 'state.json')
LOG_PATH              = os.path.join(BASE, '..', 'wins_log.json')
FEEDBACK_LOG          = os.path.join(BASE, '..', 'feedback_log.json')
FEEDBACK_SIGNAL_PATH  = os.path.join(BASE, '..', 'feedback_signal.json')

DEFAULT_STATE = {
    "prediction": "waiting",
    "stress_count": 0,
    "music_active": False,
    "now_playing": {"track": "Not started", "artist": ""},
    "agent_reasoning": "Run: python src/main_loop.py in a second terminal",
    "interventions_tried": [],
    "wins_count": 0,
    "status_message": "Open a second terminal and run the main loop",
    "signal_saturated": False,
}


@app.after_request
def security_headers(response):
    response.headers['Cache-Control']           = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma']                  = 'no-cache'
    response.headers['X-Content-Type-Options']  = 'nosniff'
    response.headers['X-Frame-Options']         = 'DENY'
    return response


@app.route('/')
def index():
    return render_template('dashboard.html')



@app.route('/state')
def state():
    try:
        with open(STATE_PATH, encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify(DEFAULT_STATE)


@app.route('/memory')
def memory():
    try:
        with open(LOG_PATH, encoding='utf-8') as f:
            log = json.load(f)
        return jsonify(log[-20:])
    except Exception:
        return jsonify([])


@app.route('/profile')
def profile():
    """Average audio features of all wins — drives the calm audio profile card."""
    try:
        with open(LOG_PATH, encoding='utf-8') as f:
            log = json.load(f)
        wins = [e for e in log if e.get('status') == 'win' and e.get('audio_features')]
        if not wins:
            return jsonify(None)
        keys = ['tempo', 'energy', 'valence', 'acousticness', 'instrumentalness']
        result = {}
        for k in keys:
            vals = [w['audio_features'][k] for w in wins if k in w.get('audio_features', {})]
            if vals:
                result[k] = round(sum(vals) / len(vals), 2)
        result['sample_count'] = len(wins)
        return jsonify(result)
    except Exception:
        return jsonify(None)


@app.route('/sessions')
def sessions():
    """Per-session win rate and average resolution time — drives the learning progress panel."""
    try:
        with open(LOG_PATH, encoding='utf-8') as f:
            log = json.load(f)
        buckets = {}
        for entry in log:
            sn = entry.get('session_number', 1)
            if sn not in buckets:
                buckets[sn] = {'wins': 0, 'fails': 0, 'response_times': []}
            if entry['status'] == 'win':
                buckets[sn]['wins'] += 1
                rt = entry.get('response_seconds', 0)
                if rt > 0:
                    buckets[sn]['response_times'].append(rt)
            else:
                buckets[sn]['fails'] += 1
        result = []
        for sn in sorted(buckets.keys()):
            b = buckets[sn]
            total = b['wins'] + b['fails']
            avg_rt = round(sum(b['response_times']) / len(b['response_times']), 1) \
                     if b['response_times'] else None
            result.append({
                'session':               sn,
                'wins':                  b['wins'],
                'fails':                 b['fails'],
                'total':                 total,
                'win_rate':              round(b['wins'] / total * 100) if total else 0,
                'avg_response_seconds':  avg_rt,
            })
        return jsonify(result)
    except Exception:
        return jsonify([])


@app.route('/feedback')
def feedback():
    """Preference model insights — feedback counts, top/worst tags, phase."""
    try:
        with open(FEEDBACK_LOG, encoding='utf-8') as f:
            log = json.load(f)
        n_pos = sum(1 for e in log if e.get('feedback') == 1)
        n_neg = sum(1 for e in log if e.get('feedback') == 0)

        # Aggregate tag win rates
        tag_counts = defaultdict(lambda: {'pos': 0, 'neg': 0})
        for entry in log:
            fb  = entry.get('feedback')
            key = 'pos' if fb == 1 else 'neg' if fb == 0 else None
            if key:
                for tag in entry.get('tags', []):
                    tag_counts[tag][key] += 1

        tag_scores = []
        for tag, c in tag_counts.items():
            support = c['pos'] + c['neg']
            if support >= 2:
                score = round((c['pos'] + 1) / (support + 2), 2)
                tag_scores.append({'tag': tag, 'score': score, 'support': support})
        tag_scores.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({
            'total':      len(log),
            'positive':   n_pos,
            'negative':   n_neg,
            'phase':      2 if len(log) >= 10 else 1,
            'top_tags':   tag_scores[:5],
            'worst_tags': tag_scores[-5:][::-1] if len(tag_scores) >= 5 else [],
        })
    except Exception:
        return jsonify({'total': 0, 'positive': 0, 'negative': 0,
                        'phase': 1, 'top_tags': [], 'worst_tags': []})


def _write_feedback_signal(action):
    """Write a feedback signal file that main_loop.py reads on its next tick."""
    payload = {'action': action,
               'timestamp': datetime.datetime.now().isoformat()}
    if atomic_write_json(payload, FEEDBACK_SIGNAL_PATH):
        return jsonify({'ok': True, 'action': action})
    else:
        return jsonify({'ok': False, 'error': 'write failed'}), 500


@app.route('/feedback/up', methods=['POST'])
def feedback_up():
    """Signal main_loop that the current track is helping."""
    return _write_feedback_signal('up')


@app.route('/feedback/down', methods=['POST'])
def feedback_down():
    """Signal main_loop to skip the current track and try something else."""
    return _write_feedback_signal('down')


# ── Focus Mode toggle + feedback ──────────────────────────────────────────────

@app.route('/focus/on', methods=['POST'])
def focus_on():
    """Manually start Focus Mode (instrumental music via Spotify)."""
    return _write_feedback_signal('focus_on')


@app.route('/focus/off', methods=['POST'])
def focus_off():
    """Manually stop Focus Mode and return to Calm Mode monitoring."""
    return _write_feedback_signal('focus_off')


@app.route('/focus/up', methods=['POST'])
def focus_feedback_up():
    """Signal that the current focus track is helping concentration."""
    return _write_feedback_signal('focus_up')


@app.route('/focus/down', methods=['POST'])
def focus_feedback_down():
    """Signal that the current frequency isn't helping — try the next one."""
    return _write_feedback_signal('focus_down')


@app.route('/blink/double', methods=['POST'])
def blink_double():
    """Simulate an intentional double-blink command for demo."""
    return _write_feedback_signal('double_blink')


@app.route('/feedback/force_stress', methods=['POST'])
def force_stress():
    """Debug: Force stress level to 5/5 to trigger intervention."""
    return _write_feedback_signal('force_stress')


@app.route('/hardware/recalibrate', methods=['POST'])
def recalibrate():
    """Force the EEG sensor to re-run its blink calibration sequence."""
    return _write_feedback_signal('recalibrate')


if __name__ == '__main__':
    app.run(port=5050, debug=False)  # debug=False prevents double-loading
