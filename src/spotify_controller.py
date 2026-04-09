"""Spotify and Last.fm integration for MindTune-OS.

In plain English: this module handles everything related to music playback and
music metadata. It talks to the Spotify API (search, play)
and the Last.fm API (get genre tags for tracks and artists). It also reads and
writes wins_log.json — the persistent memory of which tracks helped reduce stress.
"""

import os
import json
import time
import datetime
import tempfile
import urllib.request
import urllib.parse
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

SCOPE = ('user-read-playback-state user-modify-playback-state '
         'user-library-modify user-library-read '
         'playlist-modify-public playlist-modify-private '
         'playlist-read-private playlist-read-collaborative')

# .cache is written to project root (one level up from src/)
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.cache')


def get_spotify_client():
    """Returns an authenticated spotipy.Spotify object using SpotifyOAuth.

    First run: opens a browser for Spotify login. After logging in, copy the
    full redirect URL from the browser address bar and paste it into the
    terminal when prompted. This only happens once — spotipy saves a .cache
    token file and reuses it on every subsequent run.
    """
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv('SPOTIFY_CLIENT_ID'),
            client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
            redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),  # must be http://127.0.0.1:8888/callback
            scope=SCOPE,
            cache_path=CACHE_PATH,
        ),
        requests_timeout=5,  # prevent hung connections from blocking the main loop
        retries=0,           # fail fast on 429 — spotipy's default (retries=3) sleeps for
                             # the full Retry-After value (can be hours) before raising
    )
    return sp


def get_active_device(sp):
    """Returns device_id of the currently active Spotify device, or None.

    Prefers is_active=True (the device currently playing) over the first
    device in the list, so playback goes to the right device when multiple
    are visible (e.g. laptop + TV).
    """
    try:
        result = sp.devices()
    except Exception as e:
        print(f"Warning: could not fetch Spotify devices: {e}")
        return None
    devices = (result or {}).get('devices', [])
    if not devices:
        print("ERROR: No active device. Open Spotify and play something first.")
        return None
    for d in devices:
        if d['is_active']:
            return d['id']
    return devices[0]['id']


def get_now_playing(sp):
    """Returns dict {track, artist, uri} if something is playing, else None."""
    try:
        result = sp.current_playback()
    except Exception:
        return None
    if result is None or not result.get('is_playing'):
        return None
    # item is None when a podcast, ad, or local file is playing — skip gracefully
    if result.get('item') is None:
        return None
    track_name  = result['item']['name']
    artists     = result['item'].get('artists', [])
    artist_name = artists[0]['name'] if artists else 'Unknown Artist'
    track_uri   = result['item']['uri']
    progress_ms = result.get('progress_ms', 0)
    duration_ms = result['item'].get('duration_ms', 0)
    return {
        'track':       track_name,
        'artist':      artist_name,
        'uri':         track_uri,
        'progress_ms': progress_ms,
        'duration_ms': duration_ms
    }


def save_to_log(track_info, context, status, log_path):
    """Appends one win or failure entry to wins_log.json.

    status: 'win' or 'failed'
    track_info: dict with keys track, artist, uri
    context: dict with query, reason, stress_before, stress_after,
             and either response_seconds (win) or seconds_played (failed)
    """
    try:
        with open(log_path, encoding='utf-8') as f:
            log = json.load(f)
    except Exception:
        log = []

    entry = {'timestamp': datetime.datetime.now().isoformat(), 'status': status}
    entry['track']        = track_info['track']
    entry['artist']       = track_info['artist']
    entry['uri']          = track_info['uri']
    entry['query']        = context['query']
    entry['reason']       = context['reason']
    entry['stress_before'] = context['stress_before']
    if 'session_number' in context:
        entry['session_number'] = context['session_number']

    if status == 'win':
        entry['stress_after']     = context['stress_after']
        entry['response_seconds'] = context['response_seconds']
    else:
        entry['stress_after']  = context['stress_after']
        entry['seconds_played'] = context.get('seconds_played', 30)

    log.append(entry)
    # Atomic write — prevents corrupt JSON if the process is killed mid-write
    try:
        dir_path = os.path.dirname(os.path.abspath(log_path))
        with tempfile.NamedTemporaryFile('w', dir=dir_path,
                                         delete=False, suffix='.tmp') as tmp:
            tmp_name = tmp.name
            json.dump(log, tmp, indent=2)
        os.replace(tmp_name, log_path)
        print(f"Memory saved [{status.upper()}]: '{track_info['track']}'")
    except Exception as e:
        print(f"Warning: could not write wins_log.json: {e}")
        try:
            os.unlink(tmp_name)
        except Exception:
            pass


def save_track_to_wins(sp, wins_playlist_id, context=None, track_info=None):
    """Logs a WIN to wins_log.json.

    Uses track_info if provided (preferred — avoids re-querying Spotify and
    prevents mis-attribution when the user manually changes tracks).
    Falls back to get_now_playing() only when track_info is not supplied.

    wins_log.json is the source of truth for the dashboard and agent memory.
    Spotify playlist/library writes are skipped — both are blocked for
    Development Mode apps by Spotify's Extended Access policy (Nov 2024).
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wins_log.json')

    if track_info is None:
        track_info = get_now_playing(sp)
    if not track_info:
        print("Warning: track info unavailable — cannot save win.")
        return

    # Always write to log
    if context:
        save_to_log(track_info, context, 'win', log_path)

    # Spotify write APIs (library + playlist) are blocked in Development Mode
    # by Spotify's Extended Access policy (changed Nov 2024) — silently skipped.
    # wins_log.json is the source of truth; dashboard and agent both read from it.


def get_memory(log_path):
    """Reads wins_log.json and returns {'wins': [...], 'fails': [...]} as human-readable strings."""
    try:
        with open(log_path, encoding='utf-8') as f:
            log = json.load(f)
    except Exception:
        return {'wins': [], 'fails': []}

    wins  = []
    fails = []

    for entry in log[-20:]:  # only 20 most recent entries
        if entry['status'] == 'win':
            line = (
                f"WIN: '{entry['track']}' by {entry['artist']} — "
                f"searched '{entry['query']}' because: {entry['reason']} "
                f"(stress {entry.get('stress_before','?')}/5 → {entry.get('stress_after','?')}/5 "
                f"in {entry.get('response_seconds','?')}s)"
            )
            wins.append(line)
        else:
            line = (
                f"FAILED: '{entry['track']}' by {entry['artist']} — "
                f"searched '{entry['query']}' because: {entry['reason']} "
                f"(stress stayed at {entry['stress_after']}/5 after 30s — did not help)"
            )
            fails.append(line)

    return {'wins': wins, 'fails': fails}


def get_wins_tracks(sp, wins_playlist_id):
    """Returns list of 'Track Name by Artist Name' strings from the Wins playlist."""
    try:
        result = sp.playlist_items(wins_playlist_id)
        tracks = []
        for item in result['items']:
            if item['track'] is None:
                continue
            name    = item['track']['name']
            artists = item['track'].get('artists', [])
            artist  = artists[0]['name'] if artists else 'Unknown Artist'
            tracks.append(f"{name} by {artist}")
        return tracks
    except Exception:
        return []


def get_win_artists(log_path):
    """Returns unique artist names from WIN entries in wins_log.json, most recent first.

    Used to seed Last.fm similar-artist lookups.
    """
    try:
        with open(log_path, encoding='utf-8') as f:
            log = json.load(f)
        seen   = set()
        result = []
        for entry in reversed(log):
            if entry.get('status') == 'win' and entry.get('artist'):
                a = entry['artist']
                if a not in seen:
                    seen.add(a)
                    result.append(a)
        return result
    except Exception:
        return []


def get_similar_artists(artist_name, api_key, limit=20):
    """Query Last.fm artist.getSimilar and return a list of similar artist name strings.

    Uses stdlib urllib — no extra packages required.
    Returns [] on any network or API error, or when api_key is not set.
    Last.fm free API: https://www.last.fm/api/account/create
    """
    if not api_key or not artist_name:
        return []

    params = urllib.parse.urlencode({
        'method':  'artist.getSimilar',
        'artist':  artist_name,
        'api_key': api_key,
        'format':  'json',
        'limit':   limit,
    })
    url = f'https://ws.audioscrobbler.com/2.0/?{params}'

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        artists = data.get('similarartists', {}).get('artist', [])
        return [a['name'] for a in artists if a.get('name')]
    except Exception as e:
        print(f"Last.fm error for '{artist_name}': {e}")
        return []


# Module-level rate-limit gate: when Spotify returns 429, all search_and_play
# calls return None instantly until the cooldown expires. This prevents the
# intervention loop from blocking the main loop for 9+ seconds per tick.
_spotify_blocked_until = 0.0

def is_spotify_rate_limited():
    """Return True if we are currently in a Spotify 429 cooldown."""
    return time.time() < _spotify_blocked_until

def _set_spotify_blocked(retry_after_seconds=60):
    global _spotify_blocked_until
    _spotify_blocked_until = time.time() + retry_after_seconds
    print(f"Spotify rate-limited — skipping all searches for {retry_after_seconds:.0f}s")


def search_and_play(sp, query, retries=2):
    """Searches Spotify for query, plays the first result. Returns track dict or None.
    Fails immediately (no sleep) if currently rate-limited to keep the main loop fluid.
    """
    global _spotify_blocked_until

    if is_spotify_rate_limited():
        remaining = int(_spotify_blocked_until - time.time())
        print(f"Spotify blocked ({remaining}s remaining) — skipping '{query}'")
        return None

    for attempt in range(retries):
        try:
            results = sp.search(q=query, type='track', limit=2)
            break
        except Exception as e:
            err_str = str(e)
            if '429' in err_str:
                # Extract Retry-After if present, else default to 60s cooldown
                import re
                match = re.search(r'Retry will occur after[:\s]+(\d+)', err_str)
                cooldown = int(match.group(1)) if match else 60
                _set_spotify_blocked(cooldown)
                return None   # bail immediately — no point retrying
            if attempt < retries - 1:
                print(f"Warning: Spotify search failed for '{query}' (attempt {attempt+1}): {e}. Retrying...")
                time.sleep(1)
            else:
                print(f"Warning: Spotify search failed for '{query}': {e}")
                return None

    items = (results or {}).get('tracks', {}).get('items', [])
    if not items:
        print(f"Warning: no results for query '{query}'")
        return None

    track      = items[0]
    track_uri  = track.get('uri', '')
    track_name = track.get('name', 'Unknown Track')
    artists    = track.get('artists', [])
    artist     = artists[0]['name'] if artists else 'Unknown Artist'

    if not track_uri:
        print(f"Warning: track URI missing for query '{query}'")
        return None

    device_id = get_active_device(sp)
    if device_id is None:
        print("Warning: no active device — cannot start playback.")
        return None

    for attempt in range(retries):
        try:
            sp.start_playback(device_id=device_id, uris=[track_uri])
            print(f"Now playing: '{track_name}' by {artist}")
            return {'track': track_name, 'artist': artist, 'uri': track_uri}
        except Exception as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt, 1)
                print(f"Error starting playback (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Error starting playback after {retries} attempts: {e}")
                return None


def get_artist_tags(artist_name, api_key, limit=10):
    """Return a list of Last.fm tag strings for an artist.

    Used to pre-fetch music features for candidate artists BEFORE playing so
    the preference model can score them with real tags (not empty []).
    Returns [] on any error or when api_key is not set.
    """
    if not api_key or not artist_name:
        return []

    params = urllib.parse.urlencode({
        'method':  'artist.getTopTags',
        'artist':  artist_name,
        'api_key': api_key,
        'format':  'json',
        'limit':   limit,
    })
    url = f'https://ws.audioscrobbler.com/2.0/?{params}'
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        tags = (data.get('toptags', {}) or {}).get('tag', [])
        return [t['name'].lower() for t in tags if t.get('name')][:limit]
    except Exception:
        return []


def get_track_tags(track_name, artist_name, api_key, limit=10):
    """Return a list of Last.fm tag strings for a specific track.

    Falls back to artist-level tags if track tags are sparse (< 3 results).
    Returns [] on any error or when api_key is not set.

    Used by the preference model to build music feature vectors:
        tags like 'ambient', 'piano', 'instrumental' become binary features
        that the model correlates with EEG state → feedback outcomes.
    """
    if not api_key or not track_name or not artist_name:
        return []

    def _fetch_tags(method, params):
        params.update({'api_key': api_key, 'format': 'json', 'limit': limit})
        url = f'https://ws.audioscrobbler.com/2.0/?{urllib.parse.urlencode(params)}'
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            # track.getTopTags returns data['toptags']['tag']
            # artist.getTopTags returns same shape
            tags = (data.get('toptags', {}) or {}).get('tag', [])
            return [t['name'].lower() for t in tags if t.get('name')]
        except Exception:
            return []

    tags = _fetch_tags('track.getTopTags',
                       {'method': 'track.getTopTags',
                        'track': track_name, 'artist': artist_name})

    # If track has < 3 tags (obscure track), fall back to artist-level tags
    if len(tags) < 3:
        artist_tags = _fetch_tags('artist.getTopTags',
                                  {'method': 'artist.getTopTags',
                                   'artist': artist_name})
        # Merge, deduplicate, keep track tags first
        seen = set(tags)
        for t in artist_tags:
            if t not in seen:
                seen.add(t)
                tags.append(t)

    return tags[:limit]






def get_wins_count(log_path):
    """Returns the number of WIN entries in wins_log.json."""
    try:
        with open(log_path, encoding='utf-8') as f:
            log = json.load(f)
        return sum(1 for e in log if e.get('status') == 'win')
    except Exception:
        return 0


if __name__ == '__main__':
    sp     = get_spotify_client()
    device = get_active_device(sp)
    print("Active device:", device)

    print("Now playing:", get_now_playing(sp))

    result = search_and_play(sp, 'ambient piano stress relief')
    print("Test search:", result)

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wins_log.json')
    print("Memory:", get_memory(log_path))
