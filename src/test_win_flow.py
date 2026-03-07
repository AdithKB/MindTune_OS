"""
Manual test for the win flow.
Run this standalone to verify:
  1. search_and_play finds and plays a track
  2. save_track_to_wins adds it to the Wins playlist
  3. wins_log.json gets a WIN entry
  4. get_memory returns the win in its wins list
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spotify_controller import (
    get_spotify_client, get_now_playing,
    search_and_play, save_track_to_wins,
    get_memory, get_wins_count
)
from dotenv import load_dotenv
load_dotenv()

WINS_ID  = os.getenv('SPOTIFY_WINS_PLAYLIST_ID')
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wins_log.json')

sp = get_spotify_client()

print("=" * 55)
print("WIN FLOW TEST")
print("=" * 55)

# 1. Check device
now = get_now_playing(sp)
print(f"\n[1] Currently playing: {now}")

# 2. Search and play a track
print("\n[2] Searching and playing test track...")
result = search_and_play(sp, 'weightless marconi union')
if not result:
    print("ERROR: search_and_play returned None — is Spotify open?")
    sys.exit(1)
print(f"    Playing: '{result['track']}' by {result['artist']}")

# 3. Check wins count before
count_before = get_wins_count(LOG_PATH)
print(f"\n[3] Wins log count before: {count_before}")

# 4. Save to Wins playlist + log
win_context = {
    'query':            'weightless marconi union',
    'reason':           'Extremely slow BPM (60bpm) — clinically studied for cortisol reduction',
    'stress_before':    4,
    'stress_after':     0,
    'response_seconds': 42,
}
print("\n[4] Saving to Wins playlist and logging...")
save_track_to_wins(sp, WINS_ID, context=win_context)

# 5. Verify log count increased
count_after = get_wins_count(LOG_PATH)
print(f"\n[5] Wins log count after:  {count_after}")
if count_after > count_before:
    print("    PASS — WIN entry written to log")
else:
    print("    FAIL — log count did not increase")

# 6. Verify wins_log.json
memory = get_memory(LOG_PATH)
print(f"\n[6] Memory wins ({len(memory['wins'])} entries):")
for line in memory['wins']:
    print(f"    {line}")
print(f"    Memory fails ({len(memory['fails'])} entries):")
for line in memory['fails']:
    print(f"    {line}")

if memory['wins']:
    print("\n    PASS — WIN entry written to wins_log.json")
else:
    print("\n    FAIL — no win entries found in wins_log.json")

# 7. Full memory state
print(f"\n[7] Full memory state:")
print(f"    wins:  {len(memory['wins'])} entries")
print(f"    fails: {len(memory['fails'])} entries")

print("\n" + "=" * 55)
print("TEST COMPLETE")
print("=" * 55)
