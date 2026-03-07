"""Groq LLM music suggestion agent for MindTune-OS.

In plain English: when the EEG classifier has detected stress and the simpler
strategies (Spotify recommendations, Last.fm similar artists) haven't found
anything to play, this module asks an LLM to suggest a Spotify search query.
It passes the user's stress history, past wins, and past failures to the LLM
so the suggestion is personalised rather than generic.
"""

import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Cache Groq client at module level to save instantiation time on every call
_client_cached = None

def _get_groq_client():
    global _client_cached
    if _client_cached is None:
        key = os.getenv('GROQ_API_KEY')
        if not key:
            return None
        _client_cached = Groq(api_key=key)
    return _client_cached


# Rotating fallback queries used when Groq is unavailable (rate limit, network, etc.)
# Diverse enough that the agent can cycle through them without repeating immediately.
_FALLBACK_QUERIES = [
    ('ambient piano stress relief',      'Fallback — gentle piano, proven for relaxation'),
    ('nature sounds rain meditation',    'Fallback — rain sounds, widely effective for calm'),
    ('Weightless Marconi Union',         'Fallback — clinically studied stress-reduction track'),
    ('soft classical strings calm',      'Fallback — slow classical, low tempo aids relaxation'),
    ('binaural beats alpha waves',       'Fallback — alpha-frequency audio targets relaxed state'),
    ('lo-fi chill study beats',          'Fallback — steady low-energy rhythm for focus'),
    ('spa music deep relaxation',        'Fallback — minimal, slow spa-style ambient'),
    ('acoustic guitar peaceful morning', 'Fallback — gentle acoustic, low valence/energy'),
    ('tibetan singing bowls meditation', 'Fallback — resonant tones used in mindfulness'),
    ('ocean waves ambient sleep',        'Fallback — consistent white-noise pattern for calm'),
]


def _fallback_query(interventions_tried):
    """Returns the first fallback query not already in interventions_tried."""
    tried_lower = {q.lower() for q in interventions_tried}
    for query, reason in _FALLBACK_QUERIES:
        if query.lower() not in tried_lower:
            return (query, reason)
    # All exhausted — cycle back to the first one
    return _FALLBACK_QUERIES[0]


def get_music_suggestion(stress_history, interventions_tried, memory, audio_profile=None, mode='calm'):
    """Queries the Groq LLM for a Spotify search query based on stress history and memory.

    Parameters:
        stress_history:      list of recent predictions e.g. ['stressed', 'stressed', 'calm']
        interventions_tried: list of queries already tried this session e.g. ['lo-fi hip hop']
        memory:              dict {'wins': [...], 'fails': [...]} from get_memory()
        mode:                'calm' (stress relief) or 'focus' (attention protection)

    Returns:
        tuple (query_string, reason_string)
    """
    # Cap memory shown in prompt to limit tokens: 5 most recent wins, 3 most recent fails
    wins_shown  = memory['wins'][-5:]
    fails_shown = memory['fails'][-3:]

    # Only show last 10 tried queries to keep prompt short
    tried_recent = interventions_tried[-10:]
    tried_str = ', '.join(tried_recent) if tried_recent else 'None tried yet'

    prompt_lines = [
        f"The system is in {mode.upper()} mode.",
        f"User has EEG readings: {stress_history}",
        f"Queries already tried this session — do not repeat: {tried_str}",
        "",
    ]

    if mode == 'focus':
        prompt_lines.append(
            "GOAL: Attention protection. Suggest high-energy, mid-tempo instrumental music "
            "without lyrics (e.g. lo-fi, deep work techno, binaural focus) to aid concentration."
        )
    else:
        prompt_lines.append(
            "GOAL: Stress relief. Suggest low-BPM, acoustic, gentle music (e.g. ambient, "
            "piano, nature sounds) to aid relaxation."
        )
    prompt_lines.append("")

    if wins_shown:
        prompt_lines.append("Past tracks/styles that WORKED for this user:")
        for line in wins_shown:
            prompt_lines.append(line)
        prompt_lines.append("")
        prompt_lines.append(
            "Use past wins to understand what STYLE and TEMPO works for this user. "
            "Suggest a fresh track in a similar style — do NOT repeat the exact same "
            "track name, artist, or search query that has already been tried."
        )
        prompt_lines.append("")

    if fails_shown:
        prompt_lines.append("Past tracks that FAILED for this user (do NOT suggest similar):")
        for line in fails_shown:
            prompt_lines.append(line)
        prompt_lines.append("")

    if audio_profile and audio_profile.get('sample_count', 0) >= 2:
        prompt_lines.append(
            f"Audio profile of this user's winning tracks "
            f"({audio_profile['sample_count']} wins averaged): "
            f"~{audio_profile.get('tempo', '?')} BPM, "
            f"energy={audio_profile.get('energy', '?')}, "
            f"valence={audio_profile.get('valence', '?')}, "
            f"acousticness={audio_profile.get('acousticness', '?')}."
        )
        prompt_lines.append(
            "Prefer tracks with similar audio characteristics to this profile."
        )
        prompt_lines.append("")

    if not wins_shown and not fails_shown:
        prompt_lines.append(
            "First session — no history. Suggest a well-established starting "
            "point for stress relief."
        )
        prompt_lines.append("")

    prompt_lines += [
        "Use the history to reason: pick something in the style of past wins "
        "but always choose a fresh track — never repeat what's already been tried.",
        "",
        "Respond in EXACTLY this format and nothing else:",
        "QUERY: <Spotify search query, 3 to 6 words>",
        "REASON: <one sentence explaining the choice, referencing past wins or avoiding past fails>",
    ]

    prompt = '\n'.join(prompt_lines)

    try:
        client = _get_groq_client()
        if not client:
            return _fallback_query(interventions_tried)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            max_completion_tokens=100,  # NOT max_tokens — that param is deprecated
        )
        text = response.choices[0].message.content
        print("Agent used: Groq")
    except Exception as e:
        print(f"Groq error: {e}")
        return _fallback_query(interventions_tried)

    # Parse QUERY: and REASON: lines — strip markdown bold/italic and leading punctuation
    # so the parser handles responses like "**QUERY:** ..." or "  QUERY: ..."
    query  = None
    reason = None
    for raw_line in text.splitlines():
        line = re.sub(r'[\*\_]+', '', raw_line).strip()  # strip ** and __ markdown
        if re.match(r'QUERY\s*:', line, re.IGNORECASE):
            query = re.split(r'QUERY\s*:', line, maxsplit=1, flags=re.IGNORECASE)[1].strip()
        elif re.match(r'REASON\s*:', line, re.IGNORECASE):
            reason = re.split(r'REASON\s*:', line, maxsplit=1, flags=re.IGNORECASE)[1].strip()

    if not query or not reason:
        print(f"Agent parse error — raw response was: {text!r}")
        return _fallback_query(interventions_tried)

    return (query, reason)


if __name__ == '__main__':
    # Test 1: no history
    result = get_music_suggestion(
        ['stressed', 'stressed', 'stressed'],
        [],
        {'wins': [], 'fails': []}
    )
    print("No history:", result)
    print()

    # Test 2: with wins and failures
    result = get_music_suggestion(
        ['stressed', 'stressed', 'stressed'],
        ['lo-fi beats'],
        {
            'wins': [
                "WIN: 'Weightless' by Marconi Union — searched 'ambient stress relief slow' "
                "because: Very slow 60bpm (stress 5/5 → 0/5 in 38s)"
            ],
            'fails': [
                "FAILED: 'Lo-Fi Study Beats' by ChillHop — searched 'lo-fi hip hop study' "
                "because: Consistent rhythm (stress stayed at 4/5 after 30s)"
            ]
        }
    )
    print("With memory:", result)
