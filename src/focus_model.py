"""Focus Mode preference model.

Focus Mode now uses Spotify music (instrumental, lo-fi, mid-tempo) instead of
focus music. The learning model is identical to the Calm Mode PreferenceModel
but stored in a separate file so focus wins/fails don't pollute calm mode history.

Everything — Phase 1 frequency scoring, Phase 2 SGD, partial_fit, get_insights() —
is inherited directly from PreferenceModel. Only the log file path differs.
"""

import os
from preference_model import PreferenceModel

BASE           = os.path.dirname(os.path.abspath(__file__))
FOCUS_LOG_PATH = os.path.join(BASE, '..', 'focus_feedback_log.json')


def FocusPreferenceModel():
    """Return a PreferenceModel configured for Focus Mode data."""
    return PreferenceModel(log_path=FOCUS_LOG_PATH)
