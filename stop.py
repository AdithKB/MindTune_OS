#!/usr/bin/env python3
"""stop.py — cross-platform process stopper for MindTune-OS.

Equivalent to stop.sh but works on Windows, macOS, and Linux.
Usage:
    python stop.py
"""

import sys
from launch import do_stop

if __name__ == "__main__":
    do_stop()
