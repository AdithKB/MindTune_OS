"""Shared utility functions for MindTune-OS.

In plain English: this module contains small helper functions used by
multiple parts of the system. Keeping them here avoids copy-pasting the
same logic across files and ensures every module uses the same
battle-tested implementation.

Current contents:
  atomic_write_json — safely write a JSON file without risking corruption.
"""

import os
import json
import tempfile

def atomic_write_json(data, target_path):
    """Write data to a JSON file atomically using a temporary file.
    
    This ensures that if the process is killed or the computer loses power
    during the write, the target file is never left in a corrupted/partial state.
    """
    tmp_name = None
    try:
        dir_path = os.path.dirname(os.path.abspath(target_path))
        with tempfile.NamedTemporaryFile('w', dir=dir_path, delete=False, suffix='.tmp') as tmp:
            tmp_name = tmp.name
            json.dump(data, tmp)
        os.replace(tmp_name, target_path)
        return True
    except Exception as e:
        print(f"Warning: atomic_write_json failed for {target_path} ({e})")
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
        return False
