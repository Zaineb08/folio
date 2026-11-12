"""Point the app at a throwaway data folder, so tests never touch a real library."""

import os
import tempfile

os.environ["FOLIO_DATA"] = tempfile.mkdtemp(prefix="folio-test-")
os.environ["FOLIO_PORT"] = "8799"

from folio import config  # noqa: E402  Only once the folder above is set.

# Tests behave as on a PC with a large graphics card, whatever the PC running them has.
config.gpu_memory_gb = lambda: 12.0
