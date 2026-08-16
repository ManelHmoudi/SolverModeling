"""Project launcher. Starts the local module menu."""

import os
import sys

# PYTHONHASHSEED is read by CPython once at interpreter startup, before any
# user code runs -- setting it via os.environ from inside an already-running
# process has no effect on that process. Re-exec with it set in the
# environment BEFORE the new interpreter starts is the only way to actually
# pin hash randomization from within the script itself.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from app import main

if __name__ == "__main__":
    main()