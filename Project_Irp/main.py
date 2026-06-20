"""Project launcher. Starts the local module menu."""

import os
os.environ.setdefault("PYTHONHASHSEED", "0")

from app import main

if __name__ == "__main__":
    main()