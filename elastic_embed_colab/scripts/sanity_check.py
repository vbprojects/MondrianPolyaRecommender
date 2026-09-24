#!/usr/bin/env python3
"""Run from any directory: python sanity_check.py --config /path/to/saved-config.yaml."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from elastic_embed.sanity import main

if __name__ == '__main__':
    main()
