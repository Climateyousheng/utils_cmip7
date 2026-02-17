#!/usr/bin/env python3
"""
Thin wrapper around the installed CLI entry-point utils-cmip7-validate-ppe.

Exists for development convenience (run directly without installing the package).

Usage:
    python scripts/validate_ppe.py xqhuc
    python scripts/validate_ppe.py xqhuc --top-n 20 --top-k 40
    python scripts/validate_ppe.py xqhuc --highlight xqhua,xqhub
    python scripts/validate_ppe.py xqhuc --param-viz --param-viz-vars GPP NPP CVeg

For full option reference see: utils-cmip7-validate-ppe --help
"""

import sys
from pathlib import Path

# Add src to path for development installs (no-op when package is installed)
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from utils_cmip7.cli import validate_ppe_cli

if __name__ == '__main__':
    validate_ppe_cli()
