"""Bağımsız Blender ile sentetik üretim başlatıcısı."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import bpy  # noqa: F401
except ImportError:
    sys.stderr.write("HATA: Bu script Blender icinde calismali.\n")
    raise SystemExit(2) from None

from miragebench.synth.cli import main  # noqa: E402

raise SystemExit(main(sys.argv))
