"""Generate panels for the small bundled educational example set."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(ROOT/'src'/'mole_segmentation.py'),'--input-dir',str(ROOT/'samples'),'--output-dir',str(ROOT/'outputs')],check=True)
