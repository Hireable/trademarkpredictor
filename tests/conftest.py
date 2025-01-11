import sys
from pathlib import Path

# Set the root of the project
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))