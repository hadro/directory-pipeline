import sys
from pathlib import Path

# main.py lives at the repo root and is not an installed package — make it
# importable as `import main` regardless of where pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
