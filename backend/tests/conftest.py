# backend/tests/conftest.py
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour imports relatifs
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))
