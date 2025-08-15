import os
import sys

# Ensure project root is on sys.path for imports like `services.api.main`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
	# use tab indentation as file default if any; otherwise fallback to 4 spaces
	sys.path.insert(0, ROOT)