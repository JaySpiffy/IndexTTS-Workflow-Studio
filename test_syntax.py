# test_syntax.py
from typing import Tuple

MAX_VERSIONS_ALLOWED = 5
ProblemLine = Tuple[str, str, str, int, dict, dict, *([dict]*MAX_VERSIONS_ALLOWED), dict]

print("Minimal syntax test passed!")