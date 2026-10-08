import sys
from pathlib import Path

# The research package lives next to its tests (research/expert_deltas/expert_deltas); it runs in the root environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
