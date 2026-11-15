"""Remote Hands: mocap teleoperation of the Aalto G1 (see ../REMOTE_HANDS_SPEC.md, ../REMOTE_HANDS_ARCHITECTURE.md)."""
import os
import sys
from pathlib import Path

# The Aalto repo is used as a library. Default: the sibling checkout; override with $AALTO_REPO.
AALTO_REPO = Path(os.environ.get("AALTO_REPO", Path(__file__).resolve().parents[2] / "aaltonomous-humanoid"))
if str(AALTO_REPO) not in sys.path:
    sys.path.insert(0, str(AALTO_REPO))

# Local secrets and settings: remote-hands/.env (git-ignored), KEY=value lines. Real environment wins.
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
