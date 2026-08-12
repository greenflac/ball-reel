"""The brief for one ball-reel: a face, a subject, overlay copy, a shape.

One text brief plus a face photo becomes the whole run. The face is the only
binary input; everything else is text, exactly as the eval repo takes a text
brief and turns it into prompts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Brief:
    id: str
    #: Path to the reference FACE photo. The one non-text input. The character
    #: in every generated frame must read as this person.
    face_ref: str
    #: What happens in the clip. The product (a gymnastics/fitness ball) and the
    #: action (a person jumping on it) live here as words.
    subject: str
    #: Lines of copy to render on the frame. Rendered on the STILL layers only —
    #: never baked into the moving region, because video models melt text.
    overlay_text: dict[str, str] = field(default_factory=dict)
    width: int = 1080
    height: int = 1920
    #: Clip length (s) and frame shape handed to the video model. 9:16 for reels.
    duration: int = 4
    aspect_ratio: str = "9:16"

    def seed_for(self, *parts: str) -> int:
        """Deterministic seed so an offline replay is exact."""
        digest = hashlib.md5(
            "|".join([self.id, *parts]).encode("utf-8", "ignore")
        ).hexdigest()
        return int(digest[:8], 16)


#: The shipped demo brief. A real face path is supplied at run time; this is the
#: text half, committed so the offline replay is exact.
DEMO_BRIEF = Brief(
    id="ball_jump_01",
    face_ref="fixtures/face_ref.png",
    subject=(
        "A person bouncing on a large fitness/gymnastics ball in a bright home "
        "studio, mid-jump, natural morning light, energetic and real."
    ),
    overlay_text={
        "headline": "Bounce your morning awake",
        "sub": "10 minutes, one ball",
    },
)


def resolve_face_ref(brief: Brief, root: Path) -> str:
    """Absolute path to the brief's reference face, relative to a package root."""
    p = Path(brief.face_ref)
    return str(p if p.is_absolute() else root / p)
