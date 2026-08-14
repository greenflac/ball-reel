"""The generation gateway: start-frame -> video, offline by default.

Three stages, one interface, and every live call sits behind a flag that is off
by default — the same shape as the eval repo's gateway, and for the same
reason: the demo must replay with no key, no network and nothing spent, and a
live run must be a deliberate act.

STAGES
  1. start_frame  -- a face-conditioned still: the reference face placed into
     the ball-jump scene. Live: an image model with an identity/reference
     adapter (InstantID/PuLID-class, or Flux Kontext). Offline: a cached frame.
  2. video        -- start frame -> motion. Live: an image-to-video model
     (Kling / Seedance) driven by the start frame. Offline: a cached frame
     sequence. NOTE: no such model is wired yet — this is the real gap, and the
     live path raises rather than pretends.
  3. voice        -- (optional) TTS in Russian + audio-driven lipsync on a
     talking start frame. Live: an RU-TTS voice + a lipsync model
     (LivePortrait / Sync). Offline: a cached, silent stand-in. The lipsync is
     driven by AUDIO, not language — "Russian" is a property of the TTS layer.

Every offline artefact is drawn locally (see tools/make_fixtures.py). The
verdicts a critic reads are hand-authored in that tool and are SYNTHETIC — the
CLI says so, exactly as the eval repo does.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .brief import Brief, resolve_face_ref


class LiveNotWired(RuntimeError):
    """Raised when a live stage is asked for but its model is not integrated.

    Honesty over pretence: an unwired live path fails loudly instead of silently
    returning a stand-in dressed as a real call.
    """


@dataclass(frozen=True)
class Strategy:
    """One framing of the same brief. The prompt-engineering lever."""

    id: str
    label: str
    lead: str
    hypothesis: str


#: A small, real set of different bets (not one prompt re-seeded).
STRATEGIES: tuple[Strategy, ...] = (
    Strategy("energy_led", "Energy-led",
             "Peak of the jump, both feet off the ball, hair and light in motion.",
             "The airborne instant stops the thumb faster than a calm pose."),
    Strategy("product_led", "Product-led",
             "The ball dominant and legible in the lower third, the person on it above.",
             "A clear product read matters more than the athlete for a shopping brief."),
    Strategy("routine_led", "Routine-led",
             "A with-me framing: one calm rep, timestamp-friendly, home-studio calm.",
             "Companion/with-me framing carries retention on lifestyle subjects."),
)


@dataclass(frozen=True)
class Frame:
    stage: str
    strategy_id: str
    path: str
    from_cache: bool


class Gateway:
    """Offline replay from fixtures; live behind an explicit flag."""

    def __init__(self, root: Path, *, live: bool = False):
        self.root = Path(root)
        self.live = live
        self._index = self._load_index()

    def _load_index(self) -> dict:
        idx = self.root / "fixtures" / "index.json"
        if idx.exists():
            return json.loads(idx.read_text(encoding="utf-8"))
        return {}

    def _cached(self, stage: str, strategy_id: str) -> str | None:
        key = f"{stage}/{strategy_id}"
        rel = self._index.get(key)
        if not rel:
            return None
        p = self.root / rel
        return str(p) if p.exists() else None

    def start_frame(self, brief: Brief, strategy: Strategy) -> Frame:
        """Face-conditioned start still for one strategy."""
        cached = self._cached("start", strategy.id)
        if not self.live:
            if cached is None:
                raise FileNotFoundError(
                    f"no cached start frame for {strategy.id}; run "
                    f"tools/make_fixtures.py first, or pass live=True with a model."
                )
            return Frame("start", strategy.id, cached, from_cache=True)
        # Live: an image model with a face-identity adapter (+ optional LoRA),
        # conditioned on the reference face. See live_gen.start_frame_live.
        from . import live_gen

        face = resolve_face_ref(brief, self.root)
        out = self.root / "live_out" / "start" / f"{strategy.id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        prompt = f"{strategy.lead} {brief.subject}"
        path = live_gen.start_frame_live(
            prompt, face, brief.seed_for("start", strategy.id),
            brief.width, brief.height, out,
            lora_path=os.environ.get("CHARACTER_LORA") or None,
        )
        return Frame("start", strategy.id, path, from_cache=False)

    def video(self, brief: Brief, strategy: Strategy, start: Frame) -> list[str]:
        """Start frame -> a sequence of video frames (offline: a cached set)."""
        if not self.live:
            frames = sorted(
                str(p) for p in (self.root / "fixtures" / "video" / strategy.id).glob("*.png")
            )
            if not frames:
                raise FileNotFoundError(
                    f"no cached video frames for {strategy.id}; run make_fixtures.py."
                )
            return frames
        # Live: image-to-video (Kling/Seedance). See live_gen.video_live.
        from . import live_gen

        motion = (f"{strategy.lead} The person jumps on the fitness ball: it "
                  f"compresses and rebounds, they leave and land back on it.")
        out_dir = self.root / "live_out" / "video" / strategy.id
        return live_gen.video_live(start.path, motion, out_dir)

    def voice(self, brief: Brief, start: Frame, script_ru: str = "") -> str | None:
        """Optional: RU-TTS + audio-driven lipsync on a talking start frame."""
        if not self.live:
            return None  # silent stand-in offline; the report says so
        if not script_ru:
            return None  # no line to speak
        from . import live_gen

        out = self.root / "live_out" / "voice" / f"{start.strategy_id}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        return live_gen.voice_live(start.path, script_ru, out)
