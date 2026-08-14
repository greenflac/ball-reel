"""The critic: score a candidate clip, then split a run into what ships.

Two kinds of number, kept apart on purpose (the eval repo's central shape):

* COMPUTED, no model: identity drift and motion presence (identity.py). A
  sceptic recomputes them from the committed frames and Pillow.
* OPINION, a model's judgement: aesthetic / trend / hook. Offline these are
  SYNTHETIC hand-authored verdicts read from fixtures and labelled as such; live
  they come from a multimodal judge. This skeleton ships only the computed half
  wired end to end and a synthetic stand-in for the opinion half, so nothing
  here pretends a judge ran when it did not.

The bar is three stated gates, not one blend, because they fail differently:
identity is not tradeable (a clip that stops being the same person is unusable
however pretty), motion presence is a floor (a frozen jump does not ship), and
the opinion total is the coarse "good enough overall". All three are arguable
defaults, and moving them is a config change, not a fork.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import identity


@dataclass(frozen=True)
class Bar:
    # max_identity_drift is calibrated to the PROXY's scale, not to a face
    # encoder's. On the dhash proxy over these frames a consistent clip drifts
    # ~0.03 and a composition that changes character each frame ~0.28, so 0.20
    # sits between them; it is chosen against the frames this package has, the
    # way the eval repo anchors its constants. At live time identity_drift
    # becomes arcface_drift and this bar is re-derived on cosine-distance scale.
    max_identity_drift: float = 0.20   # computed (proxy), not tradeable
    min_motion: float = 0.02           # computed, a floor
    min_opinion: float = 0.60          # opinion, coarse overall


DEFAULT_BAR = Bar()


@dataclass
class Scored:
    strategy_id: str
    strategy_label: str
    worst_drift: float
    motion: float
    opinion: float
    opinion_synthetic: bool
    frames: int
    accepted: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "strategy_id", "strategy_label", "worst_drift", "motion",
            "opinion", "opinion_synthetic", "frames", "accepted", "reason")}


def _opinion(root: Path, strategy_id: str) -> tuple[float, bool]:
    """Read the (synthetic) opinion verdict for a strategy. (score, synthetic)."""
    v = root / "fixtures" / "judge" / f"{strategy_id}.json"
    if v.exists():
        d = json.loads(v.read_text(encoding="utf-8"))
        return float(d.get("opinion", 0.0)), bool(d.get("synthetic", True))
    return 0.0, True


def score(root: Path, strategy_id: str, strategy_label: str,
          video_frames: list[str], reference_frame: str,
          bar: Bar = DEFAULT_BAR, *, drift_fn=None, max_drift: float | None = None,
          opinion_override: float | None = None) -> Scored:
    # reference_frame is the canonical start frame: identity stability is
    # measured as each video frame's drift away from the frame the clip began on.
    # drift_fn defaults to the offline perceptual proxy; a live run passes
    # identity.arcface_drift, and max_drift moves to the cosine scale with it.
    drift_fn = drift_fn or identity.identity_drift
    max_drift = bar.max_identity_drift if max_drift is None else max_drift
    drift = drift_fn(video_frames, reference_frame)
    worst = drift["worst"][1] if drift["worst"][1] is not None else 1.0
    motion = identity.motion_presence(video_frames)["motion"]
    # opinion_override is the REAL judge's score (MVP: a Pollinations VLM). When
    # absent we fall back to the synthetic hand-authored fixture verdict.
    if opinion_override is not None:
        opinion, synthetic = float(opinion_override), False
    else:
        opinion, synthetic = _opinion(Path(root), strategy_id)

    scale = "proxy" if drift_fn is identity.identity_drift else "arcface"
    reason = ""
    if worst > max_drift:
        reason = (f"identity drift {worst:.2f} > {max_drift:.2f} "
                  f"({scale}) — face stops reading as the same person")
    elif motion < bar.min_motion:
        reason = f"motion {motion:.3f} < {bar.min_motion:.2f} — clip is near-static"
    elif opinion < bar.min_opinion:
        reason = f"opinion {opinion:.2f} < {bar.min_opinion:.2f}"
    return Scored(
        strategy_id=strategy_id, strategy_label=strategy_label,
        worst_drift=round(worst, 4), motion=round(motion, 4),
        opinion=round(opinion, 4), opinion_synthetic=synthetic,
        frames=len(video_frames), accepted=(reason == ""), reason=reason)


def rank(scored: list[Scored]) -> list[Scored]:
    """Accepted first; within that, least drift, then most motion, then opinion."""
    return sorted(scored, key=lambda s: (
        s.accepted, -s.worst_drift, s.motion, s.opinion), reverse=True)
