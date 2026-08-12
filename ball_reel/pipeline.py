"""Orchestrate one run: face + brief -> N strategies -> clips -> ranked, accepted.

Deterministic and offline by default. The multi-agent role split the vacancy
asks for maps onto these stages one-to-one:

  deconstructor -> STRATEGIES (how the reference/idea is framed, gen.py)
  analyst       -> the ball's affordance encoded in each strategy's lead
  adapter       -> start_frame: the real face placed into the scene (gen.py)
  critic        -> score + bar (critic.py), on computed axes + a synthetic opinion

This skeleton runs the whole chain on cached frames and prints a ranked table
and an acceptance line. The live generation path (ball_reel.produce) runs the
same chain end to end on the Pollinations gateway: Flux Kontext for the
face-conditioned start frame, Seedance for image-to-video, ArcFace for identity.
"""

from __future__ import annotations

from pathlib import Path

from .brief import Brief, DEMO_BRIEF, resolve_face_ref
from .critic import DEFAULT_BAR, Bar, rank, score
from .gen import STRATEGIES, Gateway


def run(root: Path, brief: Brief = DEMO_BRIEF, *, live: bool = False,
        bar: Bar = DEFAULT_BAR) -> dict:
    root = Path(root)
    gw = Gateway(root, live=live)
    _ = resolve_face_ref(brief, root)  # the input face; identity checked vs start frame
    # Live runs measure identity with the real ArcFace instrument on the cosine
    # scale; offline uses the perceptual proxy on its own (0.20) scale.
    drift_fn = max_drift = None
    if live:
        from .identity import arcface_drift
        from .identity_arcface import SAME_PERSON_MAX
        drift_fn, max_drift = arcface_drift, SAME_PERSON_MAX
    scored = []
    any_synthetic = False
    for strat in STRATEGIES:
        start = gw.start_frame(brief, strat)
        frames = gw.video(brief, strat, start)
        s = score(root, strat.id, strat.label, frames, start.path, bar,
                  drift_fn=drift_fn, max_drift=max_drift)
        any_synthetic = any_synthetic or s.opinion_synthetic
        scored.append(s)
    ranked = rank(scored)
    accepted = [s for s in ranked if s.accepted]
    return {"ranked": ranked, "accepted": len(accepted),
            "considered": len(ranked), "synthetic": any_synthetic,
            "live": live, "brief": brief.id}


def render(result: dict) -> str:
    lines = []
    lines.append(f"brief   : {result['brief']}   [9:16]")
    lines.append(f"mode    : {'LIVE' if result['live'] else 'offline (fixtures)'}")
    lines.append(f"stages  : face -> start_frame -> video -> critic")
    lines.append("")
    lines.append(f"accepted {result['accepted']}/{result['considered']}  "
                 f"(identity drift <= {DEFAULT_BAR.max_identity_drift:.2f} proxy, "
                 f"motion >= {DEFAULT_BAR.min_motion:.2f}, "
                 f"opinion >= {DEFAULT_BAR.min_opinion:.2f})")
    if result["synthetic"]:
        lines.append("  ^ opinion axis is SYNTHETIC (hand-authored fixture verdicts). "
                     "Computed axes (identity drift, motion) are real over the frames.")
    lines.append("")
    lines.append(f"{'#':<3} {'strategy':<12} {'id_drift*':<10} {'motion':<8} "
                 f"{'opinion~':<9} {'ships':<6} reason")
    lines.append("-" * 78)
    for i, s in enumerate(result["ranked"], 1):
        lines.append(f"{i:<3} {s.strategy_label:<12} {s.worst_drift:<10.3f} "
                     f"{s.motion:<8.3f} {s.opinion:<9.2f} "
                     f"{'yes' if s.accepted else 'no':<6} {s.reason}")
    lines.append("")
    lines.append("* id_drift is a PERCEPTUAL PROXY (dhash vs reference face), not a "
                 "face-embedding distance. ~opinion is synthetic offline.")
    lines.append("Live path (ball_reel.produce) runs the whole chain on ONE gateway, "
                 "Pollinations: face upload -> Flux Kontext start frame -> Seedance "
                 "image-to-video -> ArcFace identity + motion gate, retry until it holds.")
    return "\n".join(lines)
