"""MVP runner: real frames (yours, from Pollinations) -> real judge -> ranked.

The offline replay proves the mechanism on synthetic frames. THIS runs the same
critic on frames you actually generated, and scores the opinion axis with a real
Pollinations vision model instead of a hand-authored stand-in. No GPU: image
generation and the judge are both HTTP calls to Pollinations.

Two ways to get frames in:

  A. Generate them here (needs network to Pollinations):
        python3 -m ball_reel.mvp --generate
     Uses the strategy prompts below + the brief; writes mvp_input/<strategy>/NN.png.

  B. Generate them by hand on your bench and drop them in:
        mvp_input/energy_led/00.png 01.png ...   (a few frames per strategy,
        same character, simulating the jump sequence — this is the MVP stand-in
        for a real video-model clip, and it is labelled as that)
     then:  python3 -m ball_reel.mvp

Either way it prints the ranked run with a REAL opinion score and writes
mvp_report.md. Identity is the perceptual proxy unless you install the live
extra and pass --arcface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Which Pollinations models to route through. Names are the bench's; override
#: via env. Set these to whatever your Pollinations exposes.
IMG_MODEL = os.environ.get("POLLINATIONS_IMAGE_MODEL", "flux")
JUDGE_MODEL = os.environ.get("POLLINATIONS_JUDGE_MODEL", "openai")

from .brief import DEMO_BRIEF, Brief, resolve_face_ref
from .critic import DEFAULT_BAR, rank, score
from .gen import STRATEGIES

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "mvp_input"


def prompts_for(brief: Brief) -> dict[str, str]:
    """The exact prompt per strategy — what to paste into your Pollinations bench.

    Kept here so the frames you generate and the run that scores them cannot
    drift apart: one place writes the prompt.
    """
    return {s.id: f"{s.lead} {brief.subject} Vertical 9:16." for s in STRATEGIES}


def generate(brief: Brief = DEMO_BRIEF, *, n: int = 4) -> None:
    """Option A: generate n frames per strategy via Pollinations."""
    from . import pollinations
    for sid, prompt in prompts_for(brief).items():
        for i in range(n):
            out = INPUT / sid / f"{i:02d}.png"
            pollinations.image(
                prompt, out, model=IMG_MODEL, seed=brief.seed_for(sid, str(i)),
                width=brief.width, height=brief.height)
            print(f"generated {out}")


def _frames(sid: str) -> list[str]:
    d = INPUT / sid
    return sorted(str(p) for p in d.glob("*.png")) if d.exists() else []


def run(brief: Brief = DEMO_BRIEF, *, use_arcface: bool = False,
        real_judge: bool = True, bar=DEFAULT_BAR) -> dict:
    drift_fn = max_drift = None
    if use_arcface:
        from .identity import arcface_drift
        from .identity_arcface import SAME_PERSON_MAX
        drift_fn, max_drift = arcface_drift, SAME_PERSON_MAX

    scored, judged_by = [], "synthetic fallback"
    for strat in STRATEGIES:
        frames = _frames(strat.id)
        if not frames:
            continue
        opinion = None
        if real_judge:
            from . import pollinations
            # judge the first frame as the representative still
            opinion = pollinations.opinion_of(frames[0], model=JUDGE_MODEL)
            judged_by = f"Pollinations VLM (real, model={JUDGE_MODEL})"
        s = score(ROOT, strat.id, strat.label, frames, frames[0], bar,
                  drift_fn=drift_fn, max_drift=max_drift, opinion_override=opinion)
        scored.append(s)
    if not scored:
        raise FileNotFoundError(
            f"no frames under {INPUT}. Generate them (--generate) or drop yours in.")
    ranked = rank(scored)
    return {"ranked": ranked, "accepted": sum(s.accepted for s in ranked),
            "considered": len(ranked), "judged_by": judged_by,
            "identity": "arcface" if use_arcface else "proxy"}


def render(result: dict) -> str:
    out = [f"MVP run — real frames, opinion by {result['judged_by']}, "
           f"identity: {result['identity']}",
           f"accepted {result['accepted']}/{result['considered']}", ""]
    out.append(f"{'#':<3} {'strategy':<12} {'id_drift':<9} {'motion':<8} "
               f"{'opinion':<8} {'ships':<6} reason")
    out.append("-" * 76)
    for i, s in enumerate(result["ranked"], 1):
        out.append(f"{i:<3} {s.strategy_label:<12} {s.worst_drift:<9.3f} "
                   f"{s.motion:<8.3f} {s.opinion:<8.2f} "
                   f"{'yes' if s.accepted else 'no':<6} {s.reason}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if "--generate" in argv:
        generate()
        return 0
    result = run(use_arcface="--arcface" in argv,
                 real_judge="--no-judge" not in argv)
    text = render(result)
    print(text)
    (ROOT / "mvp_report.md").write_text("# MVP report\n\n```\n" + text + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
