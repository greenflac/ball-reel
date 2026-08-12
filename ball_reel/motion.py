"""Whether the clip MOVES well: does it loop, and is the motion physical.

Identity asks "is this the same person". These ask the other two questions a
reel actually fails on — it cuts visibly when it repeats, or the motion is
generator soup: a limb teleports, the body morphs between frames, the subject
freezes for half the clip.

Both measures here are RATIOS against the clip's own motion, not absolute pixel
thresholds. That matters: a calm clip and a violent one have completely
different frame-to-frame magnitudes, so any fixed number would be tuned to one
and wrong for the other. Dividing by the clip's own median step makes the same
threshold mean the same thing across clips.

numpy only — no model, no network. The arithmetic is unit-tested on synthetic
sequences, so a sceptic can recompute every number here.
"""

from __future__ import annotations

from pathlib import Path

#: A loop is seamless when the first->last step is no larger than this multiple
#: of an ordinary step inside the clip. Measured live on seedance-2.0: passing
#: the start frame as BOTH keyframes gave 0.09, while the same prompt with only
#: a start frame gave 0.61 (a visible jump). 0.30 sits between those, closer to
#: the good one, so it accepts a real loop and rejects a merely-similar ending.
SEAMLESS_MAX = 0.30

#: A single step this many times the median step is a discontinuity, not motion
#: — the mark of a teleport or a morph rather than a body moving.
JUMP_MAX = 4.0

#: Below this, in the same normalised units, nothing is happening.
STILL_MIN = 0.15


def _gray(path: str | Path, side: int = 96):
    """A small grayscale array. Downscaled so the measure tracks BODY movement
    rather than sensor noise and compression shimmer."""
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        small = im.convert("L").resize((side, side), Image.BILINEAR)
    return np.asarray(small, dtype="float64")


def _steps(frames: list[str]) -> list[float]:
    """Mean absolute difference between each adjacent pair of frames."""
    import numpy as np

    arrs = [_gray(f) for f in frames]
    return [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]


def loop_seam(frames: list[str]) -> dict:
    """How visible the cut is when the clip repeats.

    ``ratio`` is the first->last difference over the MEDIAN adjacent step. Below
    1.0 the seam is smaller than an ordinary frame transition, i.e. the repeat
    is less visible than the motion already on screen.

    The median, not the mean, sets the scale: one morph artefact would inflate a
    mean and quietly make a bad loop look acceptable.
    """
    import numpy as np

    if len(frames) < 3:
        return {"ratio": None, "seam": None, "typical_step": None,
                "seamless": False, "note": "need at least 3 frames to judge a loop."}
    steps = _steps(frames)
    typical = float(np.median(steps))
    seam = float(np.abs(_gray(frames[-1]) - _gray(frames[0])).mean())
    if typical == 0:
        return {"ratio": None, "seam": round(seam, 3), "typical_step": 0.0,
                "seamless": False, "note": "the clip does not move at all."}
    ratio = seam / typical
    return {"ratio": round(ratio, 3), "seam": round(seam, 3),
            "typical_step": round(typical, 3),
            "seamless": ratio <= SEAMLESS_MAX,
            "note": (f"loop seam {ratio:.2f}x a typical frame step "
                     f"({'seamless' if ratio <= SEAMLESS_MAX else 'visible cut on repeat'}; "
                     f"bar {SEAMLESS_MAX}).")}


def motion_quality(frames: list[str]) -> dict:
    """Is the movement continuous and physical, or does it jump and morph.

    Returns ``worst_jump`` (largest step over the median), ``jumps`` (their
    indices), ``moving`` (there is motion at all) and ``smooth``. A generator
    that loses the thread produces one enormous step between two frames while
    the rest are ordinary — exactly what the ratio exposes.
    """
    import numpy as np

    if len(frames) < 3:
        return {"worst_jump": None, "jumps": [], "moving": False, "smooth": False,
                "activity": None, "note": "need at least 3 frames to judge motion."}
    steps = _steps(frames)
    typical = float(np.median(steps))
    if typical == 0:
        return {"worst_jump": None, "jumps": [], "moving": False, "smooth": False,
                "activity": 0.0, "note": "static clip: nothing moves."}
    ratios = [s / typical for s in steps]
    worst = max(ratios)
    jumps = [i for i, r in enumerate(ratios) if r > JUMP_MAX]
    # Activity is the typical step against the frame's own brightness scale, so
    # "nothing happens" is separated from "the camera is just dark".
    activity = typical / max(float(np.mean(_gray(frames[0]))), 1.0)
    moving = activity >= STILL_MIN / 10
    return {"worst_jump": round(worst, 3), "jumps": jumps, "moving": moving,
            "smooth": not jumps, "activity": round(activity, 4),
            "note": (f"largest frame step {worst:.1f}x the median"
                     + (f"; {len(jumps)} discontinuity(ies) at {jumps} — "
                        f"limbs teleport or the body morphs there"
                        if jumps else "; motion is continuous") + ".")}


def best_loop_cut(frames: list[str], *, min_keep: float = 0.5) -> dict:
    """Find where to cut so the clip loops, without generating anything new.

    An end-frame keyframe is the cheap way to get a loop, but only models that
    DECLARE `end_frame` accept one — the rest ignore the second keyframe
    silently (measured: veo, which declares it, closed at 0.17; wan, which does
    not, came back at 1.71). When no end frame is available, the clip usually
    still PASSES THROUGH a pose close to its opening one — so the loop becomes a
    trimming problem rather than a generation one.

    Scans candidate end frames in the last ``1 - min_keep`` of the clip and
    returns the one closest to frame 0, with the seam ratio it achieves. Costs
    one decode, no tokens, and cannot make the clip worse: if nothing beats the
    original ending, ``cut_at`` is the last frame.
    """
    import numpy as np

    if len(frames) < 4:
        return {"cut_at": len(frames) - 1, "ratio": None, "seamless": False,
                "note": "too few frames to search for a loop point."}
    arrs = [_gray(f) for f in frames]
    steps = [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]
    typical = float(np.median(steps)) or 1.0
    first = int(len(frames) * min_keep)
    scores = {i: float(np.abs(arrs[i] - arrs[0]).mean()) / typical
              for i in range(max(first, 2), len(frames))}
    cut = min(scores, key=lambda i: scores[i])
    ratio = round(scores[cut], 3)
    kept = (cut + 1) / len(frames)
    return {"cut_at": cut, "ratio": ratio, "seamless": ratio <= SEAMLESS_MAX,
            "kept_fraction": round(kept, 3),
            "note": (f"best loop point is frame {cut}/{len(frames) - 1} "
                     f"(seam {ratio:.2f}x a typical step, keeps {kept:.0%} of the "
                     f"clip){'' if ratio <= SEAMLESS_MAX else ' — still visible'}.")}


def trim_to_loop(mp4_path: str | Path, frames: list[str], out_mp4: str | Path,
                 *, fps: int) -> dict:
    """Cut an mp4 at its best loop point with ffmpeg. Returns the cut report.

    ``fps`` must be the rate the frames were extracted at, since the frame index
    is converted back to a timestamp with it.
    """
    import subprocess

    cut = best_loop_cut(frames)
    if cut["ratio"] is None:
        return cut
    duration = (cut["cut_at"] + 1) / float(fps)
    subprocess.run(["ffmpeg", "-y", "-i", str(mp4_path), "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)],
                   check=True, capture_output=True)
    return {**cut, "out": str(out_mp4), "duration": round(duration, 3)}


#: Motion wording that describes the PHYSICS rather than the vibe. Video models
#: invent floaty, weightless bouncing when the prompt only names the action;
#: naming contact, compression and weight transfer is what produces a bounce
#: that reads as a real body on a real ball.
PHYSICAL_MOTION = (
    "The ball compresses under their weight and rebounds, driving the bounce. "
    "Their feet stay in contact with the ball, knees absorb the landing, arms "
    "counterbalance. Real weight and momentum, continuous single take, no cuts, "
    "no camera move."
)

#: Appended when the clip has to loop. The end-frame keyframe does the actual
#: work (see pollinations.video_loop); this stops the model from getting there
#: by freezing or fading, which technically matches the frame and looks dead.
LOOP_MOTION = (
    "One complete bounce cycle that ends exactly where it began, so the clip "
    "repeats seamlessly. Keep moving through the final frame — do not slow to a "
    "stop, freeze or fade."
)
