"""produce: a face photo -> a video of THAT person jumping on a ball, reliably.

This is the generation pipeline, not the eval. "Reliably" is not "never fails" —
no generator gives that. It means: the pipeline will not hand you a clip where
the face drifted into someone else or where nothing moved. It generates,
CHECKS identity and motion, and RETRIES; if no attempt passes, it returns the
best one clearly flagged as not-passing rather than pretending.

Everything runs through ONE gateway — Pollinations (gen.pollinations.ai) — the
same stack the vacancy names. No separate video host, no local GPU. The exact
endpoints are pinned in POLLINATIONS_CONTRACT.md ([verified live] vs [confirm]):

  1. start frame -- Flux Kontext from the LOCAL face via /v1/images/edits
     (multipart). The real face as a signal, not a prompt describing one; the
     bytes go straight to the allowed host, no media upload for the still.
     pollinations.images_edit(prompt, face_photo). [verified live]
  2. reject early-- ArcFace drift on the start frame vs the photo. If it already
     isn't this person, retry the still before spending a video call.
  3. upload      -- the accepted start frame -> a media URL, because the video
     endpoint references its input by URL. pollinations.upload(). [confirm live]
  4. video       -- start frame URL -> jump motion, image-to-video on Seedance.
     pollinations.video(model="seedance-2.0", image_url=start_url). [confirm live]
  5. gate        -- ArcFace identity across the extracted frames + motion
     presence. Pass -> done. Fail -> next attempt, new seed.

The identity check (insightface/ArcFace) is real and local — the one thing that
must not be outsourced to the thing being judged. Everything generative is a
Pollinations HTTP call. Nothing here runs in THIS repo (no key/network); it is
the real pipeline against the documented endpoints, to run on your bench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .brief import DEMO_BRIEF, Brief
from .gen import STRATEGIES, Strategy

#: Stack, as named in the vacancy. Override per bench via env at call sites.
START_MODEL = "kontext"        # Flux Kontext — face-conditioned still
VIDEO_MODEL = "seedance-2.0"   # Seedance — image-to-video from the start frame


@dataclass
class Attempt:
    n: int
    strategy_id: str
    start_frame: str
    video_frames: list[str]
    worst_identity_drift: float
    motion: float
    passed: bool
    reason: str = ""
    clip_path: str = ""


@dataclass
class Result:
    face_photo: str
    passed: bool
    clip_path: str = ""
    clip_frames: list[str] = field(default_factory=list)
    start_frame: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "face_photo": self.face_photo, "passed": self.passed,
            "clip_path": self.clip_path, "start_frame": self.start_frame,
            "clip_frames": self.clip_frames, "note": self.note,
            "attempts": [a.__dict__ for a in self.attempts],
        }


def produce(
    face_photo: str,
    brief: Brief = DEMO_BRIEF,
    *,
    strategy: Strategy | None = None,
    attempts: int = 4,
    max_identity_drift: float | None = None,
    min_motion: float = 0.02,
    start_model: str = START_MODEL,
    video_model: str = VIDEO_MODEL,
    out_dir: str | Path = "produce_out",
) -> Result:
    """Generate a jump clip of the person in `face_photo`, retrying until it holds.

    Returns a Result whose `passed` says whether identity held across a moving
    clip. On success `clip_path` is the accepted mp4 and `clip_frames` its
    extracted frames; on failure they are the best (lowest-drift) attempt and
    `passed` is False — the caller must see that, not be handed a silent
    near-miss. The whole generative chain is Pollinations; identity is ArcFace.
    """
    from . import pollinations
    from .identity import arcface_drift, motion_presence
    from .identity_arcface import SAME_PERSON_MAX

    strat = strategy or STRATEGIES[0]
    bar = SAME_PERSON_MAX if max_identity_drift is None else max_identity_drift
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    still_prompt = (
        f"{strat.lead} {brief.subject} Keep this exact person's face and identity."
    )
    motion_prompt = (
        f"{strat.lead} The person jumps on the fitness ball: it compresses and "
        f"rebounds, they leave and land back on it. Keep the same person, same face."
    )

    tries: list[Attempt] = []
    best: Attempt | None = None
    for n in range(attempts):
        seed = brief.seed_for("produce", strat.id, str(n))

        # 1. Flux Kontext start frame from the LOCAL face (multipart edits path;
        #    verified live, no media host needed for the still). seed is folded
        #    into the prompt because /v1/images/edits takes no seed param.
        start = pollinations.images_edit(
            f"{still_prompt} (variation {seed})", face_photo,
            out_dir / f"start_{n:02d}.png",
            model=start_model, width=brief.width, height=brief.height,
        )

        # 2. reject the start frame early if it isn't this person
        start_drift = arcface_drift([start], face_photo)["worst"][1]
        if start_drift is not None and start_drift > bar:
            tries.append(Attempt(
                n, strat.id, start, [], round(start_drift, 4), 0.0, False,
                f"start frame not the same person "
                f"({start_drift:.2f} > {bar:.2f}) — retried before video"))
            continue

        # 3. start frame -> jump video (Seedance image-to-video)
        start_url = pollinations.upload(start)
        mp4 = pollinations.video(
            motion_prompt, out_dir / f"video_{n:02d}.mp4",
            model=video_model, image_url=start_url,
            duration=brief.duration, aspect_ratio=brief.aspect_ratio,
        )
        frames = pollinations.extract_frames(mp4, out_dir / f"frames_{n:02d}")

        # 4. gate: identity across the clip + motion present
        drift = arcface_drift(frames, face_photo)
        worst = drift["worst"][1] if drift["worst"][1] is not None else 1.0
        motion = motion_presence(frames)["motion"]
        passed = worst <= bar and motion >= min_motion
        reason = "" if passed else (
            f"identity drift {worst:.2f} > {bar:.2f}" if worst > bar
            else f"motion {motion:.3f} < {min_motion:.2f}")
        att = Attempt(n, strat.id, start, frames, round(worst, 4),
                      round(motion, 4), passed, reason, clip_path=mp4)
        tries.append(att)
        if best is None or att.worst_identity_drift < best.worst_identity_drift:
            best = att
        if passed:
            res = Result(face_photo, True, mp4, frames, start, tries,
                         f"passed on attempt {n}")
            _write_report(out_dir, res)
            return res

    note = (f"no attempt held identity across a moving clip in {attempts} tries. "
            f"Best drift {best.worst_identity_drift if best else 'n/a'}. "
            f"Returning the best attempt, flagged NOT passing — do not ship blind.")
    res = Result(face_photo, False,
                 best.clip_path if best else "",
                 best.video_frames if best else [],
                 best.start_frame if best else "", tries, note)
    _write_report(out_dir, res)
    return res


def _write_report(out_dir: Path, res: Result) -> None:
    (out_dir / "produce_report.json").write_text(json.dumps(res.to_dict(), indent=2))


def render(res: Result) -> str:
    head = (f"produce — face: {Path(res.face_photo).name}   "
            f"{'PASSED' if res.passed else 'NOT PASSED'}")
    lines = [head, res.note, "", f"{'try':<4}{'start/clip drift':<18}"
             f"{'motion':<8}{'passed':<8}reason", "-" * 70]
    for a in res.attempts:
        lines.append(f"{a.n:<4}{a.worst_identity_drift:<18.3f}{a.motion:<8.3f}"
                     f"{'yes' if a.passed else 'no':<8}{a.reason}")
    if res.passed:
        lines += ["", f"clip: {res.clip_path}  ({len(res.clip_frames)} frames)"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ball_reel.produce",
        description="face photo -> jump-on-ball clip, with a reliability loop (Pollinations)")
    ap.add_argument("--face", required=True, help="path to the person's face photo")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--start-model", default=START_MODEL)
    ap.add_argument("--video-model", default=VIDEO_MODEL)
    ap.add_argument("--out", default="produce_out")
    args = ap.parse_args(argv)
    res = produce(args.face, attempts=args.attempts,
                  start_model=args.start_model, video_model=args.video_model,
                  out_dir=args.out)
    print(render(res))
    return 0 if res.passed else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
