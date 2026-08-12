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

#: A framing floor appended to every prompt, still and motion alike. This is not
#: art direction — it is what makes the identity check possible at all. Measured
#: live: an unconstrained "peak of the jump" prompt put the subject across the
#: room at a 64-86px face, where ArcFace distances stop discriminating and the
#: honest verdict collapses to "cannot verify".
#:
#: The wording is deliberately photographic, because that is what the model
#: obeys. A/B'd on kontext against the same reference face (face size / drift):
#:   "head fills one sixth of the frame height"  ->  81px / 0.175
#:   "framed knees-up, ball at the bottom edge"  ->  85px / 0.168
#:   "framed head-to-ball, body fills the frame" ->  98px / 0.130   <- this one
#:   "waist-up, head one fifth of the height"    -> 151px / 0.053 (loses the ball)
#:   "close-up of face and shoulders"            -> 320px / 0.100 (loses the ball)
#: Abstract fractions of the frame do nothing; naming the crop works. Closer
#: framing also measurably improves identity fidelity, but past waist-up the
#: product leaves the shot — so this is the tightest crop that still shows the
#: person ON the ball, which is the thing the reel has to sell.
FRAMING = (
    "Framing: tight vertical shot, from the top of the head down to the ball, "
    "the body filling the frame. Face large, sharp, unobstructed and facing the "
    "camera. No wide room shots, no distant framing."
)

#: Framing wording for the VIDEO prompt, from most specific to least, stepped
#: down one notch per failed attempt. Seedance runs its own moderation and
#: answers 422 `content_policy_violation` (E005 "input or output flagged as
#: sensitive") on wording it dislikes — hit live by FRAMING's "the body filling
#: the frame / face large" on a prompt whose plainer form had just succeeded.
#: The trigger is phrasing, not the person, so retrying the identical prompt
#: only burns attempts; each retry drops to a plainer description instead. The
#: last entry is empty: the bare motion prompt, already proven to pass.
VIDEO_FRAMING_STEPS: tuple[str, ...] = (
    FRAMING,
    "Framing: keep the person close to the camera, head clearly visible and in "
    "focus, the ball at the bottom of the shot.",
    "Framing: keep the person close to the camera.",
    "",
)


@dataclass
class Attempt:
    n: int
    strategy_id: str
    start_frame: str
    video_frames: list[str]
    #: Median drift over judgeable frames — the number the verdict rests on.
    #: (Named for the field's role in the report, not for the worst frame; the
    #: worst frame is kept in `identity` alongside coverage and p90.)
    worst_identity_drift: float
    motion: float
    passed: bool
    reason: str = ""
    clip_path: str = ""
    identity: dict = field(default_factory=dict)


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
    from .identity_arcface import (HARD_DRIFT_MAX, MIN_COVERAGE,
                                   SAME_PERSON_MAX, START_MIN_FACE_PX)

    strat = strategy or STRATEGIES[0]
    bar = SAME_PERSON_MAX if max_identity_drift is None else max_identity_drift
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    still_prompt = (
        f"{strat.lead} {brief.subject} Keep this exact person's face and identity."
        f" {FRAMING}"
    )
    base_motion = (
        f"{strat.lead} The person jumps on the fitness ball: it compresses and "
        f"rebounds, they leave and land back on it. Keep the same person, same face."
    )

    tries: list[Attempt] = []
    best: Attempt | None = None
    for n in range(attempts):
        seed = brief.seed_for("produce", strat.id, str(n))
        # Step the video framing down a notch per attempt (see
        # VIDEO_FRAMING_STEPS): repeating a prompt the provider's moderation
        # just refused would spend every remaining attempt on the same refusal.
        step = VIDEO_FRAMING_STEPS[min(n, len(VIDEO_FRAMING_STEPS) - 1)]
        motion_prompt = f"{base_motion} {step}".strip()

        # 1. Flux Kontext start frame from the LOCAL face (multipart edits path;
        #    verified live, no media host needed for the still). seed is folded
        #    into the prompt because /v1/images/edits takes no seed param.
        try:
            start = pollinations.images_edit(
                f"{still_prompt} (variation {seed})", face_photo,
                out_dir / f"start_{n:02d}.png",
                model=start_model, width=brief.width, height=brief.height,
            )
        except Exception as e:  # noqa: BLE001 — a refused/failed still costs one
            tries.append(Attempt(   # attempt, not the run; the loop is the point
                n, strat.id, "", [], 1.0, 0.0, False,
                f"start frame generation failed: {_short(e)}"))
            continue

        # 2. reject the start frame early if it isn't this person. This is the
        #    cheap gate: a bad still costs one image call to redo, whereas
        #    letting it through costs a video call (~18x an image on seedance).
        #    One still is not a moving clip, so here the single frame IS the
        #    median and the worst — no quantile to take.
        start_check = arcface_drift([start], face_photo,
                                    min_face_px=START_MIN_FACE_PX)
        start_drift = start_check["median"]
        if start_drift is None or start_drift > bar:
            why = (f"start frame not the same person ({start_drift:.2f} > "
                   f"{bar:.2f})" if start_drift is not None
                   else f"start frame unusable: {start_check['note']}")
            tries.append(Attempt(
                n, strat.id, start, [], 1.0 if start_drift is None else
                round(start_drift, 4), 0.0, False,
                f"{why} — retried before spending a video call",
                identity=_identity_summary(start_check)))
            continue

        # 3. start frame -> jump video (Seedance image-to-video). The video call
        #    is the expensive one AND the one that can be refused upstream, so a
        #    failure here is recorded and retried rather than raised: a pipeline
        #    that dies on one 422 is not the "reliable" this module claims.
        try:
            start_url = pollinations.upload(start)
            mp4 = pollinations.video(
                motion_prompt, out_dir / f"video_{n:02d}.mp4",
                model=video_model, image_url=start_url,
                duration=brief.duration, aspect_ratio=brief.aspect_ratio,
            )
            frames = pollinations.extract_frames(mp4, out_dir / f"frames_{n:02d}")
        except Exception as e:  # noqa: BLE001 — see above
            tries.append(Attempt(
                n, strat.id, start, [], 1.0, 0.0, False,
                f"video generation failed: {_short(e)}",
                identity=_identity_summary(start_check)))
            continue

        # 4. gate: identity across the clip + motion present.
        #    Identity is judged on the MEDIAN of the frames whose face is big
        #    enough to embed, not the worst frame — measured live, a clip of the
        #    real person still spikes on the one blurred frame at the top of the
        #    jump, so a worst-frame bar rejects honest footage. `p90` still
        #    catches a clip that turns into someone else partway through, and
        #    `coverage` stops "too small to verify" from passing by default.
        drift = arcface_drift(frames, face_photo)
        motion = motion_presence(frames)["motion"]
        median = drift["median"]
        p90 = drift["p90"]
        coverage = drift["coverage"]
        if median is None or coverage < MIN_COVERAGE:
            passed, score = False, 1.0
            reason = (f"identity not verifiable: only {coverage:.0%} of frames "
                      f"had a face big enough to identify — {drift['note']}")
        else:
            score = median
            passed = (median <= bar and p90 <= HARD_DRIFT_MAX
                      and motion >= min_motion)
            if passed:
                reason = ""
            elif median > bar:
                reason = f"identity drift (median) {median:.2f} > {bar:.2f}"
            elif p90 > HARD_DRIFT_MAX:
                reason = (f"identity unstable: p90 {p90:.2f} > "
                          f"{HARD_DRIFT_MAX:.2f} (drifts inside the clip)")
            else:
                reason = f"motion {motion:.3f} < {min_motion:.2f}"
        att = Attempt(n, strat.id, start, frames, round(score, 4),
                      round(motion, 4), passed, reason, clip_path=mp4,
                      identity=_identity_summary(drift))
        tries.append(att)
        if best is None or att.worst_identity_drift < best.worst_identity_drift:
            best = att
        if passed:
            res = Result(face_photo, True, mp4, frames, start, tries,
                         f"passed on attempt {n}")
            _write_report(out_dir, res)
            return res

    note = (f"no attempt held identity across a moving clip in {attempts} tries. "
            f"Best median drift {best.worst_identity_drift if best else 'n/a'} "
            f"(bar {bar}). Returning the best attempt, flagged NOT passing — "
            f"do not ship blind.")
    res = Result(face_photo, False,
                 best.clip_path if best else "",
                 best.video_frames if best else [],
                 best.start_frame if best else "", tries, note)
    _write_report(out_dir, res)
    return res


def _short(e: Exception, limit: int = 240) -> str:
    """One readable line from a provider error, so the report says WHY it failed.

    Content-policy refusals are the ones worth recognising by name: they are a
    property of the prompt, not a transient fault, so the next attempt should
    reword rather than simply re-roll.
    """
    text = " ".join(str(e).split())
    if "content_policy_violation" in text or "content moderation" in text:
        text = f"refused by provider content moderation — {text}"
    return text[:limit]


def _identity_summary(drift: dict) -> dict:
    """The identity evidence worth keeping in the report, without every frame.

    Carries what makes the verdict auditable: the spread (median/p90/worst), how
    much of the clip was actually judgeable, and the face sizes that decided
    that — so a reader can tell "different person" from "face too small to tell".
    """
    worst_name, worst_val = drift.get("worst", (None, None))
    px = drift.get("face_px") or {}
    return {
        "median": drift.get("median"), "p90": drift.get("p90"),
        "worst": worst_val, "worst_frame": worst_name,
        "coverage": drift.get("coverage"), "judgeable": drift.get("judgeable"),
        "frames": drift.get("readable"),
        "too_small": len(drift.get("too_small") or []),
        "no_face": len(drift.get("no_face") or []),
        "face_px_min": min(px.values()) if px else None,
        "face_px_max": max(px.values()) if px else None,
        "note": drift.get("note", ""),
    }


def _write_report(out_dir: Path, res: Result) -> None:
    (out_dir / "produce_report.json").write_text(json.dumps(res.to_dict(), indent=2))


def render(res: Result) -> str:
    head = (f"produce — face: {Path(res.face_photo).name}   "
            f"{'PASSED' if res.passed else 'NOT PASSED'}")
    lines = [head, res.note, "",
             f"{'try':<4}{'drift(med)':<12}{'p90':<8}{'cover':<8}{'face px':<10}"
             f"{'motion':<8}{'ok':<5}reason", "-" * 96]
    for a in res.attempts:
        i = a.identity or {}
        p90 = f"{i['p90']:.3f}" if i.get("p90") is not None else "-"
        cov = f"{i['coverage']:.0%}" if i.get("coverage") is not None else "-"
        lo, hi = i.get("face_px_min"), i.get("face_px_max")
        px = f"{lo}-{hi}" if lo is not None else "-"
        lines.append(f"{a.n:<4}{a.worst_identity_drift:<12.3f}{p90:<8}{cov:<8}"
                     f"{px:<10}{a.motion:<8.3f}"
                     f"{'yes' if a.passed else 'no':<5}{a.reason}")
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
