"""A driving video -> a validated motion spec the generator can be asked for.

The pipeline's weak point is that motion is requested in prose. "Bounces on the
ball" does not say how fast, how far, or with what expression, so the generator
picks — and picks differently every run. A driving video already contains those
answers; this module turns it into numbers.

What comes out is a SPEC, not footage: per-frame pose and expression, plus the
derived quantities that actually describe the movement — amplitude, cadence,
velocity, expression peaks. Those are transferable to a person with a different
body, which raw pixels are not.

Everything here is CPU and local (MediaPipe + numpy + ffmpeg). That is the point
of splitting it out: extraction, normalisation and validation do not depend on
which generator runs afterwards, so this half can be built and trusted before
any decision about a GPU stack, and survives changing that decision.

HONEST LIMITS, stated up front because a spec that hides them is worse than none:

  * Camera motion is NOT separated from subject motion. Vertical amplitude is
    measured from hip position, so a camera that tilts or pushes in during the
    take is charged to the subject. Use locked-off footage, or treat amplitude
    and cadence as suspect. `validate()` says so rather than letting it pass.
  * Proportions are the SOURCE person's. Normalising by torso length makes the
    numbers comparable across bodies, but this is not full motion retargeting —
    joint angles are not solved onto a different skeleton.
  * Expression needs a large, sharp face. On full-body framing the face is often
    too small to read blendshapes at all, and the spec reports that as missing
    rather than as neutral.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Sampling rate for the spec. 12 fps resolves a bounce of a few cycles per
#: second (Nyquist) without paying for frames that carry no new information.
SPEC_FPS = 12

#: A face smaller than this (px, shorter bbox side) does not produce trustworthy
#: blendshapes — same reasoning and same scale as the identity floor.
MIN_EXPRESSION_FACE_PX = 100

#: Blendshapes worth naming in a summary. The full 52 stay in the per-frame
#: track; these are the ones a person would actually direct.
KEY_SHAPES = ("mouthSmileLeft", "mouthSmileRight", "jawOpen", "eyeBlinkLeft",
              "eyeBlinkRight", "browInnerUp", "browOuterUpLeft",
              "eyeSquintLeft", "eyeSquintRight", "cheekPuff")


@dataclass
class MotionSummary:
    """The movement, in numbers that transfer to a different body."""

    #: Peak-to-peak vertical travel of the hips, in torso lengths.
    amplitude: float | None = None
    #: Repetitions per second (bounces/s), from zero crossings of hip height.
    cadence_hz: float | None = None
    #: Mean joint speed, torso lengths per second.
    velocity: float | None = None
    #: Intensity peak: the 95th-percentile frame-to-frame step, same units.
    #: A percentile and not the maximum, because a single flickering landmark
    #: produces one enormous step — measured live, a max of 14.9 against a mean
    #: of 0.93, which describes the tracker rather than the movement.
    peak_velocity: float | None = None
    #: How far the pose ranges over the take, in torso lengths.
    pose_range: float | None = None


@dataclass
class ExpressionSummary:
    """The face, as directable coefficients rather than adjectives."""

    #: Peak value each key blendshape reaches over the take.
    peaks: dict = field(default_factory=dict)
    #: Mean over frames where the face was large enough to read.
    means: dict = field(default_factory=dict)
    #: Frames whose face was measurable, out of those sampled.
    coverage: float = 0.0


@dataclass
class DrivingSpec:
    """Everything extracted from one driving video."""

    source: str
    fps: int
    frames: int
    duration: float
    #: Offset of this segment within the source video, seconds.
    start: float = 0.0
    motion: MotionSummary = field(default_factory=MotionSummary)
    expression: ExpressionSummary = field(default_factory=ExpressionSummary)
    #: Fraction of sampled frames in which a body was found.
    pose_coverage: float = 0.0
    #: Per-frame track, kept so a consumer can condition frame by frame rather
    #: than only on the summary.
    track: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self, *, with_track: bool = True) -> dict:
        d = asdict(self)
        if not with_track:
            d.pop("track")
        return d

    def save(self, path: str | Path, *, with_track: bool = True) -> str:
        Path(path).write_text(json.dumps(self.to_dict(with_track=with_track),
                                         indent=2))
        return str(path)

    def validate(self) -> list[str]:
        """Problems that should stop this spec being used, worst first.

        Returned rather than raised: a caller may knowingly proceed on a partial
        spec (motion only, no expression), but must not do so by accident.
        """
        problems = []
        if self.frames < 6:
            problems.append(f"only {self.frames} frames sampled: too short to "
                            f"describe a movement.")
        if self.pose_coverage < 0.8:
            problems.append(f"body found in only {self.pose_coverage:.0%} of "
                            f"frames: the motion track has holes.")
        if self.motion.cadence_hz is None or self.motion.amplitude is None:
            problems.append("no periodic motion measured: amplitude/cadence "
                            "are unusable for conditioning.")
        if self.expression.coverage < 0.5:
            problems.append(f"face readable in only {self.expression.coverage:.0%} "
                            f"of frames: expression is NOT specified by this "
                            f"video — do not claim it is.")
        problems.extend(self.warnings)
        return problems


def _sample_frames(video_path: str | Path, out_dir: str | Path,
                   fps: int = SPEC_FPS, start: float = 0.0,
                   length: float | None = None) -> list[str]:
    import subprocess

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video_path)]
    if length:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-vf", f"fps={fps}", str(out_dir / "%04d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(str(p) for p in out_dir.glob("*.png"))


def _blendshapes(path: str | Path, model_path: str | Path) -> dict | None:
    """52 ARKit-style expression coefficients, or None if no face is read."""
    import mediapipe as mp  # type: ignore
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision  # type: ignore
    from PIL import Image

    global _FACE
    try:
        _FACE
    except NameError:
        _FACE = None
    if _FACE is None:
        _FACE = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True))
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    r = _FACE.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not r.face_blendshapes:
        return None
    return {b.category_name: round(float(b.score), 4)
            for b in r.face_blendshapes[0]}


def _hip_raw(points: dict) -> tuple[float, float] | None:
    """Raw hip height and raw torso length, both in image units.

    Deliberately NOT divided here. Dividing per frame by that frame's own torso
    length looks scale-invariant and is a trap: projected torso length collapses
    toward zero whenever the person bends or lies down, so the ratio explodes.
    Measured on a real stretching video, that produced an "amplitude" of 5.1
    torso-lengths — a person does not travel five torso-lengths vertically. The
    caller divides by a per-segment constant instead; see `_segment_scale`.
    """
    import numpy as np

    need = ("l_hip", "r_hip", "l_shoulder", "r_shoulder")
    if any(points.get(n) is None or points[n][2] < 0.5 for n in need):
        return None
    hip_y = (points["l_hip"][1] + points["r_hip"][1]) / 2
    sho_y = (points["l_shoulder"][1] + points["r_shoulder"][1]) / 2
    hip = np.array([(points["l_hip"][0] + points["r_hip"][0]) / 2, hip_y])
    sho = np.array([(points["l_shoulder"][0] + points["r_shoulder"][0]) / 2, sho_y])
    return hip_y, float(np.linalg.norm(sho - hip))


def _segment_scale(torsos: list[float]) -> float | None:
    """One body-size unit for the whole segment: the median torso length.

    A constant rather than a per-frame value, because the quantity being removed
    is CAMERA DISTANCE, which is fixed for a locked-off take — the same
    assumption the module already declares. The median ignores the frames where
    the torso foreshortens, instead of letting them dominate.
    """
    import numpy as np

    usable = [t for t in torsos if t > 1e-6]
    if len(usable) < 3:
        return None
    scale = float(np.median(usable))
    return scale if scale > 1e-6 else None


def _cadence(series: list[float], fps: int) -> float | None:
    """Repetitions per second, by counting crossings of the mean.

    Crossings rather than an FFT because a few seconds of footage hold only a
    few cycles — too few bins for a meaningful spectrum, and a peak picked from
    three bins would be a number pretending to be a measurement.

    BIASED LOW ON SHORT WINDOWS, by design. A window containing N cycles shows
    2N-1 crossings, not 2N, because the last half-cycle has no crossing after
    it: two cycles in two seconds measure 0.75 Hz rather than 1.0. Adding the
    missing half would fix that and would also invent a cycle in data that
    merely trends in one direction, which crosses the mean exactly once. An
    under-report on a short clip is recoverable; a fabricated rhythm is not.
    The bias shrinks with window length (six cycles in six seconds: 0.92 Hz),
    so measure over a longer segment when the number matters.
    """
    import numpy as np

    if len(series) < 4:
        return None
    arr = np.asarray(series, dtype="float64")
    centred = arr - arr.mean()
    if float(np.std(centred)) < 1e-3:
        return 0.0
    signs = np.sign(centred)
    signs[signs == 0] = 1
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    seconds = len(series) / float(fps)
    return round(crossings / 2.0 / seconds, 3) if seconds > 0 else None


def extract(video_path: str | Path, *, fps: int = SPEC_FPS,
            start: float = 0.0, length: float | None = None,
            work_dir: str | Path | None = None,
            face_model: str | Path | None = None,
            keep_track: bool = True) -> DrivingSpec:
    """Measure a driving video, or one segment of it, into a DrivingSpec.

    `start`/`length` (seconds) exist because a real reference video is usually a
    ROUTINE, not one movement. Summarising several different exercises into one
    amplitude and one cadence produces numbers that describe none of them — so
    pick the segment holding the movement you actually want, with `survey()`.

    `face_model` points at MediaPipe's face_landmarker.task; without it the
    expression track is simply absent (and `validate()` reports that) rather
    than silently reported as neutral.
    """
    import os
    import tempfile

    import numpy as np

    from .pose import _normalise, landmarks

    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory()
        work_dir = tmp.name
    try:
        frames = _sample_frames(video_path, work_dir, fps=fps,
                                start=start, length=length)
        face_model = face_model or os.environ.get(
            "BALL_REEL_FACE_MODEL", "~/.mediapipe/face_landmarker.task")
        face_model = Path(str(face_model)).expanduser()
        have_face_model = face_model.exists()

        track, poses = [], []
        raw_hips: list[tuple[int, float, float]] = []   # (row index, hip_y, torso)
        shape_rows: list[dict] = []
        for i, f in enumerate(frames):
            pts = landmarks(f)
            row: dict = {"i": i, "t": round(i / float(fps), 3)}
            if pts is not None:
                raw = _hip_raw(pts)
                norm = _normalise(pts)
                if raw is not None:
                    raw_hips.append((len(track), raw[0], raw[1]))
                if norm is not None:
                    poses.append({k: v[0] for k, v in norm.items()})
                    row["pose"] = {k: [round(float(v[0][0]), 4),
                                       round(float(v[0][1]), 4)]
                                   for k, v in norm.items()}
            if have_face_model:
                bs = _blendshapes(f, face_model)
                if bs:
                    shape_rows.append(bs)
                    row["blendshapes"] = {k: bs[k] for k in KEY_SHAPES if k in bs}
            track.append(row)

        spec = DrivingSpec(source=str(video_path), fps=fps, frames=len(frames),
                           duration=round(len(frames) / float(fps), 3),
                           start=start, track=track if keep_track else [])
        spec.pose_coverage = round(len(poses) / len(frames), 3) if frames else 0.0

        # Height in body units, using ONE scale for the segment (see
        # _segment_scale) so a frame in which the torso foreshortens cannot
        # inflate the amplitude.
        scale = _segment_scale([t for _, _, t in raw_hips])
        heights: list[float] = []
        if scale:
            for idx, hip_y, _ in raw_hips:
                h = hip_y / scale
                heights.append(h)
                track[idx]["hip_height"] = round(h, 4)
        elif raw_hips:
            spec.warnings.append(
                "body size could not be established: amplitude and cadence "
                "not measured.")

        if heights:
            spec.motion.amplitude = round(float(max(heights) - min(heights)), 4)
            spec.motion.cadence_hz = _cadence(heights, fps)
        if len(poses) >= 2:
            steps = []
            for a, b in zip(poses, poses[1:]):
                shared = [k for k in a if k in b]
                steps.append(float(np.mean([np.linalg.norm(b[k] - a[k])
                                            for k in shared])))
            spec.motion.velocity = round(float(np.mean(steps)) * fps, 4)
            spec.motion.peak_velocity = round(
                float(np.percentile(steps, 95)) * fps, 4)
            first = poses[0]
            spread = [float(np.mean([np.linalg.norm(p[k] - first[k])
                                     for k in p if k in first])) for p in poses]
            spec.motion.pose_range = round(float(max(spread)), 4)

        if shape_rows:
            keys = [k for k in KEY_SHAPES if k in shape_rows[0]]
            spec.expression.peaks = {k: round(max(r.get(k, 0.0) for r in shape_rows), 3)
                                     for k in keys}
            spec.expression.means = {k: round(float(np.mean([r.get(k, 0.0)
                                                             for r in shape_rows])), 3)
                                     for k in keys}
        spec.expression.coverage = (round(len(shape_rows) / len(frames), 3)
                                    if frames else 0.0)
        if not have_face_model:
            spec.warnings.append(
                f"no face model at {face_model}: expression NOT extracted. "
                f"Download face_landmarker.task or set BALL_REEL_FACE_MODEL.")
        spec.warnings.append(
            "camera motion is not separated from subject motion: amplitude and "
            "cadence assume a locked-off camera.")
        return spec
    finally:
        if tmp is not None:
            tmp.cleanup()


def to_prompt(spec: DrivingSpec) -> str:
    """The spec as directable words, for generators that take only text.

    A lossy but honest fallback: a text generator cannot be handed a per-frame
    track, so the summary is rendered as pace and expression. Anything the spec
    could not measure is left out rather than described as neutral — the whole
    reason for measuring was to stop inventing these.
    """
    bits = []
    m = spec.motion
    if m.cadence_hz:
        bits.append(f"about {m.cadence_hz:.1f} bounces per second")
    if m.amplitude:
        bits.append(f"rising and falling roughly {m.amplitude:.2f} torso-lengths")
    if m.peak_velocity and m.velocity and m.velocity > 0:
        ratio = m.peak_velocity / m.velocity
        bits.append("with sharp accelerations" if ratio > 2.5 else "at an even pace")
    e = spec.expression
    if e.coverage >= 0.5 and e.peaks:
        smile = max(e.peaks.get("mouthSmileLeft", 0),
                    e.peaks.get("mouthSmileRight", 0))
        if smile > 0.4:
            bits.append("smiling broadly")
        elif smile > 0.15:
            bits.append("with a slight smile")
        if e.peaks.get("browInnerUp", 0) > 0.4:
            bits.append("eyebrows raised")
    return ("Motion: " + ", ".join(bits) + ".") if bits else ""
