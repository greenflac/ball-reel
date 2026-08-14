"""Motion as a CONSTRAINT, not a request: a chain of pinned keyframes.

Asking a video model for "about 0.6 bounces per second" is a wish. The model
reads it as flavour and invents its own trajectory, which is exactly the
hallucination the pipeline exists to prevent — and no amount of prompt wording
fixes it, because prose is not a specification of geometry.

The gateway does, however, accept TWO keyframes per clip (`image=<start>|<end>`,
verified on wan-fast, veo, wan-pro, seedance-2.0). That is the opening. Instead
of one clip described in words, generate a chain:

    K0 ──clip──▶ K1 ──clip──▶ K2 ──clip──▶ ... ──clip──▶ K0

Each Kᵢ is the TARGET PERSON rendered in a pose taken from the driving video, so
at every node the motion is pinned to real geometry. The model only interpolates
between adjacent nodes, and how far it can wander is bounded by how close the
nodes are — which makes keyframe density the accuracy/cost dial, an engineering
choice rather than a hope.

Two properties fall out for free:

  * The chain LOOPS by construction when the last keyframe is the first one.
  * Every keyframe is checkable BEFORE any video call: render it, measure its
    pose against the driving frame it came from, and re-render if it missed.
    Stills are ~18x cheaper than video on seedance, so rejecting here is where
    the money is saved.

HONEST LIMIT: between two keyframes the trajectory is still the model's guess.
This constrains motion at the nodes, not continuously — that needs per-frame
conditioning (ControlNet/DWPose), which this gateway does not have. Denser
keyframes tighten the bound and cost proportionally more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Models whose `video_capabilities` include `end_frame`, cheapest first.
#: A model without it silently ignores the second keyframe and the whole scheme
#: degrades to unconstrained generation — so the choice is not cosmetic.
#: Модели с end_frame и их цена, pollen за секунду. ОДНА таблица на проект:
#: раньше список моделей жил в трёх местах (здесь, в прайс-таблице прогона и в
#: докстринге `pollinations.video_loop`), и они разошлись — докстринг называл
#: `wan`, у которого end_frame нет, и умалчивал про `wan-fast`, на котором всё
#: и считается. Оператор, поверивший докстрингу, получил бы ValueError после
#: предполёта, дыма и всех кейфреймов.
#:
#: Снято с GET /video/models 2026-08-12 [проверено live]. end_frame есть ровно
#: у четырёх, и у всех четырёх max_reference_images=2 — то есть цепочка
#: start|end физически возможна только на них.
END_FRAME_POLLEN = {"wan-fast": 0.01, "veo": 0.08, "wan-pro": 0.1,
                    "seedance-2.0": 0.18}
END_FRAME_MODELS = tuple(END_FRAME_POLLEN)

#: Допустимая длительность сегмента, секунд. Шлюз валидирует её ПО МОДЕЛИ и
#: отвечает 400 при выходе за диапазон — то есть провал приходит на последнем,
#: платном шаге, когда все кейфреймы уже отрисованы. Дефолт прогона `2` живёт
#: только на wan-семействе; на seedance и veo он бы упал.
#:
#: `veo` принимает НЕ диапазон, а три конкретных значения — поэтому таблица
#: хранит множество, а не пару границ.
SEGMENT_SECONDS = {
    "wan-fast": frozenset(range(2, 16)),      # 2 c проверено live
    "wan-pro": frozenset(range(2, 16)),       # семейство wan по докам
    "seedance-2.0": frozenset(range(4, 16)),  # 2 c даёт 400 [проверено live]
    "veo": frozenset((4, 6, 8)),              # только эти три
}


def segment_seconds_ok(model: str, seconds: int) -> tuple:
    """Пройдёт ли такая длительность на этой модели. (ok, что делать)."""
    allowed = SEGMENT_SECONDS.get(model)
    if allowed is None:
        return True, f"диапазон {model} не проверялся — сверить с /video/models"
    if seconds in allowed:
        return True, ""
    ok_values = sorted(allowed)
    shown = (f"{ok_values[0]}..{ok_values[-1]}" if len(ok_values) > 3
             else "/".join(str(v) for v in ok_values))
    return False, (f"{model} принимает {shown} c, а не {seconds}: шлюз вернёт "
                   f"400 на последнем — платном — шаге, когда кейфреймы уже "
                   f"отрисованы. Ближайшее допустимое: "
                   f"{min(ok_values, key=lambda v: abs(v - seconds))}.")

#: Keyframes per chain by default. Enough to pin a bounce at its extremes
#: (top, bottom, top) plus the return to the start.
DEFAULT_KEYFRAMES = 5

#: A rendered keyframe must land at least this close to the driving pose it was
#: asked to reproduce, or the node is not pinning anything. Same torso-length
#: scale as pose.SAME_POSE_MAX, and the same reasoning: nothing has moved yet.
KEYFRAME_POSE_MAX = 0.25


@dataclass
class Keyframe:
    index: int
    #: Time in the driving video this pose came from.
    t: float
    driving_frame: str
    rendered: str = ""
    url: str = ""
    pose_distance: float | None = None
    identity_drift: float | None = None
    accepted: bool = False
    reason: str = ""


@dataclass
class ChainResult:
    keyframes: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    clip_path: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {"clip_path": self.clip_path, "note": self.note,
                "keyframes": [k.__dict__ for k in self.keyframes],
                "segments": self.segments}


def pick_keyframe_times(track: list, count: int = DEFAULT_KEYFRAMES) -> list[int]:
    """Choose which frames of the driving track become keyframes.

    Not evenly spaced: a movement is defined by its EXTREMES — the top and
    bottom of a bounce — and uniform sampling can land entirely on the way up
    and miss both. So the turning points of hip height are taken first, and even
    spacing only fills in if there are not enough of them.

    Returns indices into `track`, always including the first frame so the chain
    can close back onto it.
    """
    import numpy as np

    idx = [i for i, r in enumerate(track) if r.get("hip_height") is not None]
    if len(idx) < 3:
        return idx[:count]
    heights = np.array([track[i]["hip_height"] for i in idx])
    # Local extrema: a frame higher (or lower) than both neighbours.
    turns = [idx[j] for j in range(1, len(idx) - 1)
             if (heights[j] - heights[j - 1]) * (heights[j + 1] - heights[j]) < 0]
    chosen = [idx[0]] + turns
    if len(chosen) < count:
        step = max(1, len(idx) // count)
        chosen += [idx[j] for j in range(0, len(idx), step)]
    chosen = sorted(set(chosen))
    if len(chosen) > count:
        # Keep the ends and thin the middle evenly, so the span is preserved.
        keep = np.linspace(0, len(chosen) - 1, count).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(keep.tolist()))]
    return chosen


def render_keyframes(driving_frames: list[str], indices: list[int],
                     face_photo: str, *, out_dir: str | Path,
                     body_ref: str = "", subject_text: str = "",
                     scene: str = "", model: str = "nanobanana",
                     attempts: int = 2) -> list[Keyframe]:
    """Render the target person in each chosen driving pose, and check each one.

    The pose arrives as a REFERENCE IMAGE (the driving frame itself), not as
    words, for the same reason build does: geometry does not survive being
    described. The prompt names the role of each reference by position, and pins
    the face to the first — so a driving frame showing a different person
    contributes their POSE and nothing else.

    Every keyframe is verified before it can be used, and re-rendered if it
    missed. This is the cheap gate: a still costs a fraction of a video call.
    """
    from . import pollinations
    from .identity import arcface_drift
    from .identity_arcface import START_MIN_FACE_PX
    from .pose import pose_delta, landmarks

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    face_url = pollinations.upload(face_photo)
    body_url = pollinations.upload(body_ref) if body_ref else ""

    roles = ["the FIRST image for the person's face and identity"]
    if body_url:
        roles.append("the SECOND image for their body build and clothing")
    roles.append(f"the {'THIRD' if body_url else 'SECOND'} image for the POSE "
                 f"and camera angle only")
    clause = ("Use " + ", and ".join(roles)
              + ". The face must come only from the first image.")

    out: list[Keyframe] = []
    for n, i in enumerate(indices):
        src = driving_frames[i]
        kf = Keyframe(index=n, t=float(i), driving_frame=src)
        target = landmarks(src)
        pose_url = pollinations.upload(src)
        urls = [face_url] + ([body_url] if body_url else []) + [pose_url]
        prompt = " ".join(p for p in (
            scene, subject_text, clause,
            "Reproduce the pose exactly as shown in the pose reference.") if p)

        for attempt in range(attempts):
            path = out_dir / f"kf_{n:02d}{'' if attempt == 0 else f'_r{attempt}'}.png"
            try:
                rendered = pollinations.compose(prompt, urls, path, model=model,
                                                seed=1000 + n * 10 + attempt)
            except Exception as e:  # noqa: BLE001 — a refused still costs a retry
                kf.reason = f"render failed: {str(e)[:160]}"
                continue
            kf.rendered = rendered
            got = landmarks(rendered)
            if target is None or got is None:
                kf.reason = "no body detected in keyframe or driving frame"
                continue
            delta = pose_delta(target, got)
            kf.pose_distance = None if delta is None else delta["mean"]
            drift = arcface_drift([rendered], face_photo,
                                  min_face_px=START_MIN_FACE_PX)
            kf.identity_drift = drift["median"]
            if kf.pose_distance is None:
                kf.reason = "pose not measurable in keyframe"
                continue
            if kf.pose_distance > KEYFRAME_POSE_MAX:
                kf.reason = (f"pose off by {kf.pose_distance} "
                             f"(> {KEYFRAME_POSE_MAX}) — node would not pin "
                             f"the motion")
                continue
            kf.accepted, kf.reason = True, ""
            break
        if kf.accepted and kf.rendered:
            kf.url = pollinations.upload(kf.rendered)
        out.append(kf)
    return out


def generate_chain(keyframes: list, prompt: str, out_dir: str | Path, *,
                   model: str = "wan-fast", seconds_per_segment: int = 2,
                   loop: bool = True, aspect_ratio: str = "9:16") -> ChainResult:
    """Generate one clip per adjacent keyframe pair and join them.

    `loop` appends a final segment back to the first keyframe, which is what
    makes the chain close — a stronger guarantee than asking for a loop in
    words, because the endpoint is the same image the clip started from.
    """
    from . import pollinations

    if model not in END_FRAME_MODELS:
        raise ValueError(
            f"{model} does not support end_frame; the chain needs it to pin "
            f"each segment. Use one of {END_FRAME_MODELS}.")
    usable = [k for k in keyframes if k.accepted and k.url]
    if len(usable) < 2:
        return ChainResult(keyframes=keyframes,
                           note=f"only {len(usable)} usable keyframe(s): "
                                f"nothing to chain. Fix the keyframes first.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(usable, usable[1:]))
    if loop:
        pairs.append((usable[-1], usable[0]))

    segments = []
    for i, (a, b) in enumerate(pairs):
        seg = out_dir / f"seg_{i:02d}.mp4"
        try:
            pollinations.video(prompt, seg, model=model,
                               image_url=[a.url, b.url],
                               duration=seconds_per_segment,
                               aspect_ratio=aspect_ratio)
            segments.append({"i": i, "from": a.index, "to": b.index,
                             "path": str(seg), "ok": True,
                             "usage": dict(pollinations.LAST_VIDEO_USAGE)})
        except Exception as e:  # noqa: BLE001
            segments.append({"i": i, "from": a.index, "to": b.index,
                             "ok": False, "error": str(e)[:200]})

    good = [s["path"] for s in segments if s.get("ok")]
    if not good:
        return ChainResult(keyframes=keyframes, segments=segments,
                           note="every segment failed; no clip produced.")
    joined = _concat(good, out_dir / "chain.mp4")
    missing = len(pairs) - len(good)
    note = (f"chained {len(good)}/{len(pairs)} segment(s) across "
            f"{len(usable)} keyframe(s)"
            + (f"; {missing} segment(s) failed — the motion has gaps there"
               if missing else "") + ".")
    return ChainResult(keyframes=keyframes, segments=segments,
                       clip_path=joined, note=note)


def _concat(paths: list[str], out_mp4: str | Path) -> str:
    """Join segments. Re-encoded rather than stream-copied, because the parts
    come back from the API with independently chosen headers and a copy-concat
    produces a file that plays only sometimes."""
    import subprocess

    # Всё абсолютное и без cwd. Раньше здесь было наоборот: имена внутри
    # segments.txt писались относительно cwd, а сам segments.txt и выходной
    # файл передавались относительными путями ВМЕСТЕ с cwd=каталог склейки —
    # то есть ffmpeg искал run_out/chain/segments.txt внутри run_out/chain и
    # падал всегда. Падал он при этом ПОСЛЕ того, как все платные сегменты уже
    # сгенерированы, и трейсбек летел наружу мимо отчёта.
    out_mp4 = Path(out_mp4).resolve()
    listing = out_mp4.parent / "segments.txt"
    listing.write_text(
        "".join(f"file '{Path(p).resolve()}'\n" for p in paths), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(out_mp4)], check=True, capture_output=True)
    return str(out_mp4)
