"""produce: a face photo -> a video of THAT person jumping on a ball, reliably.

This is the generation pipeline, not the eval. "Reliably" is not "never fails" —
no generator gives that. It means: the pipeline will not hand you a clip where
the face drifted into someone else or where nothing moved. It generates,
CHECKS identity and motion, and RETRIES; if no attempt passes, it returns the
best one clearly flagged as not-passing rather than pretending.

Everything generative runs through ONE gateway — Pollinations
(gen.pollinations.ai). Every endpoint below is verified live; the response
shapes, ranges and prices are pinned in POLLINATIONS_CONTRACT.md.

  1. start frame -- from the LOCAL face. With only a face: Flux Kontext via
     /v1/images/edits (multipart, no media host needed). With a body or pose
     reference: a multi-reference model instead, because Kontext takes exactly
     one reference and CANNOT accept them. See subject.py for why build has to
     arrive as a picture and clothing does not.
  2. reject early-- ArcFace on the still, and pose distance if a pose was
     specified. Both here rather than after the video, because a bad still
     costs one image call to redo and a video call costs ~18x that.
  3. upload      -- the accepted still -> a public media URL, because the video
     endpoint fetches its input server-side.
  4. video       -- image-to-video. For a loop the still goes in as BOTH
     keyframes, so the model must return the subject to where it began.
  5. gate        -- on the extracted frames: ArcFace identity (median of the
     judgeable frames), motion presence and continuity, limb-length stability,
     loop seam, and pose wander. Pass -> done. Fail -> next attempt.

Every judge is LOCAL — ArcFace for the face, MediaPipe for the body, numpy for
motion. That is the point: the thing being judged must not also be the judge.
Everything generative is an HTTP call; nothing here needs a GPU.

Thresholds are calibrated on live clips, not chosen, and the calibration is
recorded next to each constant. Where a measurement cannot separate two
explanations — a rewritten pose from ordinary motion, a different person from a
face too small to read — it reports that instead of guessing, and the caller
fails it rather than shipping it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .brief import DEMO_BRIEF, Brief
from .gen import STRATEGIES, Strategy
from .motion import (LOOP_MOTION, PHYSICAL_MOTION, loop_seam, motion_quality,
                     trim_to_loop)
from .pose import POSE_WANDER_MAX, limb_consistency, pose_drift
from .subject import UNSPECIFIED, Subject

#: Frame-extraction rate for every gate. One constant because the loop trimmer
#: converts a frame index back into a timestamp with it — a mismatch here would
#: cut the clip in the wrong place.
FRAME_FPS = 6

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
    #: Loop seam as a multiple of a typical frame step; None if unmeasurable.
    loop_ratio: float | None = None
    seamless: bool = False
    #: Largest frame-to-frame step over the median — a teleport/morph detector.
    worst_jump: float | None = None
    #: Mean joint displacement from the pose reference, in torso lengths.
    pose_distance: float | None = None
    #: Worst limb-length variation across the clip (rubber-body detector).
    limb_wobble: float | None = None
    #: Имя ПЕРВОЙ несработавшей проверки (см. CHECK_ORDER), None если прошла.
    #: Отдельно от `reason` намеренно: по имени строится статистика «что
    #: ломается первым», а текст причины меняется при первой же правке
    #: формулировки, и статистика по нему разъезжается молча.
    check: str | None = None


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
    subject: Subject = UNSPECIFIED,
    loop: bool = True,
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

    # The subject is what stops the model inventing a stranger's body: its text
    # half goes in the prompt, its reference images (build, pose) go in as extra
    # inputs — and having any of those switches the start-frame model, since the
    # single-reference editor cannot take them at all.
    start_model = subject.start_model(start_model)
    still_prompt = " ".join(p for p in (
        strat.lead, brief.subject,
        "Keep this exact person's face and identity.",
        subject.to_prompt(), subject.reference_clause(), FRAMING) if p)
    base_motion = " ".join(p for p in (
        strat.lead,
        "The person bounces on the fitness ball, one continuous take.",
        PHYSICAL_MOTION,
        "Keep the same person, same face, same clothing.",
        LOOP_MOTION if loop else "") if p)

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
            refs = subject.reference_roles
            if len(refs) > 1:
                # Multi-reference: every reference must be a public URL, and the
                # ORDER has to match reference_clause()'s "FIRST/SECOND/THIRD".
                urls = [pollinations.upload(face_photo if p == "__face__" else p)
                        for p, _ in refs]
                start = pollinations.compose(
                    f"{still_prompt} (variation {seed})", urls,
                    out_dir / f"start_{n:02d}.png", model=start_model,
                    width=brief.width, height=brief.height, seed=seed % 2147483647)
            else:
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

        # 2b. and if a pose was specified, did the still actually reproduce it?
        #     Judged HERE and not on the video, because on a moving clip a
        #     rewritten pose and ordinary motion score the same (both 0.05-0.30
        #     in torso units) and cannot be told apart. On the still nothing has
        #     moved yet, so the number means what it says.
        if subject.pose_ref and start_drift is not None and start_drift <= bar:
            from .pose import SAME_POSE_MAX, pose_drift

            pose = pose_drift([start], subject.pose_ref,
                              max_pose_distance=SAME_POSE_MAX)
            if not pose["held"]:
                tries.append(Attempt(
                    n, strat.id, start, [], round(start_drift, 4), 0.0, False,
                    f"start frame did not reproduce the reference pose: "
                    f"{pose['note']} — retried before spending a video call",
                    identity=_identity_summary(start_check),
                    pose_distance=pose["median"], check="start_pose"))
                continue
        if start_drift is None or start_drift > bar:
            why = (f"start frame not the same person ({start_drift:.2f} > "
                   f"{bar:.2f})" if start_drift is not None
                   else f"start frame unusable: {start_check['note']}")
            tries.append(Attempt(
                n, strat.id, start, [], 1.0 if start_drift is None else
                round(start_drift, 4), 0.0, False,
                f"{why} — retried before spending a video call",
                identity=_identity_summary(start_check),
                # Два РАЗНЫХ отказа, и путать их дорого: «не тот человек» —
                # это про генератор, «нечего судить» — про то, что лицо в
                # кадре мельче порога и вердикт невозможен. Первое чинится
                # другой моделью старт-кадра, второе — кадрированием.
                check=("start_identity" if start_drift is not None
                       else "start_not_verifiable")))
            continue

        # 3. start frame -> jump video (Seedance image-to-video). The video call
        #    is the expensive one AND the one that can be refused upstream, so a
        #    failure here is recorded and retried rather than raised: a pipeline
        #    that dies on one 422 is not the "reliable" this module claims.
        try:
            start_url = pollinations.upload(start)
            # For a loop, the start frame goes in as BOTH keyframes, so the
            # model has to bring the subject back to where it began.
            mp4 = pollinations.video(
                motion_prompt, out_dir / f"video_{n:02d}.mp4",
                model=video_model,
                image_url=[start_url, start_url] if loop else start_url,
                duration=brief.duration, aspect_ratio=brief.aspect_ratio,
            )
            frames = pollinations.extract_frames(mp4, out_dir / f"frames_{n:02d}",
                                                 fps=FRAME_FPS)
            if loop:
                # Models that do not declare end_frame ignore the second
                # keyframe (wan and happyhorse, live: 1.71 and 1.61), so close
                # the loop locally when it did not close. One decode, no tokens.
                seam = loop_seam(frames)
                if seam["ratio"] is not None and not seam["seamless"]:
                    cut = trim_to_loop(mp4, frames,
                                       out_dir / f"video_{n:02d}_loop.mp4",
                                       fps=FRAME_FPS)
                    if cut.get("seamless") or (cut.get("ratio") or 9) < seam["ratio"]:
                        mp4 = cut["out"]
                        frames = pollinations.extract_frames(
                            mp4, out_dir / f"frames_{n:02d}_loop", fps=FRAME_FPS)
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
        quality = motion_quality(frames)
        seam = loop_seam(frames)
        median = drift["median"]
        # Anatomy across the clip: a real limb keeps its length, a hallucinated
        # one stretches. Independent of identity — the face can be perfect while
        # the body rubber-bands.
        limbs = limb_consistency(frames)
        wander = (pose_drift(frames, subject.pose_ref,
                             max_pose_distance=POSE_WANDER_MAX)
                  if subject.pose_ref else None)
        p90 = drift["p90"]
        coverage = drift["coverage"]
        v = verdict_detail(
            drift=drift, motion=motion, quality=quality, seam=seam,
            limbs=limbs, wander=wander, bar=bar, min_motion=min_motion,
            loop=loop)
        passed, score, reason = v["passed"], v["score"], v["reason"]
        att = Attempt(n, strat.id, start, frames, round(score, 4),
                      round(motion, 4), passed, reason, clip_path=mp4,
                      identity=_identity_summary(drift),
                      loop_ratio=seam["ratio"], seamless=bool(seam["seamless"]),
                      worst_jump=quality["worst_jump"],
                      pose_distance=(wander or {}).get("median"),
                      limb_wobble=(limbs.get("worst") or (None, None))[1],
                      check=v["check"])
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


#: Проверки гейта в том порядке, в котором они применяются. Порядок — часть
#: смысла: сначала то, что делает вердикт невозможным, потом идентичность,
#: потом движение, анатомия, поза, одежда, луп. Имена стабильны, потому что по
#: ним строится статистика «что ломается первым» — а такая статистика бесполезна,
#: если категории переименовываются вместе с формулировкой сообщения.
#: Первые три — ЭКРАН СТАРТ-КАДРА, он отрабатывает ДО единого видео-вызова и
#: потому дешёвый. Он не был представлен в таксономии вовсе, и первый же живой
#: прогон это обнажил: пятнадцать сессий подряд отсеялись именно здесь, а в
#: статистику попали как «error» — то есть самый частый отказ пайплайна был
#: невидим для отчёта, который эту статистику и продаёт.
CHECK_ORDER = ("start_not_verifiable", "start_identity", "start_pose",
               "not_verifiable", "identity_median", "identity_p90",
               "motion_amount", "motion_physical", "anatomy", "pose_wander",
               "garment", "loop", "semantic")

#: Где проходит граница трат. Проверки левее неё стоят одну картинку, правее —
#: ещё и видео-вызов. Разделение попадает в отчёт: «отсеяли до видео» и
#: «отсеяли после» — это разные деньги, и смешивать их в одном проценте брака
#: значит скрывать главное достоинство дешёвого экрана.
PRE_VIDEO_CHECKS = ("start_not_verifiable", "start_identity", "start_pose")


def verdict(**kw) -> tuple:
    """Совместимое представление: (прошло, оценка, причина).

    Тонкая обёртка над `_verdict`. Существует, чтобы имя несработавшей проверки
    можно было добавить, не ломая всех, кто распаковывает тройку.
    """
    passed, score, reason, _ = _verdict(**kw)
    return passed, score, reason


def verdict_detail(**kw) -> dict:
    """То же самое плюс ИМЯ первой несработавшей проверки.

    Нужно стенду: гистограмма «что ломается первым» строится по именам, а не по
    тексту сообщения. Текст меняется при первой же правке формулировки, и
    статистика, собранная по нему, разъезжается молча.
    """
    passed, score, reason, check = _verdict(**kw)
    return {"passed": passed, "score": score, "reason": reason,
            "check": check}


def _verdict(*, drift: dict, motion: float, quality: dict, seam: dict,
             limbs: dict, wander: dict | None, bar: float, min_motion: float,
             loop: bool, garment: dict | None = None,
             semantic: dict | None = None) -> tuple:
    """Свести измерения в один вердикт: (прошло, оценка, причина, проверка).

    Вынесено из `produce` отдельной чистой функцией не ради красоты. Пока эта
    логика жила внутри цикла генерации, проверить её можно было только платным
    прогоном, и мутационный аудит показал ровно это: снятый `MIN_COVERAGE`
    не ронял ни одного теста. Порог, который нечем проверить, — это порог,
    который завтра сдвинут молча.

    Порядок причин важен: сначала то, что делает вердикт невозможным
    (нечего было судить), затем идентичность, затем движение, затем анатомия,
    поза и луп. Возвращается ПЕРВАЯ несработавшая проверка, а не все сразу —
    читателю отчёта нужно знать, с чего начинать, а не список из шести пунктов.
    """
    from .identity_arcface import HARD_DRIFT_MAX, MIN_COVERAGE

    median, p90 = drift.get("median"), drift.get("p90")
    coverage = drift.get("coverage") or 0.0
    if median is None or coverage < MIN_COVERAGE:
        return False, 1.0, (
            f"identity not verifiable: only {coverage:.0%} of frames had a "
            f"face big enough to identify — {drift.get('note', '')}"), "not_verifiable"

    checks = (
        ("identity_median", median <= bar,
         lambda: f"identity drift (median) {median:.2f} > {bar:.2f}"),
        ("identity_p90", p90 is not None and p90 <= HARD_DRIFT_MAX,
         lambda: f"identity unstable: p90 {p90:.2f} > {HARD_DRIFT_MAX:.2f} "
                 f"(drifts inside the clip)"),
        ("motion_amount", motion >= min_motion,
         lambda: f"motion {motion:.3f} < {min_motion:.2f}"),
        ("motion_physical", quality.get("smooth", True),
         lambda: f"motion not physical: {quality.get('note', '')}"),
        ("anatomy", limbs.get("anatomical", True),
         lambda: f"not anatomical: {limbs.get('note', '')}"),
        ("pose_wander", wander is None or wander.get("held"),
         lambda: f"pose wandered off the reference: {wander.get('note', '')}"),
        ("garment", garment is None or garment.get("stable"),
         lambda: f"garment drifts between keyframes: {garment.get('note', '')}"),
        ("loop", seam.get("seamless") or not loop,
         lambda: f"does not loop: {seam.get('note', '')}"),
        # СЕМАНТИКА — ПОСЛЕДНЕЙ, и это не вкус, а цена: ось стоит секунды
        # (загрузка CLIP 4 с плюс 0.04-0.11 с на кадр), остальные — доли
        # секунды в numpy. До неё должны доходить только клипы, прошедшие всё
        # прочее.
        #
        # Условие именно `!= "mismatched"`, а НЕ `== "matches"`. Веса CLIP
        # (~600 МБ) на машине могут отсутствовать, и тогда исход честно
        # `not_measurable`. Гейт, падающий на этом, выключил бы выпуск везде,
        # где весов нет. Но и читаться как «прошло» этот исход не должен:
        # поэтому весь `semantic` кладётся в отчёт целиком, и доля
        # `not_measurable` видна отдельно. Ноль на выходе, означающий «никто не
        # смог», этот проект однажды уже принял за успех.
        ("semantic",
         semantic is None or semantic.get("verdict") != "mismatched",
         lambda: f"frame is not what was ordered: "
                 f"{(semantic or {}).get('note', '')}"),
    )
    for name, ok, why in checks:
        if not ok:
            return False, median, why(), name
    return True, median, "", None


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
             f"{'motion':<8}{'loop':<8}{'jump':<7}{'ok':<5}reason", "-" * 112]
    for a in res.attempts:
        i = a.identity or {}
        p90 = f"{i['p90']:.3f}" if i.get("p90") is not None else "-"
        cov = f"{i['coverage']:.0%}" if i.get("coverage") is not None else "-"
        lo, hi = i.get("face_px_min"), i.get("face_px_max")
        px = f"{lo}-{hi}" if lo is not None else "-"
        lp = f"{a.loop_ratio:.2f}{'*' if a.seamless else ''}" if a.loop_ratio is not None else "-"
        jm = f"{a.worst_jump:.1f}x" if a.worst_jump is not None else "-"
        lines.append(f"{a.n:<4}{a.worst_identity_drift:<12.3f}{p90:<8}{cov:<8}"
                     f"{px:<10}{a.motion:<8.3f}{lp:<8}{jm:<7}"
                     f"{'yes' if a.passed else 'no':<5}{a.reason}")
    lines.append("(loop = seam as a multiple of a typical frame step; * = seamless)")
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
    ap.add_argument("--video-model", default=VIDEO_MODEL,
                    help="seedance-2.0 refuses some references outright; "
                         "wan/veo/happyhorse-1.1 accepted the same inputs")
    ap.add_argument("--out", default="produce_out")
    ap.add_argument("--no-loop", action="store_true",
                    help="do not ask for (or gate on) a seamless loop")
    # The subject: what the face photo cannot carry.
    ap.add_argument("--body-ref", default="",
                    help="photo to copy BUILD and CLOTHING from (text cannot "
                         "set build; switches to a multi-reference model)")
    ap.add_argument("--pose-ref", default="",
                    help="photo to copy the POSE from (face still comes only "
                         "from --face)")
    for f in ("gender", "age", "build", "hair", "outfit", "footwear", "posture"):
        ap.add_argument(f"--{f}", default="")
    ap.add_argument("--from-photo", action="store_true",
                    help="fill unset gender/age from the face photo itself")
    args = ap.parse_args(argv)

    spec = {f: getattr(args, f) for f in
            ("gender", "age", "build", "hair", "outfit", "footwear", "posture")}
    spec.update(body_ref=args.body_ref, pose_ref=args.pose_ref)
    subject = (Subject.from_photo(args.face, **{k: v or None for k, v in spec.items()})
               if args.from_photo else Subject(**spec))

    res = produce(args.face, attempts=args.attempts, subject=subject,
                  loop=not args.no_loop,
                  start_model=args.start_model, video_model=args.video_model,
                  out_dir=args.out)
    print(render(res))
    return 0 if res.passed else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
