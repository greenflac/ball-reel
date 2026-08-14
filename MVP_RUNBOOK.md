# MVP runbook — run it on your Pollinations bench

Goal: the same critic that runs offline on synthetic frames, now run on frames
YOU generated, with a REAL vision judge (Pollinations), no GPU.

## 0. Install
```bash
pip install Pillow requests
export POLLINATIONS_API_KEY=sk_...   # your bench key (Bearer), from enter.pollinations.ai

# for the real identity check + the full video path (produce.py):
pip install insightface onnxruntime numpy   # buffalo_l (~281 MB) downloads on first use
apt-get install -y ffmpeg                   # frame extraction
```
Egress must allow **both** `gen.pollinations.ai` and `media.pollinations.ai` —
the video endpoint fetches its start frame from the media host server-side, so a
proxy that blocks it breaks image-to-video specifically.

## 0b. Preflight before spending anything
```bash
python3 -m ball_reel.doctor           # free/cheap checks, no video spend
python3 -m ball_reel.doctor --video   # + one short seedance clip (spends)
```
`doctor` now prints its summary in Russian and with **three** counters, not two —
the line is `ПРОШЛО 6, ОТКАЗОВ 0, НЕПРОВЕРЕНО 0 (из 6)`. It used to be documented
here as `ALL CHECKS PASSED (6/6)`; that string no longer exists in the code.

The third counter is the point, not decoration. Without `--video` the most
expensive endpoint is not exercised at all, and the run says so explicitly —
`НЕПРОВЕРЕНО` — instead of quietly counting as a pass. Same for a check that was
skipped because an earlier one failed: it reports the reason rather than a
verdict. **`НЕПРОВЕРЕНО` is not `в порядке`.** The exit code follows: `0` only
when there are neither failures nor unchecked lines, `1` on a failure, `2` when
something could not be checked (and `2` also with no API key at all).

Override the video model/length while shaking out plumbing — `wan-fast` is ~18x
cheaper than seedance and proves the same path:
```bash
BALL_REEL_VIDEO_MODEL=wan-fast BALL_REEL_VIDEO_DURATION=2 python3 -m ball_reel.doctor --video
```

## 1. Get the prompts (one place writes them)
```bash
python3 -c "from ball_reel.mvp import prompts_for; from ball_reel.brief import DEMO_BRIEF; \
[print(f'[{k}]\n{v}\n') for k,v in prompts_for(DEMO_BRIEF).items()]"
```
Three strategies, three prompts. Same brief, three different framings — this is
the prompt-engineering lever, not three seeds of one prompt.

## 2. Make the frames — two ways

A. Automatic (needs the bench to reach Pollinations):
```bash
python3 -m ball_reel.mvp --generate     # writes mvp_input/<strategy>/NN.png
```

B. By hand on your bench: for EACH strategy, paste its prompt into Pollinations,
generate a few frames of the SAME character in slightly different positions
(vary the seed a little) to stand in for the jump sequence, and save them:
```
mvp_input/energy_led/00.png 01.png 02.png ...
mvp_input/product_led/00.png ...
mvp_input/routine_led/00.png ...
```
Honest scope: this frame sequence is the MVP stand-in for a real video-model
clip. The pipeline treats frame 0 as the canonical start and measures the rest
against it. The full end-to-end clip (Flux Kontext start → Seedance
image-to-video, one gateway) is `python3 -m ball_reel.produce --face me.jpg`.

## 3. Run
```bash
python3 -m ball_reel.mvp            # real Pollinations judge for the opinion axis
python3 -m ball_reel.mvp --arcface  # + real ArcFace identity (needs requirements-live)
python3 -m ball_reel.mvp --no-judge # skip the network judge, use offline fallback
```
Prints the ranked run and writes `mvp_report.md`:
```
accepted N/3
#  strategy     id_drift  motion  opinion  ships  reason
1  Energy-led   0.03      0.12     0.82    yes
2  ...          ...       ...      ...     no    identity drift ... / motion ...
```

## What is real here vs the offline demo
- Frames: REAL (yours, from Pollinations) — not synthetic.
- Opinion axis: REAL multimodal judge (Pollinations VLM looks at the frame,
  returns JSON, transcribes on-frame text) — not a hand-authored stand-in.
- Identity drift: perceptual PROXY by default; REAL ArcFace with `--arcface`.
- Motion: real over your frames.
- **Video: now REAL** for `produce.py` — Seedance 2.0 image-to-video, verified
  live (720x1280 h264, 4.04 s, ~97 s per call). `mvp.py` still uses a frame
  sequence as its stand-in; `produce.py` is the true end-to-end clip.
- Still not real: lipsync (a wired seam in `live_gen.py`, not built).

**Scope note added 2026-08-14.** This runbook covers the *public gateway* path
only, and that path is now the **fallback**, not the target. The target is
`python3 -m ball_reel.run_local` on your own card — `animatediff` by default,
no outbound call anywhere in the generation path, because "production cannot
reach the internet" was a requirement rather than a preference. The gateway path
is kept precisely because it is the one verified live; swapping a measured path
for an assumed one is a bad trade. For the local path see `GPU_RUNBOOK.md` and
`TEST_KIT.md` — not this file.

Also note the gate has grown well past `identity + motion` described above. It
now runs, in this order: not-verifiable → identity median → identity p90 →
motion amount → motion physical → anatomy → pose wander → **garment** → loop
(`produce.CHECK_ORDER`), plus, on the local path, garment fit and expression
fidelity as reported-but-non-blocking rows.

## The full path, and what it costs
```bash
python3 -m ball_reel.produce --face me.jpg --attempts 4
```
Per attempt: kontext still (~7 s) → ArcFace screen on the still (rejects a bad
one BEFORE any video spend) → upload (~1.5 s) → Seedance clip (~97 s, 0.72
pollen for 4 s) → ffmpeg frames → ArcFace + motion gate. Passes, or retries.

Two failure modes are normal and handled rather than fatal:
- **Provider moderation (422).** Seedance refuses some phrasings outright, so
  each retry steps the framing wording down (`VIDEO_FRAMING_STEPS`); a refusal
  costs one attempt, never the run.
- **"Identity not verifiable".** Not the same claim as "different person" — it
  means the face was too small in the frames to judge. Frame closer.

Read `produce_report.json` for the evidence behind any verdict: per-attempt
median/p90 drift, coverage, and the face pixel sizes those rest on.

## The one-line pitch for this MVP
"Same selection engine, now on my own Pollinations-generated frames with a real
vision judge — it accepts the consistent, moving clip and rejects the one whose
face drifts and the one that doesn't move, and tells you why."
