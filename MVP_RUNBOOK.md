# MVP runbook — run it on your Pollinations bench

Goal: the same critic that runs offline on synthetic frames, now run on frames
YOU generated, with a REAL vision judge (Pollinations), no GPU.

## 0. Install
```bash
pip install Pillow requests
export POLLINATIONS_API_KEY=sk_...   # your bench key (Bearer), from enter.pollinations.ai
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
- Still not real: a true video model (frames stand in for a clip), and lipsync.
  Both are wired seams (live_gen.py), not built.

## The one-line pitch for this MVP
"Same selection engine, now on my own Pollinations-generated frames with a real
vision judge — it accepts the consistent, moving clip and rejects the one whose
face drifts and the one that doesn't move, and tells you why."
