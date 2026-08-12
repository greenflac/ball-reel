# ball_reel — face photo → ball-jump video creative → critic-ranked

Walking-skeleton demo pipeline for the generative-video vacancy. One text brief
plus a reference **face photo** becomes N framings of a short vertical clip of
that person **jumping on a gymnastics ball**; a critic scores each clip and a
stated bar answers which ship. Offline-reproducible with no key, no network,
nothing spent. The live path (`ball_reel.produce`) runs the whole chain end to
end on **one gateway — Pollinations** — the stack the vacancy names: Flux
Kontext for the face-conditioned start frame, Seedance for image-to-video,
ArcFace for the identity check, retry until identity holds.

## Run

```bash
pip install Pillow
python3 -m ball_reel.tools.make_fixtures   # draw the synthetic fixtures once
python3 -m ball_reel                        # offline replay: ranked run + acceptance
python3 -m pytest ball_reel/tests -q        # the gates
```

The offline replay prints, deterministically:

```
accepted 1/3  (identity drift <= 0.20 proxy, motion >= 0.02, opinion >= 0.60)
1  Energy-led   drift 0.027  motion 0.118  opinion 0.82  ships yes
2  Routine-led  drift 0.000  motion 0.000  opinion 0.90  ships no  (frozen clip)
3  Product-led  drift 0.275  motion 0.328  opinion 0.71  ships no  (identity drifted)
```

Three strategies exercise the three failure modes the critic exists to catch: a
consistent, moving clip ships; a lively clip whose character drifts is rejected
on identity; a frozen clip is rejected on motion.

## Two kinds of number, kept apart (the discipline)

- **Computed, no model — a sceptic recomputes them from the committed frames
  and Pillow:**
  - `identity_drift` — does the face stay the same person across the clip? This
    is the vacancy's thinking task answered as a *measurement*, not a promise.
    Ships as a **perceptual proxy** (difference-hash vs the start frame); the
    real instrument is a face-recognition embedding (ArcFace/InsightFace), and
    `identity.arcface_drift` is the named seam it becomes at live time. The proxy
    is labelled everywhere it is printed.
  - `motion_presence` — did the pixels move? Rejects a frozen "video". Presence,
    not plausibility of ball physics — that is the critic's/human's call.
- **Opinion — a model's judgement:** aesthetic/trend/hook. Offline it is a
  **synthetic, hand-authored** verdict read from fixtures and flagged as such.
  Live it is a multimodal judge. This skeleton ships the computed half wired end
  to end and a stand-in for the opinion half, so nothing pretends a judge ran.

## What is NOT here yet — the honest section

- **Nothing generative runs IN this repo.** The live chain (`ball_reel.produce`)
  is written against the documented Pollinations endpoints and is guarded (lazy
  `requests`, env-read `POLLINATIONS_API_KEY`); it needs a key and network, so it
  runs on your bench, not here. The offline replay is the no-key demo.
- **The offline replay's `Gateway.video` has no local model.** Its `--live` flag
  raises `LiveNotWired` — offline is for the *mechanism*. The real generation
  lives in `ball_reel.produce` (Pollinations), a separate entry point.
- **No face encoder in the offline path.** Identity there is the perceptual
  proxy, not a real embedding distance; `produce` uses real `arcface_drift`,
  which raises until insightface is installed on the bench.
- **No lipsync / TTS.** `Gateway.voice` returns silence offline and raises live.
  Note the design fact it encodes: lipsync is **audio-driven**, so "Russian" is
  a property of the TTS layer, not of the lipsync model.
- **Every offline frame and verdict is synthetic.** Drawn locally; verdicts
  hand-authored. The offline run demonstrates the *mechanism*; it is not
  evidence about any model.
- **Bars are chosen defaults**, calibrated to the proxy's scale (a `Bar`
  dataclass), not derived from outcome data.

## Live path — one gateway, the vacancy's stack

```bash
pip install requests Pillow insightface onnxruntime   # + ffmpeg on PATH
export POLLINATIONS_API_KEY=sk_...                     # your bench key (Bearer)
python3 -m ball_reel.produce --face me.jpg --attempts 4
```

`ball_reel.produce` runs the whole chain on **Pollinations** (`gen.pollinations.ai`),
one OpenAI-compatible gateway addressed by model name — text, images, video,
audio, vision all through it. It is the reliability loop, not a single shot:

1. **Upload** the face photo → a media URL the endpoints can reference
   (`pollinations.upload`).
2. **Start frame → Flux Kontext**, conditioned on that face URL — the real face
   as a signal, not a prompt describing one (`pollinations.image(model="kontext",
   image_url=…)`).
3. **Reject early** — ArcFace drift on the start frame vs the photo; if it is
   already not this person, retry the still before spending a video call.
4. **Video → Seedance image-to-video** from the start frame
   (`pollinations.video(model="seedance-2.0", image_url=…)`); frames extracted
   with ffmpeg.
5. **Gate → ArcFace identity across the clip + motion presence.** Pass → return
   the mp4. Fail → next attempt, new seed. No attempt holds → return the best
   one **flagged NOT PASSING**, never a silent near-miss.

- **Identity → ArcFace, local.** The one thing not outsourced to the thing being
  judged. `identity_arcface.arcface_drift` crops the face, embeds it with
  InsightFace `buffalo_l`, measures cosine distance to the reference. The cosine
  arithmetic is unit-tested without the 300 MB model (`test_arcface_math.py`); the
  bar is on the cosine scale (`SAME_PERSON_MAX ≈ 0.35`), re-derived, not the
  proxy's 0.20.
- **Russian voice → ElevenLabs multilingual, same gateway.**
  `pollinations.tts(model="eleven-multilingual-v2")` makes the audio; lipsync is
  **audio-driven**, so "Russian" is a property of the TTS layer, and the lipsync
  model (LivePortrait/Sync-class) consumes that audio. TTS is wired; the lipsync
  drive is the one seam still to add.

**Optional self-hosted alternative (`live_gen.py`).** If you'd rather run on your
own GPU than the gateway, `live_gen.py` wires diffusers SDXL + IP-Adapter FaceID
(+ a `CHARACTER_LORA`) for the start frame and a `VIDEO_API_URL` host for the
video. Same contract, different backend — not required, and not the default.

The offline replay stays the reproducible, no-key demo throughout.

## Reuse

The shape is lifted from the vertical-creative-eval repo: computed-vs-opinion
split, stated acceptance bar, deterministic offline replay, gates that assert a
property and fail on a real mutation (verified: sabotaging `identity_drift`
turns the suite red). This is the video adaptation of that eval, not a rewrite.
