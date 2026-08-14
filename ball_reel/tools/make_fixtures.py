"""Draw the SYNTHETIC offline fixtures, deterministically, with Pillow.

Everything this writes is a stand-in: locally drawn frames and hand-authored
opinion verdicts. Nothing here is a model output. The three strategies are
drawn to exercise the three different failure modes the critic exists to catch,
so the offline replay demonstrates real accept/reject logic rather than a single
happy path:

  energy_led  -> a moving clip that stays the same character  -> ACCEPTED
  product_led -> the character drifts frame to frame          -> REJECTED (identity)
  routine_led -> a frozen clip, no motion                     -> REJECTED (motion)

The frames are drawn to CONTROL the two computed measures on purpose:
  * energy_led keeps one big fixed 'character' blob (so drift vs the start frame
    stays low) and sweeps a bright element across the frame (so inter-frame luma
    change — motion presence — is high). Consistent identity, real movement.
  * product_led changes the whole composition each frame (position, size and
    background band), so each later frame disagrees with the start frame past
    the identity bar.
  * routine_led is six byte-identical frames.

Run: python3 -m ball_reel.tools.make_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 216, 384  # 9:16 stand-ins; proportion is what matters
ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"


def _base(bg: int = 230) -> Image.Image:
    return Image.new("RGB", (W, H), (bg, bg, bg))


def _character(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(70, 120, 200))          # ball
    d.rectangle([cx - r // 2, cy - r - 44, cx + r // 2, cy - r], fill=(200, 150, 120))  # body
    d.ellipse([cx - 15, cy - r - 72, cx + 15, cy - r - 42], fill=(220, 180, 150))       # head


def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    index: dict[str, str] = {}

    _save(_base(), FIX / "face_ref.png")  # committed; used by the live path

    # energy_led: FIXED character (identity anchor, so gradient structure and
    # thus drift stay put) + the whole-frame exposure changing each frame
    # (strong inter-frame luma change -> motion present). Same person, real
    # movement/light. A small vertical bob adds real position change too.
    for i in range(6):
        bg = 205 + (i % 2) * 35                           # exposure swing -> motion
        img = _base(bg)
        d = ImageDraw.Draw(img)
        _character(d, W // 2, H // 2 + (i % 3) * 4, 58)   # near-fixed -> low drift
        _save(img, FIX / "video" / "energy_led" / f"{i:02d}.png")
        if i == 0:
            _save(img, FIX / "start" / "energy_led.png")
            index["start/energy_led"] = "fixtures/start/energy_led.png"

    # product_led: the coarse composition changes each frame. A large dark block
    # rotates around the frame (left/right/top/bottom/...), which flips a large
    # share of the perceptual-hash gradient bits, so each later frame disagrees
    # with the start frame past the identity bar. The character moves too.
    blocks = [
        None,                                    # frame 0 = the clean start
        (0, 0, W // 2, H),                       # left half dark
        (0, 0, W, H // 2),                       # top half dark
        (W // 2, 0, W, H),                       # right half dark
        (0, H // 2, W, H),                       # bottom half dark
        (W // 4, H // 4, 3 * W // 4, 3 * H // 4),  # centre block
    ]
    chars = [(108, 192, 58), (60, 300, 30), (150, 110, 66),
             (170, 300, 24), (50, 120, 60), (108, 210, 40)]
    for i, (block, (cx, cy, r)) in enumerate(zip(blocks, chars)):
        img = _base(230)
        d = ImageDraw.Draw(img)
        if block:
            d.rectangle(list(block), fill=(60, 70, 90))
        _character(d, cx, cy, r)
        _save(img, FIX / "video" / "product_led" / f"{i:02d}.png")
        if i == 0:
            _save(img, FIX / "start" / "product_led.png")
            index["start/product_led"] = "fixtures/start/product_led.png"

    # routine_led: six identical frames -> a frozen 'video'. drift 0, motion 0.
    frozen = _base()
    _character(ImageDraw.Draw(frozen), W // 2, H // 2 + 10, 56)
    for i in range(6):
        _save(frozen.copy(), FIX / "video" / "routine_led" / f"{i:02d}.png")
        if i == 0:
            _save(frozen.copy(), FIX / "start" / "routine_led.png")
            index["start/routine_led"] = "fixtures/start/routine_led.png"

    (FIX / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    judge = FIX / "judge"
    judge.mkdir(parents=True, exist_ok=True)
    for sid, op in [("energy_led", 0.82), ("product_led", 0.71), ("routine_led", 0.90)]:
        (judge / f"{sid}.json").write_text(
            json.dumps({"opinion": op, "synthetic": True}, indent=2), encoding="utf-8")

    print(f"fixtures written under {FIX}")


if __name__ == "__main__":
    main()
