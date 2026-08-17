# Что здесь лежит и почему в репозитории

Два файла, скачанные 17 августа 2026, — источник истины по контракту
Wan-Animate. Они здесь, а не в ссылках, по одной причине: **вторая сессия
идёт на арендованной машине, и зависеть от доступности сети в этот момент
нельзя.**

| файл | откуда | лицензия | sha256 |
|---|---|---|---|
| `video_wan2_2_14B_animate.json` | `github.com/Comfy-Org/workflow_templates` → `templates/` | **MIT**, Comfy Org — проверено по `LICENSE` того же репозитория | `06ad8b95e64215328a2a3d2f90495b5bab3e251175e4258d01b977f6dcdecb69` |
| `WanAnimateToVideo.doc.md` | `docs.comfy.org/built-in-nodes/WanAnimateToVideo.md` | документация Comfy Org | `b644959e2d4608eabafb7574b95500753bdd304e4f5e6a071459f6de21728484` |

## Кто из них главнее

**Темплейт.** Документация нод заканчивается строкой «This documentation was
AI-generated» — она поясняет, но не свидетельствует. Где два файла разойдутся,
верх за темплейтом, а расхождение дописывается сюда как находка.

Это не формальность. Первая редакция плана спринта опиралась на список
встроенных нод и утверждала, что кастомные ноды не нужны вовсе. Темплейт
показал обратное: он требует три набора, и это написано в записке внутри него
самого. Утверждение было снято именно так — файлом, а не рассуждением.

## Что из темплейта вычитано

Значения — из `widgets_values`, проводка — из `links`. Воспроизводится
разбором JSON, без Comfy:

```
WanAnimateToVideo лежит внутри сабграфа «Video Sampling and output»,
поэтому в списке нод верхнего уровня его нет — там UUID сабграфа.

reference_image  ← LoadImage                одна фотография
face_video       ← DWPreprocessor#100       hand off, body off, face ON,  512
pose_video       ← DWPreprocessor#101       hand ON,  body ON,  face off, 512
background_video ← DrawMaskOnImage          кадры драйвинга с маской
character_mask   ← BlockifyMask[32] ← GrowMask[10,True] ← Sam2Segmentation
audio            ← GetVideoComponents       звук идёт из драйвинга насквозь
width × height     640 × 640                кратность 16 обязательна
length             77                       при fps 16 это 4.8 с
KSampler           steps 6, cfg 1, euler, simple, denoise 1
ModelSamplingSD3   shift 8
LoRA ×2            lightx2v step-distill 1.0 + WanAnimate_relight 1.0
UNET               Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ
```

Три кастомных набора, названные в записке темплейта:
`comfyui_controlnet_aux`, `ComfyUI-KJNodes`, `ComfyUI-segment-anything-2`.

## Веса: URL и размеры

Сняты `curl -sIL` по ссылкам из записки темплейта 17 августа. Пересчитать —
тем же способом; размер, изменившийся у вендора, надо заметить, а не унаследовать.

```
animate-14B fp8 (Kijai)   17.14 GB
animate-14B bf16          32.18 GB
umt5-xxl fp8               6.27 GB
clip_vision_h              1.18 GB
wan 2.1 vae                0.24 GB
lightx2v distill lora      0.69 GB
relight lora               1.34 GB
                          --------
путь fp8, только веса     26.86 GB
```
