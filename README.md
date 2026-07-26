# PCB Defect Inspector

AI-assisted visual inspection for printed circuit boards. Upload a board
photo, and the app finds copper-trace defects with a YOLO11 detector, then
uses Gemini Vision to explain each one in plain language — what's wrong,
why it probably happened, how severe it is, and how to fix it.

Detects six standard defect classes: `missing_hole`, `mouse_bite`,
`open_circuit`, `short`, `spur`, `spurious_copper`.

## Architecture

```
Board photo
    │
    ▼
YOLO11 detector (training/inference.py)
    │  single-pass, or tiled sliding-window for max recall
    ▼
Defect crops (services/crop_defects.py)
    │  contrast boost + upscale + sharpen
    ▼
Gemini Vision explanation (services/explanation_engine.py)
    │  falls back to services/defect_knowledge.py per-defect
    │  if a Gemini call fails — detections are never lost
    ▼
Streamlit UI (app.py) — annotated image, verdict, JSON/Markdown reports
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY (https://aistudio.google.com/apikey)
```

You also need a trained model. Either:

- Train your own (see [Training](#training-a-model) below), or
- Drop an existing `best.pt` at `models/best.pt`, or point the
  `PCB_MODEL_PATH` environment variable at one.

Run the app:

```bash
streamlit run app.py
```

## Detection modes

The sidebar offers three scan modes, trading speed for recall:

| Mode | What it does | When to use it |
|---|---|---|
| **Balanced** | Single pass at 1280px | Good default |
| **Thorough** | 1536px + test-time augmentation (multi-view, ~2-3x slower) | Faint/uncertain defects |
| **Maximum recall (tiled scan)** | Splits large images into overlapping tiles, scans each at full detail, merges with NMS (slowest) | Tiny defects (mouse bites, spurs, pin-holes) on high-resolution photos where a whole-image resize would shrink them past recognition |

## Dataset preparation

This project trains on a merge of DeepPCB + a standard 6-class PCB Defects
dataset. Expected layout:

```
dataset/
  DeepPCB.v1i.yolov11/{train,valid,test}/{images,labels}/ + data.yaml
  PCB Defects.v1i.yolov11/{train,valid,test}/{images,labels}/ + data.yaml
```

```bash
# 1. Confirm both datasets' class lists before merging
python scripts/check_classes.py

# 2. Merge into dataset/PCB_Combined, remapping DeepPCB's class ids to
#    the standard 6-class scheme. Refuses to proceed if either
#    dataset's actual class order doesn't match what the remap table
#    assumes, rather than silently mislabeling everything.
python scripts/merge_datasets.py

# 3. Sanity-check the result: corrupt/missing/malformed labels, class
#    balance, and duplicate images leaking across train/valid/test.
python scripts/verify_dataset.py
```

## Training a model

```bash
python training/train_yolo.py
```

Useful flags: `--epochs`, `--imgsz`, `--batch`, `--model` (base checkpoint),
`--run-name`, `--resume`. Run `--help` for the full list.

Training auto-promotes the result to `models/best.pt` plus a
`models/MANIFEST.json` recording the dataset, config, and validation
metrics used — this is the file `training/inference.py` looks for first,
so the app always uses your latest trained model with no path editing.

Already have several past runs under `runs/` and want to pick a different
one without retraining?

```bash
python scripts/promote_model.py            # lists runs + their metrics, auto-picks best mAP50-95
python scripts/promote_model.py --run pcb_detector3
```

### Model resolution order

`training/inference.py` looks for weights in this order: an explicit path
→ the `PCB_MODEL_PATH` env var → `models/best.pt` → the most recently
modified `runs/*/weights/best.pt`. You should rarely need to think about
this — it's mentioned here mainly so it's clear why the app doesn't need a
hardcoded run-folder name (an earlier version did, which broke every time
the project was retrained).

## Testing

```bash
pytest
```

Runs the full unit-test suite (~100 tests, no GPU/API key/dataset
required — pure logic, mocked network calls). `tests/test_pipeline.py` is
a separate **live** smoke test against real weights + a real Gemini call;
it's skipped automatically by `pytest` unless `ultralytics` is installed,
and can be run directly:

```bash
python tests/test_pipeline.py   # needs models/best.pt + GEMINI_API_KEY + test_images/test_img.jpg
```

## Project layout

```
app.py                       Streamlit UI
services/
  pipeline.py                 Orchestrates detection -> crop -> explanation
  crop_defects.py              Crops + enhances defect regions
  prompt_builder.py            Builds the Gemini prompt (interleaved text+images)
  gemini_service.py            Gemini API wrapper (retries, timeouts, JSON parsing)
  explanation_engine.py        Ties Gemini output to detections, with fallback
  defect_knowledge.py          Static per-class reference used for fallback + prompt hints
  image_quality.py             Blur/resolution/exposure pre-checks
  report_utils.py              Severity rules, verdict logic, Markdown report builder
training/
  inference.py                 YOLO detector, single-pass + tiled inference
  tiling.py                    Tile generation + class-aware NMS (pure, no ML deps)
  model_paths.py                Model path resolution (pure, no ML deps)
  train_yolo.py                 Training entry point
scripts/
  check_classes.py              Compare class lists across dataset yamls
  merge_datasets.py             Merge + remap datasets into the standard scheme
  verify_dataset.py             Dataset integrity + leakage checks
  promote_model.py              Promote a past run's weights without retraining
tests/                         Automated unit tests (pytest) + one live smoke test
```

## Notes on defect explanations

Every defect's explanation carries an `ai_generated` flag. If Gemini is
unavailable or returns something malformed for a specific defect, that
defect still shows up in the report — with standard reference guidance
from `services/defect_knowledge.py` instead of a live Gemini review, and
a small note in the UI saying so. A detection found by the YOLO model is
never dropped just because the explanation step had a bad moment.

## Configuration reference

See `.env.example`. In short:

- `GEMINI_API_KEY` — required.
- `GEMINI_MODEL` — optional, overrides the default Gemini model.
- `PCB_MODEL_PATH` — optional, points at a specific `.pt` file.
