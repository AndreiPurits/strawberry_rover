# Datasets Overview (Detection vs Classification)

This project uses **two distinct datasets** that serve **different tasks**.  
Keeping them separate is critical to avoid breaking the pipeline.

## Dataset №1 — Detection (find strawberries / bounding boxes)

- **Source (canonical)**: `data/final_detection_dataset/`
- **Task**: *Where is strawberry on the image?* (object detection)

### Folder structure

- `data/final_detection_dataset/images/` — full images (scenes)
- `data/final_detection_dataset/labels/` — YOLO bbox labels for each image
- `data/final_detection_dataset/bbox_vis/` — visualization previews (optional, derived)

### Label format

YOLO bbox, one object per line:

`class_id x_center y_center width height`

All coordinates are **normalized** to \([0, 1]\).

### Classes (important nuance)

There are **two class conventions** used in the repository:

- **Detection training convention (1-class)**:
  - `0 = strawberry`
  - This is what Ultralytics YOLO training uses in `data/yolo_detection_dataset/` (see below).

- **Raw label content inside `final_detection_dataset` (4-class ripeness ids)**:
  - `0 = green`
  - `1 = turning`
  - `2 = ripe`
  - `3 = rotten_or_overripe`
  - This is reflected in `data/final_detection_dataset/dataset_summary.txt` and is used to generate the crop dataset.

**How this is handled today:** the script `scripts/prepare_yolo_detection_dataset.py` copies images/labels and **rewrites every label line to class_id=0** for detection training, producing:

- `data/yolo_detection_dataset/` with `names: 0: strawberry`

So detection training remains **single-class**, even though the canonical labels include ripeness ids.

### What you must NOT do

- Do **not** turn it into a classification dataset.
- Do **not** replace full images with crops.
- Do **not** delete “hard” scenes (occlusions, leaves, shadows, overlap). These are valuable for detection robustness.

### Current quick stats (from this repo)

From `data/final_detection_dataset/dataset_summary.txt`:

- **Images**: 675
- **Label files**: 675
- **bbox_vis previews**: 675
- **Total bboxes**: 2294

## Dataset №2 — Classification (ripeness / one berry crop per image)

- **Source (canonical)**: `data/final_classification_dataset/`
- **Task**: *What ripeness/state is this strawberry?* (image classification)

### Folder structure

- `data/final_classification_dataset/all/` — all berry crops (one berry per image)
- `data/final_classification_dataset/by_class/`
  - `green/`
  - `turning/`
  - `ripe/`
  - `rotten_or_overripe/`
- `data/final_classification_dataset/preview/` — quick preview sets
- `data/final_classification_dataset/reports/` — dataset reports

### Data format

- **Images are crops**, not full scenes.
- Each crop contains (ideally) **one berry** that occupies most of the frame.

### Classes

- `0 = green`
- `1 = turning`
- `2 = ripe`
- `3 = rotten_or_overripe`

### What you must NOT do

- Do **not** use full-scene images here.
- Do **not** mix this dataset into detection training.
- Do **not** ignore class imbalance when training a classifier.

### Current quick stats (from this repo)

From `data/final_classification_dataset/reports/classification_dataset_summary.txt`:

- **Total crops**: 2294
- **Crops by class**:
  - `green`: 1347
  - `turning`: 266
  - `ripe`: 681
  - `rotten_or_overripe`: 0

## Pipeline architecture (must stay strict)

```
camera image (full scene)
  -> YOLO detection (bbox)
    -> crop (single berry)
      -> classification model (ripeness)
```

## Separation check (no data changes; just analysis)

### Do the two datasets “overlap”?

- They are **linked by design**: classification crops are derived from detection images/labels.
- In this repo, crop filenames encode the original image stem (e.g. `...__objXXX__clsY`), and those bases match the detection stems.

This is **not a violation**: it’s the intended parent/child relationship (full image → crop).

### Are roles mixed anywhere?

Reviewed scripts (high-signal):

- `scripts/prepare_yolo_detection_dataset.py`
  - **Reads**: `data/final_detection_dataset/images` + `labels`
  - **Writes**: `data/yolo_detection_dataset/`
  - **Behavior**: forces class_id → `0` for detection training (**correct for 1-class detector**)

- `scripts/export_classification_crops.py`
  - **Reads**: `data/final_detection_dataset/images` + `labels`
  - **Writes**: `data/final_classification_dataset/`
  - **Behavior**: creates berry crops and splits by ripeness class (**correct for classifier dataset**)

### Potential risk to keep in mind (documented; not auto-fixed)

- Ultralytics detection training expects **single-class** `0=strawberry` (`data/yolo_detection_dataset/`).
  - If you accidentally train detection directly on `data/final_detection_dataset/` without rewriting class ids, you’ll be training a **4-class detector of ripeness**, which is a different task.

## Related auxiliary outputs (not the canonical datasets)

These folders are **helper artifacts** for labeling/QA or bootstrapping and should not be confused with the two canonical datasets:

- `data/yolo_detection_dataset/` — prepared 1-class YOLO training split derived from `final_detection_dataset`
- `data/new_photos_labeled/` — auto-labeled additions from class folders (used for extending datasets; includes bbox_vis for QA)
- `data/strawberry_box_preview/` — inference preview images for quick inspection

