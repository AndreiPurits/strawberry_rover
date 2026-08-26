# Peduncle v3.1 plastic lab — domain adaptation for Robolap plastic berries

**Does not change:** peduncle_v3 production weights, berry approach, calyx, association.

## Status gates

| Stage | Ready when |
|-------|------------|
| Capture | `--plastic-capture` after berry approach |
| Dedup | ~80–120 diverse frames |
| Label | OBB GUI group `plastic_lab` |
| Split | grouped 70/15/15 plastic; real test frozen |
| Train | `peduncle_v3.1_plastic` = real train + plastic train |
| Eval gate | plastic target recall ≥85%; real drop ≤5% mAP50; FP controlled |
| Export | lab-only `peduncle_v31_plastic.*` (production untouched) |
| Grasp | **only after gate PASS + miss-crop check** |

## Automatic capture (preferred for diversity)

Safe joint perturbations around `DOM_FINAL` / working pose (base/shoulder/elbow ± small envelope).
Gripper frozen. No peduncle/calyx control. Berry approach baseline untouched.

```bash
# 1) plan only
python3 scripts/roarm_plastic_auto_capture.py --dry-run

# 2) 5 poses smoke (operator at E-stop)
python3 scripts/roarm_plastic_auto_capture.py --smoke

# 3) full ~150 frames / 25–40 poses
python3 scripts/roarm_plastic_auto_capture.py --full --i-am-ready \
  --plastic-target-shots 150 --max-poses 40
```

Outputs → `data/peduncle_plastic_lab/raw/` (`plastic_lab__<session>__<pose>__sXXX_{full,crop}.jpg` + JSON).

Then: `tools/plastic_lab_dedup.py` → OBB GUI → train pipeline as below.

### Framing-only capture (legacy)

After a **working** berry approach:
## 2. Dedup

```bash
python3 tools/plastic_lab_dedup.py
# → data/peduncle_plastic_lab/dedup/ (+ dedup_report.json)
```

## 3. Label (OBB GUI, group `plastic_lab`)

Lab GUI matches v3 meta schema. Scans `manual_capture/`, `dedup/frames/`, and `raw/`:

```bash
python3 tools/peduncle_obb_label_gui.py --port 8765
# http://<orin-ip>:8765/?group=plastic_lab
```

Per frame: berry bbox (seeded from snap meta when present), calyx (+visibility), OBB target/other, scenario (`target_stem` / `only_other` / `no_stem` with 0 OBB).

Labels → `data/peduncle_plastic_lab/labels/`.

## 4. Build dataset + split

```bash
python3 tools/plastic_lab_build_dataset.py
```

- Plastic grouped split ~70/15/15 (no group leakage)
- Real HF train/val/test **symlinked frozen**
- Merged train yaml: real train + plastic train
- Plastic eval yaml separate

Out: `data/peduncle_v3_1_plastic/obb/`

## 5. Train

```bash
python3 tools/train_peduncle_v3_1_plastic.py
# init: runs/deploy_orin_new/models/peduncle.pt
# out:  runs/peduncle_v3_1_plastic/obb_from_v3/weights/best.pt
```

## 6. Eval v3 vs v3.1

```bash
python3 tools/eval_peduncle_v3_vs_v31.py
# → runs/peduncle_v3_1_plastic/eval/compare_v3_vs_v31.json
```

## 7. Export (gate PASS only)

```bash
python3 tools/export_peduncle_v3_1_plastic.py --install-lab
# → runs/peduncle_v3_1_plastic/export/peduncle_v31_plastic.{pt,onnx,engine}
# → runs/deploy_orin_new/models/peduncle_v31_plastic.*  (NOT peduncle.engine)
```

Lab runtime:

```bash
# production v3 (default)
python3 scripts/roarm_one_shot_approach.py --peduncle

# lab v3.1 plastic
export PEDUNCLE_V3_CONFIG=config/peduncle_v3_1_plastic_lab.yaml
python3 scripts/roarm_one_shot_approach.py --peduncle
```

Or: `python3 scripts/run_peduncle_v3_perception.py --config config/peduncle_v3_1_plastic_lab.yaml ...`

Weights (lab only, production untouched):
- `runs/deploy_orin_new/models/peduncle_v31_plastic.{pt,onnx,engine}`
- export mirror: `runs/peduncle_v3_1_plastic/export/`

## 8. Miss-crop probe

```bash
python3 tools/peduncle_manual_detector_test.py \
  --images runs/peduncle_v3_runtime/obb_diag/peduncle_crop_1787658325.jpg \
  --pt runs/peduncle_v3_1_plastic/export/peduncle_v31_plastic.pt \
  --backends pt --imgsz 640 --conf 0.15 \
  --outdir runs/peduncle_manual_test/v31_probe
```

Also probe held-out `images/plastic_test/` crops.

## Physical grasp?

**No** until: labeled pack ≥~80, gate PASS, miss-crop detected at prod conf, lab config only.
