# RoArm strawberry motion baseline — 2026-08-26

This snapshot freezes the working coarse berry approach before the canonical
RoArm-M3 kinematics layer is introduced. It is documentation only: it does not
change runtime behaviour and it does not authorize motion.

## Runtime entry point

- `scripts/run_roarm_one_shot_approach.sh`
- `scripts/roarm_one_shot_approach.py`
- Default home pose: `DOM_FINAL`
- Planner: `pipelines/roarm_berry_planner.py`
- Planner mode in the latest successful run: `prior_only`
- Physical Cartesian model remains legacy-only at this commit.

## Named poses (firmware radians)

| Pose | base | shoulder | elbow | wrist | roll | hand |
|---|---:|---:|---:|---:|---:|---:|
| HOME | 0.17 | -1.24 | 3.04 | 0.0 | 0.0 | 3.14 |
| HOME2 | 0.17 | 0.0 | 2.55 | 0.0 | 0.0 | 1.08 |
| DOM_FINAL | 0.012271846 | -0.691825335 | 2.867010093 | 0.018407769 | -0.003067962 | 1.537048749 |

Source: `config/roarm_home_joints.yaml`.

## Current operational limits

These are the legacy planner's current software envelopes, not a declaration
of the full mechanical range.

| Joint | hard | soft |
|---|---|---|
| base | [-1.57, 1.57] | [-1.25, 1.25] |
| shoulder | [-1.60, 1.60] | [-1.35, 1.45] |
| elbow | [0.20, 3.20] | [0.35, 3.05] |

Source: `pipelines/roarm_joint_limits.py`.

The old `config/roarm_learn.yaml` also contains 285/375 mm link lengths. Those
values are explicitly legacy and must not be used by the new M3 model.

## Shoulder command bias

For a DOM_FINAL-to-approach move with shoulder delta greater than 1 rad, the
current entry point subtracts 0.062 rad from the shoulder command so feedback
lands near the demonstrated target. This empirical bias belongs to the legacy
coarse approach only.

## Working learned prior

`runs/roarm_learn/dom_final_learned.json`:

- schema: `roarm_berry_learned_v2`
- successful demos: 2
- recorded episodes: 98
- preferred current DOM_FINAL demo: `operator_manual`
- demonstrated delta: base +0.273048581, shoulder +1.558524481,
  elbow -2.129165334 rad

Latest successful run (`runs/roarm_learn/one_shot_run.json`):

- `ok=true`, `planner_mode=prior_only`
- berry depth 0.420 m before, 0.116 m after
- measured end joints: base 0.277650523, shoulder 0.898912742,
  elbow 1.395922517, wrist 0.015339808, roll -0.003067962 rad
- peduncle grasp did not run (`NO_GRASP_POINT`)

The benchmark record contains 3 successful berry-only moves across its saved
presets (2 `arm_slow`, 1 `arm_turbo`). It is not re-run during Phase A because
Phase A forbids physical motion.

## Frozen file hashes (SHA-256)

| Path | SHA-256 |
|---|---|
| `config/roarm_home_joints.yaml` | `cdf1e3a5558732c2a04c3a067ff745ba026abc52239a7a66db3ab4f7751a4ef1` |
| `config/roarm_learn.yaml` | `7c5b118bc6eff2e63d9568b0b0d30f58288583e6039afd83a510e5209e625c8d` |
| `scripts/roarm_one_shot_approach.py` | `16b664bfe1d29280ddd3614766704fbf172ff1edbebb9808ad5c95d3bf4207d9` |
| `pipelines/roarm_berry_planner.py` | `b2fcf08d67eb6c54cdc0395c72afad472448cc4e503c08efa4033372df1a5860` |
| `pipelines/roarm_kinematics.py` | `c5ab469a765abca872c01c5f9a29700f82f99126fcffa59926aae20af08e7b61` |
| `pipelines/roarm_joint_limits.py` | `a682c1d95b24c0505501db8908668bd2a553e68639cc4065c276bb4e5b4c39f9` |
| `runs/roarm_learn/dom_final_learned.json` | `2e949cbe30050944cb718d06b4adde392344661dcbe9de3f2afc009662bd0200` |
| `runs/roarm_learn/one_shot_run.json` | `6333359822bfe08836067f25bb47dc8b3d65a3befbd9f114b8665206a1a242db` |
| `runs/roarm_learn/dom_final_repeat_summary.json` | `bfc14bc517809571ce9bfaa47bc6adbc2e507ea9914f00b7cef8f34da875e74d` |
| `runs/roarm_learn/berry_speed_benchmark.json` | `c705cca40acfc90902b6ea368bc8bdfef93808de4e905137d74e3295aa99790e` |

Run artifacts remain outside Git. Their hashes make this baseline auditable
without adding generated data to the repository.

## Compatibility rule

The new stack is implemented under `pipelines/roarm_m3_kinematics/` because the
repository already has a legacy `pipelines/roarm_kinematics.py`. The legacy
module and planner stay untouched. Runtime remains legacy by default; enabling
physical motion through the new stack is outside Phase A.
