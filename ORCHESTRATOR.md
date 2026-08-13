# Project Orchestrator (rules + current pipeline)

This repository contains **two tightly linked parts**:

- **CV pipeline** (datasets → 3 frozen models → ensemble → stereo/depth distance → grasp inputs)
- **ROS2 rover stack** (simulation/bringup → sensors → navigation → monitoring/web)

Your job as orchestrator: keep work aligned to the current stage, keep interfaces stable, and prevent random refactors.

## Orchestrator rules (always)

For every new implementation task:

1. Read these files first (single source of truth):
   - `README.md` (what the repo is, high-level CV pipeline)
   - `PROJECT_CONTEXT.md` (hardware + constraints)
   - `DEV_ROADMAP.md` (stages + ownership)
   - `CURRENT_STATE.md` (what is already done, **no duplicates**)
   - `NEXT_TASK.md` (what to do next)

2. Determine:
   - current stage (from `DEV_ROADMAP.md`)
   - current active task (from `NEXT_TASK.md`)
   - **one** responsible area/agent (CV / stereo / arm / ROS / web)

3. Select exactly one responsible area unless the task truly spans multiple subsystems.

4. Work only inside that area’s scope.

5. Before editing code, state:
   - selected area/agent
   - current stage
   - why this task belongs there
   - which files will be changed

6. After implementation, output:
   - what changed (short)
   - what remains
   - what the next task should be (from `DEV_ROADMAP.md`)

7. Update `CURRENT_STATE.md` after finishing the task.

## Hard rules

- Never lose project progress.
- Never restart from scratch.
- Always continue from the current roadmap position.
- Keep interfaces stable (see `TOPIC_CONTRACT.md` for ROS topics).

## Restart discipline (ROS stages)

After completing any ROS/web stage, restart the stack and verify it starts cleanly:
- simulation/bringup
- web interface (if used)

## High-signal reference docs (read when relevant)

- **Frozen models & checkpoints**: `docs/model_selection.md`
- **Datasets overview**: `docs/datasets_overview.md`
- **Final detection dataset stats**: `docs/final_detection_dataset_summary.md`
- **Ensemble pipeline + stereo/depth contract**: `reports/ensemble_pipeline/benchmark.md`
- **RoArm ROS2 bridge**: `src/roarm_ros2_http/README.md`
- **Stereo sim topics (ROS)**: `src/rover_fake_stereo/README.md`
- **Real camera perception (ROS)**: `src/rover_perception/README.md`
