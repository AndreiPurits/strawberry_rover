# Orchestrator Rules

Cursor must act as the project orchestrator.
Cursor must always use the orchestrator prompt when implementing tasks in this repository.
For every new implementation task:

1. Read these files first:
   - AGENTS.md
   - PROJECT_CONTEXT.md
   - DEV_ROADMAP.md
   - CURRENT_STATE.md

2. Determine:
   - current project phase
   - current active task
   - which single agent is responsible

3. Select exactly one responsible agent unless the task truly requires multiple stages.

4. Work only inside that agent's scope.

5. Before editing code, state:
   - selected agent
   - current phase
   - why this task belongs to that agent
   - which files will be changed

6. After implementation, output:
   - what was changed
   - what remains
   - what the next task should be according to DEV_ROADMAP.md

7. Update CURRENT_STATE.md after finishing the task.

8. Never lose project progress.
9. Never restart from scratch.
10. Always continue from the current roadmap position.
11. Restart the simulation stack after every completed stage and verify it starts cleanly.

If the next task is unclear, infer it from:
- CURRENT_STATE.md
- DEV_ROADMAP.md
- current codebase
