# Следующая задача

Этап: 6 — Веб-интерфейс мониторинга системы (Stage 6.2 next)

Ответственный агент:
Visualization & Monitoring Agent

Задача:
Реализовать Stage 6.2 — route recording в `rover_web_interface` после завершенного Stage 6.1 (web control basics).

Актуальные входные условия:
- live ROS2 источники уже подключены в web bridge: `/sim/rover_pose`, `/sim/scene_markers`, `/sim/camera/front/image_raw`, `/sim/camera/bottom_rgb/image_raw`, `/sim/stereo/debug/combined`, `/scan`, `/debug/nav_state`;
- mock источники включены: sensor grid, analytics KPI, arm cameras, rear lidar;
- активная simulation геометрия: удлиненный стартовый край грядок (+1.5 м), скорость scripted-маршрута 1.4x, наклон front camera на -15°.
- web roadmap расширен требованиями route lifecycle и manual control channel.
- Stage 6.1 уже реализован:
  - REST + WS control contract;
  - UI controls: Start/Stop/Manual-Auto;
  - keyboard control (`W/S/A/D`, `Space`);
  - publish `Twist` в `/cmd_vel` через web bridge.
  - joystick/gamepad support добавлен (Gamepad API + UI status + safety fallback).
  - runtime issue web launch устранен (clean startup + стабильный `8080` lifecycle).

Требования:
- не менять ROS2 simulation/navigation/perception пакеты;
- сохранить двухуровневую структуру: Main Field Dashboard + Rover Detail;
- Stage 6.2 (текущий implementation target):
  - добавить route recording mode в web interface;
  - записывать trajectory points с geo/RTK-полями (mock fallback допустим);
  - сохранять metadata: bed length, row spacing, snake-route structure;
  - подготовить route schema для Stage 6.3 (editing) и Stage 6.4 (execution);
- сохранить стабильность Stage 6.1 runtime/control path:
  - `GET /`, `/api/health`, `/api/state`, `WS /ws`;
  - keyboard + joystick control без регрессий.
- зафиксировать versioned route contract (backend -> frontend + storage).

Критерии готовности:
- для Stage 6.2:
  - в UI есть явный режим записи маршрута;
  - backend сохраняет route draft с trajectory + metadata;
  - сохраненный route можно отдать в JSON для последующего редактирования;
  - Stage 6.1 control-функции продолжают работать без регрессий;
  - обновлены `CURRENT_STATE.md`, `NEXT_TASK.md`, `PROJECT_OVERVIEW_RU.md`.