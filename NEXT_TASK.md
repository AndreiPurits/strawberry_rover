# Следующая задача

Этап: 6 — Веб-интерфейс мониторинга системы (Stage 6.4 next)

Ответственный агент:
Visualization & Monitoring Agent

Задача:
Реализовать Stage 6.4 — route execution mode в `rover_web_interface` после завершенного Stage 6.3 (route editing / map management).

Актуальные входные условия:
- live ROS2 источники уже подключены в web bridge: `/sim/rover_pose`, `/sim/scene_markers`, `/sim/camera/front/image_raw`, `/sim/camera/bottom_rgb/image_raw`, `/sim/stereo/debug/combined`, `/scan`, `/debug/nav_state`;
- mock источники включены: sensor grid, analytics KPI, arm cameras, rear lidar;
- Stage 6.1 уже реализован: control contract (REST+WS), Start/Stop, Manual/Auto, keyboard+joystick, публикация `Twist` в `/cmd_vel`;
- Stage 6.2 уже реализован: route recording (`start/stop/save`), in-memory route storage, route snapshot API, live route rendering (draft + saved), single-screen no-scroll layout.
- Stage 6.3 уже реализован:
  - route list panel + active route selection;
  - route actions: rename/delete/select/trim-last;
  - metadata editor (`notes`, `row_count`, `spacing_m`);
  - row metadata management (`rows add/remove`);
  - active route highlight on field map.
- Stage 6 regression-fix pass выполнен:
  - manual/start/stop/auto web-control path стабилизирован;
  - устранено неявное переключение `auto -> manual` от `WASD`;
  - устранена обрезка нижнего route action блока (кнопки полностью доступны).
- Дополнительно выполнены точечные улучшения перед Stage 6.4:
  - в `rover_navigation_node` добавлен deterministic auto-recovery возврата к centerline грядки после manual-deviation;
  - dashboard control-row и camera layout переразложены для более читаемого single-screen вида (Start/Stop рядом, 3 камеры без strip-like панелей).

Требования:
- не менять ROS2 simulation/navigation/perception пакеты;
- сохранить текущую стабильность runtime/control path:
  - `GET /`, `/api/health`, `/api/state`, `WS /ws`;
  - keyboard + joystick control без регрессий;
- Stage 6.4 (текущий implementation target):
  - добавить режим исполнения выбранного маршрута (active route execution lifecycle);
  - добавить status/state для выполнения (`idle/running/paused/finished`);
  - отображать execution progress на карте и в telemetry/status panel;
  - подготовить backend control contract для `execute/start/pause/stop`;
  - сохранить обратную совместимость Stage 6.1 и Stage 6.2/6.3 API.

Критерии готовности:
- для Stage 6.4:
  - можно выбрать saved active route и запустить его execution-mode;
  - execution state и progress корректно отображаются в UI;
  - backend execution endpoints работают стабильно;
  - Stage 6.1 control, Stage 6.2 recording и Stage 6.3 editing продолжают работать без регрессий;
  - обновлены `CURRENT_STATE.md`, `NEXT_TASK.md`, `PROJECT_OVERVIEW_RU.md`.