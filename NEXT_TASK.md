# Следующая задача

Этап: 6 — Веб-интерфейс мониторинга системы (Stage 6.3 next)

Ответственный агент:
Visualization & Monitoring Agent

Задача:
Реализовать Stage 6.3 — route editing / map management в `rover_web_interface` после завершенного Stage 6.2 (route recording).

Актуальные входные условия:
- live ROS2 источники уже подключены в web bridge: `/sim/rover_pose`, `/sim/scene_markers`, `/sim/camera/front/image_raw`, `/sim/camera/bottom_rgb/image_raw`, `/sim/stereo/debug/combined`, `/scan`, `/debug/nav_state`;
- mock источники включены: sensor grid, analytics KPI, arm cameras, rear lidar;
- Stage 6.1 уже реализован: control contract (REST+WS), Start/Stop, Manual/Auto, keyboard+joystick, публикация `Twist` в `/cmd_vel`;
- Stage 6.2 уже реализован: route recording (`start/stop/save`), in-memory route storage, route snapshot API, live route rendering (draft + saved), single-screen no-scroll layout.

Требования:
- не менять ROS2 simulation/navigation/perception пакеты;
- сохранить текущую стабильность runtime/control path:
  - `GET /`, `/api/health`, `/api/state`, `WS /ws`;
  - keyboard + joystick control без регрессий;
- Stage 6.3 (текущий implementation target):
  - добавить список сохраненных маршрутов (view/select);
  - добавить базовые edit-операции route geometry (минимум move/add/delete point или row-segment);
  - подготовить MVP-инструмент корректировки траектории (spline/corner smoothing допускается упрощенный);
  - сохранить совместимость route schema для Stage 6.4 route execution;
  - обновить backend contract для route edit API.

Критерии готовности:
- для Stage 6.3:
  - в UI доступен route manager со списком saved routes;
  - edit-операции отражаются на карте и сохраняются в backend route storage;
  - формат route JSON остается пригодным для Stage 6.4;
  - Stage 6.1 control и Stage 6.2 recording продолжают работать без регрессий;
  - обновлены `CURRENT_STATE.md`, `NEXT_TASK.md`, `PROJECT_OVERVIEW_RU.md`.