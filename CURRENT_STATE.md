# Текущее состояние проекта

## Что уже сделано
- Сформирована базовая ROS2-архитектура проекта.
- Созданы и используются ключевые пакеты (`rover_bringup`, `rover_description`, `rover_fake_lidar`, `rover_simulation`, `rover_fake_camera`, `rover_fake_stereo`, `rover_perception`, `rover_navigation`).
- Коридорная сцена заменена на упрощенную полевую: 5 грядок (rows) и ровер, маркеры в RViz без линий poles/posts по умолчанию.
- Реализована симуляция движения ровера между 5 грядками по state-machine (`scripted_rows`) с публикацией `/sim/rover_pose` и TF `sim_world -> base_link`.
- LiDAR-модель упрощена под scripted-маршрут: по умолчанию скан не использует poles/posts для геометрии ряда; оставлены только грядки/ровер-маркеры и опциональные явные препятствия (если включены параметром).
- Камерная симуляция переработана под 3 реальные роли: `front_camera`, `bottom_rgb_camera`, `bottom_stereo_camera`.
- Убрана старая неинформативная визуализация «зелёных полос».
- Обновлена TF-модель робота: арочная геометрия и нижние камеры (`bottom_rgb_camera_link`, `bottom_stereo_link`) под базой.
- Исправлена публикация виртуальных камер для RViz (валидные `Image` + RViz-friendly QoS по умолчанию).
- Убраны неожиданные цилиндры из сцены (вне набора beds/posts/rover); опциональные obstacle-маркеры отключены по умолчанию.
- Увеличены габариты ровера в URDF и scene-маркере; ширина ровера теперь больше ширины грядки.
- `rover_navigation_node` адаптирован под row-сцену и снова запущен в `bringup`.
- Добавлен debug topic `/debug/centerline` (`visualization_msgs/Marker`) для RViz.
- `rover_pose_simulator` в bringup переведен в режим `cmd_vel`, движение теперь управляется навигацией.
- Добавлена стабилизация row-following: сглаживание center estimation, сглаживание steering и ограничение шага steering (`max_steer_delta`).
- Добавлены дополнительные RViz debug signals: `/debug/heading` и `/debug/row_confidence`.
- Линии poles/posts удалены из базовой сцены и из фронтального рендера виртуальной камеры, чтобы не влиять на визуально-навигационное поведение в простом маршруте по грядкам.
- Уточнена стерео-визуализация с учетом ограничений RViz2 Foxy: сохранены `/sim/stereo/left|right`, добавлен `/sim/stereo/debug/combined` для удобного отображения.
- Грядки удлинены на 1 м в обе стороны (общая длина поля: 22 м) в LiDAR/движении/камерной геометрии.
- Виртуальные камеры перестроены с `static placeholder` на динамический рендер от `/sim/rover_pose`: фронтальная камера учитывает текущий ряд, прогресс движения и стойки впереди; нижняя RGB-камера показывает грядку под аркой и поперечный оффсет от центра ряда.
- Виртуальная стереопара теперь формируется из двух логически разнесенных по baseline ракурсов (`left/right`), кадры перестали быть идентичными, `debug/combined` сохраняется для RViz2.
- Обновлена модель ровера под концепт арки над грядкой: ширина 1.2 м, длина 1.3 м, высота 1.2 м; обновлены посадочные точки камер (`camera_front_link`, `bottom_rgb_camera_link`, `stereo_link`).
- В `rover_bringup` включен автоматический pre-cleanup (`auto_cleanup_before_start=true` по умолчанию): перед запуском стека завершает старые sim-ноды, чтобы не появлялись дубли ровера в RViz при повторных стартах.
- Рендер front/bottom виртуальных RGB-камер очищен от вспомогательных и декоративных наложений: кадры зависят только от `/sim/rover_pose` и геометрии поля.
- В `rover_navigation_node` добавлен явный детерминированный FSM полного цикла по 5 грядкам: `FOLLOW_ROW -> END_OF_ROW -> TURN_TO_NEXT_ROW -> ALIGN_NEXT_ROW -> FINISHED`.
- Добавлен новый RViz debug topic `/debug/nav_state` с текущим состоянием FSM, номером ряда и направлением прохода.
- Модель движения уточнена под движение **по центрам грядок** (bed centerlines): в `FOLLOW_ROW` добавлена pose-aware коррекция к центру текущей грядки без смены структуры FSM.
- В `FOLLOW_ROW` иерархия управления уточнена: центр грядки теперь является первичным steering-референсом, а LiDAR left/right используется только как ограниченная вторичная guard-коррекция (для безопасности/поддержки), чтобы снизить oscillation от poles.
- По умолчанию `bringup` переведен в deterministic scripted-route режим: `rover_pose_simulator` работает в `motion_mode=scripted_rows`, а `rover_navigation_node` отключен (включается явно через `enable_navigation:=true`).
- `rover_pose_simulator` использует бесконечный boustrophedon-обход грядок: проход по центру грядки, сдвиг на соседнюю в конце ряда, разворот направления, затем повтор цикла.
- Геометрия грядок уточнена асимметрично: стартовый край (со стороны начальной позиции ровера, отрицательная ось X) удлинен на +1.5 м без глобального сдвига поля.
- Скорость scripted-движения ровера увеличена в 1.4x (параметр `scripted_speed`: `0.35 -> 0.49`) при сохранении детерминированной траектории.
- Фронтальная камера наклонена вниз на 15°: обновлен монтажный TF (`camera_front_joint`) и синхронизирован рендер фронтального кадра для меньшей доли неба и большей доли грунта.
- Нижняя RGB-камера исправлена от hallucination: bed-рендер появляется только внутри реального footprint грядок в пределах поля; вне footprint показывается только грунт/фон.
- Добавлен проектный контракт интерфейсов в `TOPIC_CONTRACT.md`.
- Подготовлена архитектура веб-интерфейса как активного этапа: двухуровневая структура `Field Dashboard` + `Rover Detail`, MVP-разделение источников данных (ROS2 live + mock).
- Реализован новый пакет `rover_web_interface` (Stage 6 MVP): FastAPI backend + WebSocket bridge к ROS2 и frontend dashboard.
- В `rover_web_interface` добавлены live-источники `/sim/rover_pose`, `/sim/scene_markers`, `/sim/camera/front/image_raw`, `/sim/camera/bottom_rgb/image_raw`, `/sim/stereo/debug/combined`, `/scan`, `/debug/nav_state`.
- Реализованы REST/WS интерфейсы для веб-клиента и базовый UI: полевая карта (слои), позиция/ориентация/trail ровера, telemetry-панель, camera-потоки, rover-detail окно с mini-map и LiDAR summary.
- Mock-слой добавлен для сенсорной сетки поля, arm-cameras, rear lidar и аналитических KPI.
- Исправлен live-update пути веб-карты ровера: обнаружено, что WebSocket `/ws` не поднимался из-за отсутствующего runtime dependency (`websockets`) в окружении uvicorn, поэтому карта оставалась на initial snapshot; зависимость добавлена в пакет и установлена, WS-поток подтвержден как рабочий.
- Выполнено архитектурное расширение Stage 6 планирования: добавлены требования и этапность для web-функций route lifecycle (record/edit/execute) и manual control channel.
- Реализован Stage 6.1 (web control basics) в `rover_web_interface` без изменений simulation/navigation пакетов:
  - добавлены web-контролы `Start`, `Stop`, `Manual/Auto`;
  - добавлен keyboard control (`W/S/A/D`, `Space`) в dashboard;
  - добавлен backend control contract frontend -> backend (REST + WS control messages);
  - добавлена публикация `geometry_msgs/Twist` в `/cmd_vel` из web-bridge;
  - control-state (`mode`, `started`, `manual_allowed`, `last_command`) встроен в `/api/state` и отдельный `/api/control/state`;
  - соблюдены safety-правила: `Stop` всегда формирует zero-command, выход из `manual` обнуляет команду, `keyup/blur/Space` останавливают ручное движение.
- Доработан runtime Stage 6 web-layer:
  - устранена практическая проблема недоступности dashboard из-за конфликтов порта/висячих web-процессов;
  - в launch `rover_web_interface` добавлен pre-cleanup stale `rover_web_server` с последовательным запуском (через delay), чтобы повторный старт корректно поднимал `0.0.0.0:8080`;
  - подтверждена рабочая цепочка `GET /`, `/api/health`, `/api/state`, `WS /ws` на дефолтном порту.
- Добавлена joystick/gamepad поддержка в Stage 6.1:
  - browser Gamepad API (left stick Y -> linear, left/right X -> angular);
  - deadzone + clamp для безопасных диапазонов;
  - UI-индикация gamepad connection/live axes;
  - отображение текущего control source (`keyboard`/`joystick`);
  - при disconnect/idle joystick отправляется zero-команда (safety).
- Реализован Stage 6.2 (route recording) в `rover_web_interface`:
  - добавлены route controls в UI: `Start recording`, `Stop recording`, `Save route`;
  - backend route-state добавлен в web bridge: `recording`, `current_route`, `saved_routes`;
  - source траектории: live `/sim/rover_pose`, точки формата `x/y/yaw/timestamp/row_index`;
  - сохранение маршрутов реализовано in-memory (MVP), draft и saved routes видимы на карте;
  - добавлены REST endpoints: `/api/routes`, `/api/routes/start`, `/api/routes/stop`, `/api/routes/save`.
- Dashboard переразложен под single-screen режим (no-scroll):
  - фиксированный viewport layout (`100vh`, grid ~58/42);
  - map/cameras/controls/telemetry размещены в одном экране без вертикальной прокрутки;
  - увеличены кнопки и добавлены активные/disabled состояния для operator/demo usability.
- Реализован Stage 6.3 (route editing / map management, MVP) в `rover_web_interface`:
  - добавлен backend route-management API: `select`, `rename`, `delete`, `metadata update`, `rows add/remove`, `trim_last`;
  - добавлено понятие active route (`active_route_id`) и active route snapshot в `/api/state`;
  - в UI добавлена панель `Saved Routes` со списком маршрутов и базовыми действиями управления;
  - добавлен editor route metadata (`name`, `notes`, `row_count`, `spacing_m`) и базовый row metadata management;
  - выбранный маршрут подсвечивается на карте отдельным стилем.
- Исправлены регрессии Stage 6 web control/UI (минимальный patch, без изменения архитектуры):
  - устранен разрыв keyboard-control chain в frontend: удален auto-bootstrap manual/start по `WASD`, чтобы `auto` больше не переключался неявно в `manual` от нажатия клавиш;
  - добавлена мгновенная синхронизация frontend control-state после `Start/Stop/Mode` (через REST response), чтобы ручная команда не терялась из-за задержки WS state update;
  - подтверждена end-to-end цепочка `frontend -> backend -> /cmd_vel -> rover_pose_simulator` для сценариев manual/start/move, stop-immediate и auto-ignore-manual;
  - исправлена обрезка нижних route-кнопок в layout: переразложены высоты левой панели, включен внутренний scroll в telemetry/route area, уменьшены размеры route-action кнопок.
- Добавлен deterministic auto-recovery в `rover_navigation_node` для возврата к грядке после ручного отклонения:
  - введен recovery-state `RECOVER_ROW` c фазами `RETURN_CENTER -> ALIGN_HEADING`;
  - при off-bed отклонении узел выбирает целевую грядку детерминированно (текущая, если близко; иначе nearest row);
  - ровер возвращается к centerline целевой грядки, выравнивает курс по направлению прохода и затем продолжает штатный `FOLLOW_ROW` + обычный FSM переходов.
- Выполнен targeted UI refinement dashboard:
  - верхние control-кнопки собраны в одну компактную горизонтальную строку (Start рядом со Stop);
  - camera block перестроен на 3 одновременные панели (Front/Bottom/Stereo) без strip-like высот;
  - для camera preview задана более читаемая пропорция (приближенно 3:4) при сохранении single-screen operator style.

## Текущая активная фаза
Этап 6 — Веб-интерфейс мониторинга (MVP + Stage 6.1 + Stage 6.2 + Stage 6.3).

## Текущая активная задача
- перейти к Stage 6.4 (route execution):
  - загрузка выбранного active route в execution mode;
  - визуализация прогресса прохождения маршрута;
  - первичный follow-route lifecycle (start/pause/stop) в web UI/backend contract.

## Следующая задача после текущей
- Stage 6.4 — Route execution:
  - выбор/загрузка сохраненного active route;
  - отображение состояния исполнения маршрута и прогресса;
  - подготовка безопасного web execution контракта без изменений core ROS2 architecture.

## Ограничения текущего этапа
- Реально подключена только RGB-камера (в текущем sim bringup используется виртуальная камера).
- LiDAR и stereo пока симулируются.
- Основная визуализация сейчас идет через RViz2.
- В дальнейшем ключевые данные должны быть доступны через веб-интерфейс.
