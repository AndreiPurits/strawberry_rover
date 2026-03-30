"""Operator-oriented GUI for RoArm-M3 (local only, no ROS)."""

from __future__ import annotations

import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from roarm_client import RoArmClient, RoArmClientError
from sequence_manager import SequenceManager


JOINTS: List[Tuple[int, str]] = [
    (1, "BASE_JOINT"),
    (2, "SHOULDER_JOINT"),
    (3, "ELBOW_JOINT"),
    (4, "WRIST_JOINT"),
    (5, "ROLL_JOINT"),
    (6, "EOAT_JOINT"),
]

AXES: List[Tuple[int, str]] = [
    (1, "X"),
    (2, "Y"),
    (3, "Z"),
    (4, "T"),
    (5, "R"),
    (6, "G"),
]


class RoArmTestWindow(QMainWindow):
    """Main window split into MAIN and ADVANCED operator modes."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RoArm-M3 Operator Tool (Local, No ROS)")
        self.resize(1280, 860)
        self._client = RoArmClient(ip="192.168.1.87", timeout_sec=5.0)

        self._history_file = Path(__file__).resolve().parent / "tested_workspace.json"
        self._tested_history: List[Dict[str, Any]] = self._load_history()
        self._last_target_idx: Optional[int] = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.advanced_tab = QWidget()
        self.sequence_tab = QWidget()
        self.tabs.addTab(self.main_tab, "MAIN")
        self.tabs.addTab(self.advanced_tab, "ADVANCED")
        self.tabs.addTab(self.sequence_tab, "Sequence")
        root_layout.addWidget(self.tabs)

        self._sequence_file = Path(__file__).resolve().parent / "sequences.json"
        self._sequence_stop_requested = False
        self._sequence_table_updating = False
        self._sequence_manager = SequenceManager(self._client)

        self._build_main_tab()
        self._build_advanced_tab()
        self._build_sequence_tab()
        self._refresh_history_table()
        self._update_reachability_status()

    # -------------------- main/advanced layout --------------------
    def _build_main_tab(self) -> None:
        layout = QVBoxLayout(self.main_tab)
        layout.addWidget(self._build_connection_block())
        layout.addWidget(self._build_system_block())
        layout.addWidget(self._build_gripper_block())
        layout.addWidget(self._build_target_block())
        layout.addWidget(self._build_current_state_block())

    def _build_advanced_tab(self) -> None:
        layout = QVBoxLayout(self.advanced_tab)
        layout.addWidget(self._build_joint_block())
        layout.addWidget(self._build_axis_block())
        layout.addWidget(self._build_direct_move_block())
        layout.addWidget(self._build_debug_log_block())

    def _build_sequence_tab(self) -> None:
        layout = QVBoxLayout(self.sequence_tab)

        self.sequence_table = QTableWidget(0, 10)
        self.sequence_table.setHorizontalHeaderLabels(
            ["type", "x", "y", "z", "t", "r", "g", "mode", "reachability", "run_status"]
        )
        self.sequence_table.horizontalHeader().setStretchLastSection(True)
        self.sequence_table.cellChanged.connect(self._on_sequence_cell_changed)
        layout.addWidget(self.sequence_table)

        manage_row = QHBoxLayout()
        add_target_btn = QPushButton("+ Add Target Point")
        add_home_btn = QPushButton("+ Add Home")
        delete_btn = QPushButton("Delete step")
        up_btn = QPushButton("Move up")
        down_btn = QPushButton("Move down")
        add_target_btn.clicked.connect(self._on_add_sequence_target)
        add_home_btn.clicked.connect(self._on_add_sequence_home)
        delete_btn.clicked.connect(self._on_delete_sequence_step)
        up_btn.clicked.connect(lambda: self._on_move_sequence_step(-1))
        down_btn.clicked.connect(lambda: self._on_move_sequence_step(1))
        manage_row.addWidget(add_target_btn)
        manage_row.addWidget(add_home_btn)
        manage_row.addWidget(delete_btn)
        manage_row.addWidget(up_btn)
        manage_row.addWidget(down_btn)
        layout.addLayout(manage_row)

        io_row = QHBoxLayout()
        save_btn = QPushButton("Save sequence")
        load_btn = QPushButton("Load sequence")
        load_test_btn = QPushButton("Load test sequence")
        save_btn.clicked.connect(self._on_save_sequence)
        load_btn.clicked.connect(self._on_load_sequence)
        load_test_btn.clicked.connect(self._on_load_test_sequence)
        io_row.addWidget(save_btn)
        io_row.addWidget(load_btn)
        io_row.addWidget(load_test_btn)
        layout.addLayout(io_row)

        run_row = QHBoxLayout()
        run_btn = QPushButton("Run sequence")
        stop_btn = QPushButton("Stop")
        run_btn.clicked.connect(self._on_run_sequence)
        stop_btn.clicked.connect(self._on_stop_sequence)
        run_row.addWidget(run_btn)
        run_row.addWidget(stop_btn)
        layout.addLayout(run_row)

        self.sequence_log = QTextEdit()
        self.sequence_log.setReadOnly(True)
        self.sequence_log.setPlaceholderText("Sequence execution log")
        layout.addWidget(self.sequence_log)

    # -------------------- blocks --------------------
    def _build_connection_block(self) -> QGroupBox:
        box = QGroupBox("1) Connection")
        layout = QHBoxLayout(box)
        self.ip_edit = QLineEdit("192.168.1.87")
        self.ip_edit.setMinimumWidth(180)
        test_btn = QPushButton("Test connection")
        refresh_btn = QPushButton("Refresh status")
        test_btn.clicked.connect(self._on_test_connection)
        refresh_btn.clicked.connect(self._on_refresh_status)

        layout.addWidget(QLabel("IP:"))
        layout.addWidget(self.ip_edit)
        layout.addWidget(test_btn)
        layout.addWidget(refresh_btn)
        return box

    def _build_system_block(self) -> QGroupBox:
        box = QGroupBox("2) System")
        layout = QHBoxLayout(box)
        home_btn = QPushButton("Home")
        torque_on_btn = QPushButton("Torque ON")
        torque_off_btn = QPushButton("Torque OFF")
        home_btn.clicked.connect(self._on_home)
        torque_on_btn.clicked.connect(lambda: self._on_torque(True))
        torque_off_btn.clicked.connect(lambda: self._on_torque(False))
        layout.addWidget(home_btn)
        layout.addWidget(torque_on_btn)
        layout.addWidget(torque_off_btn)
        return box

    def _build_gripper_block(self) -> QGroupBox:
        box = QGroupBox("3) Gripper")
        layout = QHBoxLayout(box)
        open_btn = QPushButton("Open")
        close_btn = QPushButton("Close")
        open_btn.clicked.connect(self._on_open_gripper)
        close_btn.clicked.connect(self._on_close_gripper)
        layout.addWidget(open_btn)
        layout.addWidget(close_btn)
        return box

    def _build_target_block(self) -> QGroupBox:
        box = QGroupBox("4) TARGET POINT (T:104)")
        layout = QFormLayout(box)

        self.xyz_x = self._float_spin(-1000.0, 1000.0, 1.0, 235.0, 2)
        self.xyz_y = self._float_spin(-1000.0, 1000.0, 1.0, 0.0, 2)
        self.xyz_z = self._float_spin(-1000.0, 1000.0, 1.0, 234.0, 2)
        self.xyz_spd = self._float_spin(0.0, 5.0, 0.01, 0.25, 3)
        self.xyz_t = self._float_spin(-6.28, 6.28, 0.01, 0.0, 3)
        self.xyz_r = self._float_spin(-6.28, 6.28, 0.01, 0.0, 3)
        self.xyz_g = self._float_spin(-6.28, 6.28, 0.01, 3.14, 3)

        self.xyz_x.valueChanged.connect(self._on_xyz_changed)
        self.xyz_y.valueChanged.connect(self._on_xyz_changed)
        self.xyz_z.valueChanged.connect(self._on_xyz_changed)

        layout.addRow("X (mm)", self.xyz_x)
        layout.addRow("Y (mm)", self.xyz_y)
        layout.addRow("Z (mm)", self.xyz_z)
        layout.addRow("Speed", self.xyz_spd)

        self.show_trg_checkbox = QCheckBox("Show T/R/G")
        self.show_trg_checkbox.toggled.connect(self._toggle_trg_fields)
        layout.addRow(self.show_trg_checkbox)

        self.trg_widget = QWidget()
        trg_layout = QFormLayout(self.trg_widget)
        trg_layout.setContentsMargins(0, 0, 0, 0)
        trg_layout.addRow("T (rad)", self.xyz_t)
        trg_layout.addRow("R (rad)", self.xyz_r)
        trg_layout.addRow("G (rad)", self.xyz_g)
        self.trg_widget.setVisible(False)
        layout.addRow(self.trg_widget)

        action_row = QHBoxLayout()
        check_btn = QPushButton("Check")
        move_btn = QPushButton("Move")
        check_btn.clicked.connect(self._on_check_reachability)
        move_btn.clicked.connect(self._on_move_xyz)
        action_row.addWidget(check_btn)
        action_row.addWidget(move_btn)
        layout.addRow(action_row)

        self.reachability_label = QLabel("Reachability status: untested")
        self.reachability_label.setWordWrap(True)
        layout.addRow(self.reachability_label)

        mark_row = QHBoxLayout()
        mark_ok_btn = QPushButton("Mark SUCCESS")
        mark_fail_btn = QPushButton("Mark FAILED")
        mark_ok_btn.clicked.connect(lambda: self._mark_last_target("success"))
        mark_fail_btn.clicked.connect(lambda: self._mark_last_target("failed"))
        mark_row.addWidget(mark_ok_btn)
        mark_row.addWidget(mark_fail_btn)
        layout.addRow(mark_row)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["time", "x", "y", "z", "result", "mode"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addRow(QLabel("Tested workspace history"))
        layout.addRow(self.history_table)
        return box

    def _build_current_state_block(self) -> QGroupBox:
        box = QGroupBox("5) CURRENT STATE")
        layout = QVBoxLayout(box)
        self.status_view = QTextEdit()
        self.status_view.setReadOnly(True)
        self.status_view.setPlaceholderText("Press 'Refresh status' to display pose, joints, voltage and loads.")
        layout.addWidget(self.status_view)
        return box

    def _build_joint_block(self) -> QGroupBox:
        box = QGroupBox("ADVANCED 1) Joint Control (T:101)")
        layout = QGridLayout(box)
        layout.addWidget(QLabel("Joint"), 0, 0)
        layout.addWidget(QLabel("rad"), 0, 1)
        layout.addWidget(QLabel("Action"), 0, 2)
        self.joint_rad_inputs: Dict[int, QDoubleSpinBox] = {}

        for row, (joint_id, joint_name) in enumerate(JOINTS, start=1):
            rad = self._float_spin(-6.28, 6.28, 0.01, 0.0, 4)
            btn = QPushButton("Send")
            btn.clicked.connect(lambda _checked=False, j=joint_id: self._on_send_joint(j))
            self.joint_rad_inputs[joint_id] = rad
            layout.addWidget(QLabel(f"{joint_id}: {joint_name}"), row, 0)
            layout.addWidget(rad, row, 1)
            layout.addWidget(btn, row, 2)
        return box

    def _build_axis_block(self) -> QGroupBox:
        box = QGroupBox("ADVANCED 2) Axis Control (T:103)")
        layout = QHBoxLayout(box)
        self.axis_combo = QComboBox()
        for axis_id, axis_name in AXES:
            self.axis_combo.addItem(f"{axis_id}: {axis_name}", axis_id)
        self.axis_pos = self._float_spin(-1000.0, 1000.0, 0.1, 0.0, 3)
        move_btn = QPushButton("Move")
        move_btn.clicked.connect(self._on_move_axis)
        layout.addWidget(QLabel("Axis"))
        layout.addWidget(self.axis_combo)
        layout.addWidget(QLabel("pos"))
        layout.addWidget(self.axis_pos)
        layout.addWidget(move_btn)
        return box

    def _build_direct_move_block(self) -> QGroupBox:
        box = QGroupBox("ADVANCED 3) Direct Move (T:1041)")
        layout = QFormLayout(box)
        self.dir_x = self._float_spin(-1000.0, 1000.0, 1.0, 235.0, 2)
        self.dir_y = self._float_spin(-1000.0, 1000.0, 1.0, 0.0, 2)
        self.dir_z = self._float_spin(-1000.0, 1000.0, 1.0, 234.0, 2)
        self.dir_t = self._float_spin(-6.28, 6.28, 0.01, 0.0, 3)
        self.dir_r = self._float_spin(-6.28, 6.28, 0.01, 0.0, 3)
        self.dir_g = self._float_spin(-6.28, 6.28, 0.01, 3.14, 3)
        move_direct_btn = QPushButton("Move Direct")
        move_direct_btn.clicked.connect(self._on_move_xyz_direct)
        layout.addRow("X (mm)", self.dir_x)
        layout.addRow("Y (mm)", self.dir_y)
        layout.addRow("Z (mm)", self.dir_z)
        layout.addRow("T (rad)", self.dir_t)
        layout.addRow("R (rad)", self.dir_r)
        layout.addRow("G (rad)", self.dir_g)
        layout.addRow(move_direct_btn)
        return box

    def _build_debug_log_block(self) -> QGroupBox:
        box = QGroupBox("ADVANCED 4) Debug Log")
        layout = QVBoxLayout(box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Commands, raw responses, errors")
        layout.addWidget(self.log_view)
        return box

    # -------------------- helpers --------------------
    @staticmethod
    def _float_spin(min_v: float, max_v: float, step: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _sync_ip(self) -> None:
        self._client.set_ip(self.ip_edit.text())

    def _log_command(self, url: str, response: str) -> None:
        self.log_view.append(f"URL: {url}")
        self.log_view.append(f"Response: {response.strip()}")
        self.log_view.append("-" * 70)

    def _log_error(self, message: str) -> None:
        self.log_view.append(f"ERROR: {message}")
        self.log_view.append("-" * 70)

    def _run_action(self, action) -> bool:
        self._sync_ip()
        try:
            url, response = action()
            self._log_command(url, response)
            return True
        except RoArmClientError as exc:
            self._log_error(str(exc))
            QMessageBox.warning(self, "RoArm Error", str(exc))
            return False
        except Exception as exc:  # pragma: no cover
            self._log_error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Unexpected Error", str(exc))
            return False

    def _toggle_trg_fields(self, show: bool) -> None:
        self.trg_widget.setVisible(show)

    def _xyz_values(self) -> Tuple[float, float, float, float, float, float, float]:
        return (
            self.xyz_x.value(),
            self.xyz_y.value(),
            self.xyz_z.value(),
            self.xyz_t.value(),
            self.xyz_r.value(),
            self.xyz_g.value(),
            self.xyz_spd.value(),
        )

    @staticmethod
    def _point_key(x: float, y: float, z: float) -> str:
        return f"{round(x, 1)}|{round(y, 1)}|{round(z, 1)}"

    def _workspace_overlay(self, x: float, y: float, z: float) -> str:
        key = self._point_key(x, y, z)
        points = [row for row in self._tested_history if row.get("key") == key]
        if not points:
            return "untested"
        latest = points[-1].get("result", "unknown")
        if latest == "success":
            return "tested reachable"
        if latest == "failed":
            return "previously failed"
        return "untested"

    def _is_risky(self, x: float, y: float, z: float) -> tuple[bool, str]:
        r = math.sqrt(x * x + y * y)
        reasons: List[str] = []
        if r < 80:
            reasons.append("r < 80 (too close)")
        if r > 420:
            reasons.append("r > 420 (too far)")
        if z < -100:
            reasons.append("z < -100 (too low)")
        if z > 380:
            reasons.append("z > 380 (too high)")
        if reasons:
            return True, "; ".join(reasons)
        return False, "within soft heuristic zone"

    def _compose_reachability(self, x: float, y: float, z: float) -> tuple[str, str]:
        overlay = self._workspace_overlay(x, y, z)
        risky, reason = self._is_risky(x, y, z)
        if overlay == "tested reachable":
            return f"Reachability status: tested reachable ({reason})", "green"
        if overlay == "previously failed":
            return f"Reachability status: previously failed ({reason})", "orange"
        if risky:
            return f"Reachability status: risky ({reason})", "orange"
        return f"Reachability status: untested ({reason})", "#1e90ff"

    def _update_reachability_status(self) -> None:
        x, y, z, *_ = self._xyz_values()
        text, color = self._compose_reachability(x, y, z)
        self.reachability_label.setText(text)
        self.reachability_label.setStyleSheet(f"color: {color};")

    def _on_xyz_changed(self) -> None:
        self._update_reachability_status()

    def _load_history(self) -> List[Dict[str, Any]]:
        if not self._history_file.exists():
            return []
        try:
            raw = json.loads(self._history_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [entry for entry in raw if isinstance(entry, dict)]
        except Exception:
            pass
        return []

    def _save_history(self) -> None:
        self._history_file.write_text(
            json.dumps(self._tested_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _refresh_history_table(self) -> None:
        self.history_table.setRowCount(len(self._tested_history))
        for row_idx, row in enumerate(self._tested_history):
            values = [
                str(row.get("time", "")),
                str(row.get("x", "")),
                str(row.get("y", "")),
                str(row.get("z", "")),
                str(row.get("result", "")),
                str(row.get("mode", "")),
            ]
            for col_idx, value in enumerate(values):
                self.history_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.history_table.scrollToBottom()

    def _set_sequence_row(self, row_idx: int, data: Dict[str, Any]) -> None:
        defaults = {
            "type": "target",
            "x": 235.0,
            "y": 0.0,
            "z": 234.0,
            "t": 0.0,
            "r": 0.0,
            "g": 3.14,
            "mode": "104",
            "reachability": "",
            "run_status": "",
        }
        merged = {**defaults, **data}
        values = [
            str(merged["type"]),
            str(merged["x"]),
            str(merged["y"]),
            str(merged["z"]),
            str(merged["t"]),
            str(merged["r"]),
            str(merged["g"]),
            str(merged["mode"]),
            str(merged["reachability"]),
            str(merged["run_status"]),
        ]
        self._sequence_table_updating = True
        for col_idx, value in enumerate(values):
            self.sequence_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self._sequence_table_updating = False

    def _clear_sequence_status_columns(self) -> None:
        self._sequence_table_updating = True
        for row in range(self.sequence_table.rowCount()):
            self.sequence_table.setItem(row, 8, QTableWidgetItem(""))
            self.sequence_table.setItem(row, 9, QTableWidgetItem(""))
        self._sequence_table_updating = False

    def _on_add_sequence_target(self) -> None:
        row = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row)
        x, y, z, t, r, g, _spd = self._xyz_values()
        self._set_sequence_row(
            row,
            {
                "type": "target",
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "t": round(t, 3),
                "r": round(r, 3),
                "g": round(g, 3),
                "mode": "104",
            },
        )
        self._refresh_sequence_reachability()

    def _on_add_sequence_home(self) -> None:
        row = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row)
        self._set_sequence_row(row, {"type": "home", "mode": "100", "x": "", "y": "", "z": "", "t": "", "r": "", "g": ""})
        self._refresh_sequence_reachability()

    def _on_delete_sequence_step(self) -> None:
        row = self.sequence_table.currentRow()
        if row >= 0:
            self.sequence_table.removeRow(row)
            self._refresh_sequence_reachability()

    def _row_values(self, row: int) -> Dict[str, Any]:
        def cell(col: int) -> str:
            item = self.sequence_table.item(row, col)
            return item.text().strip() if item is not None else ""

        return {
            "type": cell(0).lower() or "target",
            "x": cell(1),
            "y": cell(2),
            "z": cell(3),
            "t": cell(4),
            "r": cell(5),
            "g": cell(6),
            "mode": cell(7) or "104",
        }

    def _rows_to_sequence(self) -> List[Dict[str, Any]]:
        seq: List[Dict[str, Any]] = []
        for row in range(self.sequence_table.rowCount()):
            vals = self._row_values(row)
            if vals["type"] == "home":
                seq.append({"type": "home"})
                continue
            if vals["type"] == "gripper":
                action = str(vals["mode"]).lower() if vals["mode"] else "open"
                if action not in ("open", "close"):
                    action = "open"
                seq.append({"type": "gripper", "action": action})
                continue
            if vals["type"] == "delay":
                try:
                    sec = float(vals["x"] or 1.0)
                except ValueError:
                    raise RoArmClientError(f"Invalid delay value in sequence row {row + 1}")
                seq.append({"type": "delay", "sec": sec})
                continue
            try:
                seq.append(
                    {
                        "type": "target",
                        "x": float(vals["x"]),
                        "y": float(vals["y"]),
                        "z": float(vals["z"]),
                        "t": float(vals["t"] or 0.0),
                        "r": float(vals["r"] or 0.0),
                        "g": float(vals["g"] or 3.14),
                    }
                )
            except ValueError:
                raise RoArmClientError(f"Invalid numeric value in sequence row {row + 1}")
        return seq

    def _refresh_sequence_reachability(self) -> None:
        self._sequence_table_updating = True
        for row in range(self.sequence_table.rowCount()):
            vals = self._row_values(row)
            if vals["type"] == "home":
                self.sequence_table.setItem(row, 8, QTableWidgetItem("home step"))
                continue
            if vals["type"] == "gripper":
                self.sequence_table.setItem(row, 8, QTableWidgetItem("gripper step"))
                continue
            if vals["type"] == "delay":
                self.sequence_table.setItem(row, 8, QTableWidgetItem("delay step"))
                continue
            try:
                x = float(vals["x"])
                y = float(vals["y"])
                z = float(vals["z"])
                text, _color = self._compose_reachability(x, y, z)
                self.sequence_table.setItem(row, 8, QTableWidgetItem(text.replace("Reachability status: ", "")))
            except ValueError:
                self.sequence_table.setItem(row, 8, QTableWidgetItem("invalid point"))
        self._sequence_table_updating = False

    def _on_sequence_cell_changed(self, _row: int, _col: int) -> None:
        if self._sequence_table_updating:
            return
        self._refresh_sequence_reachability()

    def _on_move_sequence_step(self, direction: int) -> None:
        row = self.sequence_table.currentRow()
        if row < 0:
            return
        target = row + direction
        if target < 0 or target >= self.sequence_table.rowCount():
            return
        current = [self.sequence_table.item(row, c).text() if self.sequence_table.item(row, c) else "" for c in range(self.sequence_table.columnCount())]
        other = [self.sequence_table.item(target, c).text() if self.sequence_table.item(target, c) else "" for c in range(self.sequence_table.columnCount())]
        self._sequence_table_updating = True
        for c, value in enumerate(other):
            self.sequence_table.setItem(row, c, QTableWidgetItem(value))
        for c, value in enumerate(current):
            self.sequence_table.setItem(target, c, QTableWidgetItem(value))
        self._sequence_table_updating = False
        self.sequence_table.selectRow(target)

    def _on_save_sequence(self) -> None:
        try:
            sequence = self._rows_to_sequence()
            self._sequence_file.write_text(json.dumps(sequence, ensure_ascii=False, indent=2), encoding="utf-8")
            self.sequence_log.append(f"Saved sequence to {self._sequence_file}")
        except Exception as exc:
            QMessageBox.warning(self, "Save sequence error", str(exc))

    def _on_load_sequence(self) -> None:
        if not self._sequence_file.exists():
            QMessageBox.information(self, "Load sequence", f"No file: {self._sequence_file}")
            return
        try:
            raw = json.loads(self._sequence_file.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Sequence file must contain a list")
            self.sequence_table.setRowCount(0)
            for step in raw:
                row = self.sequence_table.rowCount()
                self.sequence_table.insertRow(row)
                if isinstance(step, dict) and step.get("type") == "home":
                    self._set_sequence_row(row, {"type": "home", "mode": "100", "x": "", "y": "", "z": "", "t": "", "r": "", "g": ""})
                elif isinstance(step, dict) and step.get("type") == "gripper":
                    self._set_sequence_row(
                        row,
                        {
                            "type": "gripper",
                            "x": "",
                            "y": "",
                            "z": "",
                            "t": "",
                            "r": "",
                            "g": "",
                            "mode": step.get("action", "open"),
                        },
                    )
                elif isinstance(step, dict) and step.get("type") == "delay":
                    self._set_sequence_row(
                        row,
                        {
                            "type": "delay",
                            "x": step.get("sec", 1.0),
                            "y": "",
                            "z": "",
                            "t": "",
                            "r": "",
                            "g": "",
                            "mode": "delay",
                        },
                    )
                elif isinstance(step, dict):
                    self._set_sequence_row(
                        row,
                        {
                            "type": "target",
                            "x": step.get("x", 235.0),
                            "y": step.get("y", 0.0),
                            "z": step.get("z", 234.0),
                            "t": step.get("t", 0.0),
                            "r": step.get("r", 0.0),
                            "g": step.get("g", 3.14),
                            "mode": "104",
                        },
                    )
            self._refresh_sequence_reachability()
            self.sequence_log.append(f"Loaded sequence from {self._sequence_file}")
        except Exception as exc:
            QMessageBox.warning(self, "Load sequence error", str(exc))

    def _on_load_test_sequence(self) -> None:
        self.sequence_table.setRowCount(0)
        row0 = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row0)
        self._set_sequence_row(
            row0,
            {"type": "home", "mode": "100", "x": "", "y": "", "z": "", "t": "", "r": "", "g": ""},
        )
        row1 = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row1)
        x, y, z, t, r, g, _spd = self._xyz_values()
        self._set_sequence_row(
            row1,
            {
                "type": "target",
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "t": round(t, 3),
                "r": round(r, 3),
                "g": round(g, 3),
                "mode": "104",
            },
        )
        row2 = self.sequence_table.rowCount()
        self.sequence_table.insertRow(row2)
        self._set_sequence_row(
            row2,
            {"type": "home", "mode": "100", "x": "", "y": "", "z": "", "t": "", "r": "", "g": ""},
        )
        self._refresh_sequence_reachability()
        self.sequence_log.append("Loaded test sequence: Home -> Target -> Home")

    def _on_stop_sequence(self) -> None:
        self._sequence_stop_requested = True
        self.sequence_log.append("Stop requested.")

    def _append_history(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        r: float,
        g: float,
        mode: str,
    ) -> None:
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "t": float(t),
            "r": float(r),
            "g": float(g),
            "mode": mode,
            "result": "unknown",
            "key": self._point_key(x, y, z),
        }
        self._tested_history.append(row)
        self._last_target_idx = len(self._tested_history) - 1
        self._save_history()
        self._refresh_history_table()
        self._update_reachability_status()

    def _mark_last_target(self, result: str) -> None:
        if self._last_target_idx is None:
            QMessageBox.information(self, "No target", "No target has been sent yet.")
            return
        if not (0 <= self._last_target_idx < len(self._tested_history)):
            QMessageBox.warning(self, "History error", "Invalid history pointer.")
            return
        self._tested_history[self._last_target_idx]["result"] = result
        self._save_history()
        self._refresh_history_table()
        self._update_reachability_status()

    def _update_status_view(self, data: Dict[str, Any]) -> None:
        pose = f"X={data.get('x')}  Y={data.get('y')}  Z={data.get('z')}"
        joints_rad = {
            "b": data.get("b"),
            "s": data.get("s"),
            "e": data.get("e"),
            "t": data.get("t"),
            "r": data.get("r"),
            "g": data.get("g"),
        }
        joints_deg = {
            key: (None if val is None else round(math.degrees(float(val)), 2))
            for key, val in joints_rad.items()
        }
        loads = (
            f"tB={data.get('tB')} tS={data.get('tS')} tE={data.get('tE')} "
            f"tT={data.get('tT')} tR={data.get('tR')} tG={data.get('tG')}"
        )
        voltage = f"v={data.get('v')}"
        text = "\n".join(
            [
                "Pose:",
                f"  {pose}",
                "",
                "Joints (rad):",
                f"  {joints_rad}",
                "Joints (deg):",
                f"  {joints_deg}",
                "",
                "System:",
                f"  {voltage}",
                f"  {loads}",
                "",
                f"Raw: {data}",
            ]
        )
        self.status_view.setPlainText(text)

    # -------------------- actions --------------------
    def _on_test_connection(self) -> None:
        self._sync_ip()
        try:
            url, status = self._client.get_status()
            self._log_command(url, str(status))
            self._update_status_view(status)
            QMessageBox.information(self, "Connection", "Connection OK")
        except RoArmClientError as exc:
            self._log_error(str(exc))
            QMessageBox.warning(self, "Connection failed", str(exc))

    def _on_refresh_status(self) -> None:
        self._sync_ip()
        try:
            url, status = self._client.get_status()
            self._log_command(url, str(status))
            self._update_status_view(status)
        except RoArmClientError as exc:
            self._log_error(str(exc))
            QMessageBox.warning(self, "Status Error", str(exc))

    def _on_home(self) -> None:
        self._run_action(self._client.home)

    def _on_torque(self, enabled: bool) -> None:
        self._run_action(lambda: self._client.torque(enabled))

    def _on_open_gripper(self) -> None:
        self._run_action(self._client.gripper_open)

    def _on_close_gripper(self) -> None:
        self._run_action(self._client.gripper_close)

    def _on_check_reachability(self) -> None:
        self._update_reachability_status()

    def _on_move_xyz(self) -> None:
        x, y, z, t, r, g, spd = self._xyz_values()
        self._update_reachability_status()
        ok = self._run_action(lambda: self._client.move_xyz(x, y, z, t, r, g, spd))
        if ok:
            self._append_history(x, y, z, t, r, g, mode="104")

    def _on_send_joint(self, joint_id: int) -> None:
        rad = self.joint_rad_inputs[joint_id].value()
        self._run_action(lambda: self._client.joint_control(joint=joint_id, rad=rad, spd=0.0, acc=10.0))

    def _on_move_axis(self) -> None:
        axis = int(self.axis_combo.currentData())
        pos = self.axis_pos.value()
        self._run_action(lambda: self._client.axis_control(axis=axis, pos=pos, spd=0.25))

    def _on_move_xyz_direct(self) -> None:
        x = self.dir_x.value()
        y = self.dir_y.value()
        z = self.dir_z.value()
        t = self.dir_t.value()
        r = self.dir_r.value()
        g = self.dir_g.value()
        ok = self._run_action(lambda: self._client.move_xyz_direct(x, y, z, t, r, g))
        if ok:
            self._append_history(x, y, z, t, r, g, mode="1041")

    def _on_run_sequence(self) -> None:
        self._sync_ip()
        self._sequence_stop_requested = False
        self._clear_sequence_status_columns()
        self._refresh_sequence_reachability()
        try:
            sequence = self._rows_to_sequence()
        except Exception as exc:
            QMessageBox.warning(self, "Sequence error", str(exc))
            return
        if not sequence:
            QMessageBox.information(self, "Sequence", "No steps in sequence.")
            return

        self.sequence_log.append("Sequence run started.")

        def on_step_start(idx: int, step: Dict[str, Any], meta: Dict[str, Any]) -> None:
            self.sequence_table.setItem(idx, 9, QTableWidgetItem("running"))
            self.sequence_log.append(
                f"Step {idx + 1} START type={meta.get('step_type')} json={meta.get('json')} "
                f"timeout={meta.get('timeout_sec')} start={meta.get('start_time')}"
            )
            QApplication.processEvents()

        def on_step_done(idx: int, step: Dict[str, Any], url: str, response: str, meta: Dict[str, Any]) -> None:
            self.sequence_table.setItem(idx, 9, QTableWidgetItem("done"))
            self._log_command(url, response)
            self.sequence_log.append(
                f"Step {idx + 1} FINISH type={meta.get('step_type')} finish={meta.get('finish_time')} "
                f"duration={meta.get('duration_sec')}s"
            )
            if step.get("type") == "target":
                self._append_history(
                    x=float(step["x"]),
                    y=float(step["y"]),
                    z=float(step["z"]),
                    t=float(step.get("t", 0.0)),
                    r=float(step.get("r", 0.0)),
                    g=float(step.get("g", 3.14)),
                    mode="104",
                )
            QApplication.processEvents()

        def on_step_error(idx: int, step: Dict[str, Any], message: str, meta: Dict[str, Any]) -> None:
            self.sequence_table.setItem(idx, 9, QTableWidgetItem("error"))
            self._log_error(f"Step {idx + 1} failed: {message}")
            self.sequence_log.append(
                f"Step {idx + 1} ERROR type={meta.get('step_type')} json={meta.get('json')} "
                f"timeout={meta.get('timeout_sec')} start={meta.get('start_time')} "
                f"finish={meta.get('finish_time')} duration={meta.get('duration_sec')}s err={message}"
            )
            QApplication.processEvents()

        def should_stop() -> bool:
            QApplication.processEvents()
            return self._sequence_stop_requested

        run_result = self._sequence_manager.run(
            sequence=sequence,
            on_step_start=on_step_start,
            on_step_done=on_step_done,
            on_step_error=on_step_error,
            should_stop=should_stop,
        )
        if run_result == "stopped":
            self.sequence_log.append("Sequence stopped by user.")
        elif run_result == "done":
            self.sequence_log.append("Sequence completed.")
        else:
            self.sequence_log.append("Sequence finished with errors.")


def main() -> None:
    app = QApplication([])
    window = RoArmTestWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

