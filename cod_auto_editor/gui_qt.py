# cod_auto_editor/gui_qt.py
from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from .api import run_pipeline, finalize_render_with_keeps

APP_TITLE = "COD Auto Editor"

# -------------------- Custom reactive checkbox style --------------------

class CheckStyle(QtWidgets.QProxyStyle):
    """
    Draws larger, reactive checkbox indicators with a black checkmark.
    Applies to both QCheckBox widgets and item-view checkboxes in QTableWidget.
    """
    def __init__(self, base: Optional[QtWidgets.QStyle] = None):
        super().__init__(base or QtWidgets.QStyleFactory.create("Fusion"))

    def pixelMetric(self, metric, option=None, widget=None):
        if metric in (QtWidgets.QStyle.PM_IndicatorWidth, QtWidgets.QStyle.PM_IndicatorHeight):
            return 18  # bigger, easier to see
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element, option, painter, widget=None):
        # Cover both standalone checkbox and item-view checkbox indicators
        if element in (
            QtWidgets.QStyle.PE_IndicatorCheckBox,
            QtWidgets.QStyle.PE_IndicatorItemViewItemCheck,
        ):
            s = option.state
            enabled = bool(s & QtWidgets.QStyle.State_Enabled)
            checked = bool(s & QtWidgets.QStyle.State_On)
            hovered = bool(s & QtWidgets.QStyle.State_MouseOver)
            sunken  = bool(s & QtWidgets.QStyle.State_Sunken)
            has_focus = bool(s & QtWidgets.QStyle.State_HasFocus)

            # Colors tuned for your dark palette
            col_border_idle = QtGui.QColor("#3a3b42")
            col_border_hover = QtGui.QColor("#6C5CE7")
            col_border_focus = QtGui.QColor("#8B7BFF")
            col_bg_unchecked = QtGui.QColor("#0F1013")
            col_bg_checked = QtGui.QColor("#FFFFFF")
            col_bg_hover = QtGui.QColor("#171820")
            col_check = QtGui.QColor("#000000")  # BLACK checkmark

            # Choose border color by state
            border = col_border_idle
            if has_focus:
                border = col_border_focus
            elif hovered:
                border = col_border_hover

            # Choose background by state
            if checked:
                bg = col_bg_checked
                if hovered:
                    bg = QtGui.QColor("#F2F2F7")
                if sunken:
                    bg = QtGui.QColor("#E6E6EE")
            else:
                bg = col_bg_unchecked
                if hovered:
                    bg = col_bg_hover

            # Slightly dim if disabled
            if not enabled:
                border = QtGui.QColor(border)
                border.setAlpha(120)
                bg = QtGui.QColor(bg)
                bg.setAlpha(160)

            r = option.rect.adjusted(2, 2, -2, -2)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            # Fill + border
            painter.setBrush(bg)
            pen_w = 2 if (hovered or has_focus) else 1.2
            painter.setPen(QtGui.QPen(border, pen_w))
            painter.drawRoundedRect(r, 4, 4)

            # Black checkmark
            if checked:
                # Two simple strokes ✓
                p = QtGui.QPainterPath()
                p.moveTo(r.left() + r.width() * 0.22, r.center().y())
                p.lineTo(r.left() + r.width() * 0.45, r.bottom() - r.height() * 0.22)
                p.lineTo(r.right() - r.width() * 0.20, r.top() + r.height() * 0.28)
                painter.setPen(QtGui.QPen(col_check, 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPath(p)

            painter.restore()
            return None  # explicit

        # Fallback to default for everything else
        return super().drawPrimitive(element, option, painter, widget)

# -------------------- Async workers --------------------

class Worker(QtCore.QObject):
    progressed = QtCore.pyqtSignal(float)
    logged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    frameReady = QtCore.pyqtSignal(QtGui.QImage, float)  # (image, t_sec)

    def __init__(self, input_path: str, config_path: str, cancel_event: threading.Event, enable_preview: bool, review_only: bool):
        super().__init__()
        self.input_path = input_path
        self.config_path = config_path
        self.cancel_event = cancel_event
        self.enable_preview = enable_preview
        self.review_only = review_only

    @QtCore.pyqtSlot()
    def run(self):
        try:
            def _log(msg: str): self.logged.emit(str(msg))
            def _prog(p: float): self.progressed.emit(max(0.0, min(1.0, float(p))))
            def _cancelled(): return self.cancel_event.is_set()

            def _preview(frame_bgr, t_sec: float):
                if not self.enable_preview:
                    return
                try:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
                    self.frameReady.emit(qimg, float(t_sec))
                except Exception as e:
                    self.logged.emit(f"[Preview] error: {e}")

            result = run_pipeline(
                self.input_path,
                self.config_path,
                log=_log,
                progress=_prog,
                is_cancelled=_cancelled,
                frame_preview=_preview,
                review_only=self.review_only
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

class FinalizeWorker(QtCore.QObject):
    progressed = QtCore.pyqtSignal(float)
    logged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, input_path: str, config_path: str, windows: List[Tuple[float, float]]):
        super().__init__()
        self.input_path = input_path
        self.config_path = config_path
        self.windows = windows

    @QtCore.pyqtSlot()
    def run(self):
        try:
            def _log(msg: str): self.logged.emit(str(msg))
            def _prog(p: float): self.progressed.emit(max(0.0, min(1.0, float(p))))
            result = finalize_render_with_keeps(
                self.input_path, self.config_path, self.windows, log=_log, progress=_prog
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

class ClipPlayer(QtCore.QObject):
    frameReady = QtCore.pyqtSignal(QtGui.QImage, float)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, video_path: str, start_t: float, end_t: float, fps_hint: int = 30):
        super().__init__()
        self.video_path = video_path
        self.start_t = float(start_t)
        self.end_t = float(end_t)
        self._stop_ev = threading.Event()  # thread-safe stop flag
        self.fps_hint = max(1, int(fps_hint))

    @QtCore.pyqtSlot()
    def play(self):
        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                self.error.emit(f"[Player] Could not open video: {self.video_path}")
                return
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_t * 1000.0)
            delay_ms = int(1000 / self.fps_hint)

            while not self._stop_ev.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = (t_ms / 1000.0) if t_ms else 0.0
                if t > self.end_t + 0.001:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
                self.frameReady.emit(qimg, t)
                QtCore.QThread.msleep(delay_ms)
        except Exception as e:
            import traceback
            self.error.emit(f"[Player] {e}\n\n{traceback.format_exc()}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self.finished.emit()

    def stop(self):
        self._stop_ev.set()

# -------------------- GUI --------------------

class Header(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        title = QtWidgets.QLabel(APP_TITLE); title.setObjectName("AppTitle")
        subtitle = QtWidgets.QLabel(
            "Automate your CoD edits: intro trim, filler/silence, dedupe, hitmarkers, HDR/SDR render."
        ); subtitle.setWordWrap(True); subtitle.setObjectName("Subtitle")
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(title); lay.addWidget(subtitle)

class PathPicker(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()
    def __init__(self, label: str, mode: str = "file", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.mode = mode
        self.edit = QtWidgets.QLineEdit(); self.edit.setPlaceholderText(f"Choose {label.lower()}…")
        self.btn = QtWidgets.QPushButton("Browse"); self.btn.setCursor(QtCore.Qt.PointingHandCursor)
        row = QtWidgets.QHBoxLayout(self)
        row.addWidget(QtWidgets.QLabel(label)); row.addWidget(self.edit, 1); row.addWidget(self.btn)
        self.btn.clicked.connect(self._on_browse); self.edit.textChanged.connect(self.changed.emit)

    def value(self) -> str: return self.edit.text().strip()
    def set_value(self, path: str): self.edit.setText(path or "")

    def _on_browse(self):
        if self.mode == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file")
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
        if path: self.set_value(path)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.setMinimumSize(1200, 760)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[Worker] = None
        self._final_thread: Optional[QtCore.QThread] = None
        self._final_worker: Optional[FinalizeWorker] = None
        self._cancel_event = threading.Event()
        self._latest_output: Optional[str] = None
        self._last_qimage: Optional[QtGui.QImage] = None

        self._player_thread: Optional[QtCore.QThread] = None
        self._player: Optional[ClipPlayer] = None

        self._review_candidates: List[Tuple[float,float]] = []
        self._review_video_path: Optional[str] = None

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(18,18,18,18); root.setSpacing(14)
        root.addWidget(Header())

        card = QtWidgets.QFrame(); card.setObjectName("Card"); root.addWidget(card, 1)
        grid = QtWidgets.QGridLayout(card); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(12); grid.setContentsMargins(18,18,18,18)

        self.pick_input = PathPicker("Input video (.mp4)", "file")
        self.pick_config = PathPicker("Config (config.yml)", "file")
        grid.addWidget(self.pick_input, 0, 0, 1, 2)
        grid.addWidget(self.pick_config, 1, 0, 1, 2)

        self.chk_preview = QtWidgets.QCheckBox("Live frame preview (during scan)"); self.chk_preview.setChecked(True)
        self.chk_review  = QtWidgets.QCheckBox("Review clips before render"); self.chk_review.setChecked(True)

        self.btn_run = QtWidgets.QPushButton("Scan Clips"); self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_cancel = QtWidgets.QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        self.btn_cancel.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_reveal = QtWidgets.QPushButton("Open Output Folder"); self.btn_reveal.setEnabled(False)
        self.btn_reveal.setCursor(QtCore.Qt.PointingHandCursor)

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.chk_preview); actions.addWidget(self.chk_review)
        actions.addStretch(1)
        actions.addWidget(self.btn_reveal); actions.addSpacing(12); actions.addWidget(self.btn_cancel); actions.addSpacing(12); actions.addWidget(self.btn_run)
        grid.addLayout(actions, 2, 0, 1, 2)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setTextVisible(False)
        grid.addWidget(self.progress, 3, 0, 1, 2)

        # --- Preview panel ---
        self.preview = QtWidgets.QLabel("Live/Clip preview will appear here…")
        self.preview.setObjectName("Preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        grid.addWidget(self.preview, 4, 0, 1, 2)

        # --- Review panel (hidden until scan completes) ---
        self.review_box = QtWidgets.QGroupBox("Clip Review"); self.review_box.setVisible(False)
        rv = QtWidgets.QVBoxLayout(self.review_box)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Keep", "Start (s)", "End (s)", "Duration (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMouseTracking(True)  # enable hover feedback on item checkboxes
        self.table.viewport().setAttribute(QtCore.Qt.WA_Hover, True)
        rv.addWidget(self.table)

        ctl = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("Prev"); self.btn_prev.setEnabled(False)
        self.btn_play = QtWidgets.QPushButton("Play"); self.btn_play.setEnabled(False)
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_next = QtWidgets.QPushButton("Next"); self.btn_next.setEnabled(False)
        ctl.addWidget(self.btn_prev); ctl.addWidget(self.btn_play); ctl.addWidget(self.btn_stop); ctl.addWidget(self.btn_next)
        ctl.addStretch(1)
        self.btn_toggle_all = QtWidgets.QPushButton("Toggle All"); self.btn_toggle_all.setEnabled(False)
        self.btn_finalize = QtWidgets.QPushButton("Finalize Render"); self.btn_finalize.setEnabled(False)
        ctl.addWidget(self.btn_toggle_all); ctl.addWidget(self.btn_finalize)
        rv.addLayout(ctl)

        grid.addWidget(self.review_box, 5, 0, 1, 2)

        # Logs
        self.logs = QtWidgets.QPlainTextEdit(); self.logs.setReadOnly(True); self.logs.setObjectName("LogView")
        self.logs.setPlaceholderText("Logs will appear here…"); grid.addWidget(self.logs, 6, 0, 1, 2)

        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(QtWidgets.QLabel("Tip: HDR preserve uses FFmpeg (10-bit HEVC HLG)."))
        foot.addStretch(1); link = QtWidgets.QLabel('<a href="https://ffmpeg.org/">FFmpeg</a>'); link.setOpenExternalLinks(True); foot.addWidget(link)
        root.addLayout(foot)

        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_reveal.clicked.connect(self._on_reveal)

        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_toggle_all.clicked.connect(self._on_toggle_all)
        self.btn_finalize.clicked.connect(self._on_finalize)

        self.table.itemSelectionChanged.connect(self._on_row_change)

        for guess in ["input/match1.mp4"]:
            if Path(guess).exists(): self.pick_input.set_value(guess); break
        for guess in ["config.yml", "config.yaml"]:
            if Path(guess).exists(): self.pick_config.set_value(guess); break

        self._apply_dark_theme()

        # Ensure hover is active on header checkboxes too
        self.chk_preview.setAttribute(QtCore.Qt.WA_Hover, True)
        self.chk_review.setAttribute(QtCore.Qt.WA_Hover, True)

    # ---------- Actions ----------

    def _on_run(self):
        video = self.pick_input.value(); cfg = self.pick_config.value()
        if not video or not Path(video).exists():
            QtWidgets.QMessageBox.warning(self, "Missing file", "Please select a valid input video."); return
        if not cfg or not Path(cfg).exists():
            QtWidgets.QMessageBox.warning(self, "Missing config", "Please select a valid config.yml."); return

        self._stop_player_if_any()
        self._review_candidates = []
        self._review_video_path = None
        self._last_qimage = None
        self.preview.setText("Scanning…")
        self.logs.clear(); self.progress.setValue(0); self._latest_output = None
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True); self.btn_reveal.setEnabled(False)

        review = self.chk_review.isChecked()
        self.review_box.setVisible(False)

        self._cancel_event.clear()
        self._thread = QtCore.QThread(self)
        self._worker = Worker(video, cfg, self._cancel_event, enable_preview=self.chk_preview.isChecked(), review_only=review)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(lambda p: self.progress.setValue(int(round(p*100))))
        self._worker.logged.connect(self._append_log)
        self._worker.frameReady.connect(self._on_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_cancel(self):
        self._cancel_event.set()
        self._append_log("[UI] Cancel requested…")
        self._stop_player_if_any()

    def _on_reveal(self):
        out_dir = Path("output").resolve()
        if sys.platform == "darwin": os.system(f'open "{out_dir}"')
        elif os.name == "nt": os.startfile(str(out_dir))
        else: os.system(f'xdg-open "{out_dir}"')

    def _on_finished(self, result: object):
        self.btn_cancel.setEnabled(False); self.btn_reveal.setEnabled(True)
        if isinstance(result, dict) and result.get("review_mode"):
            cand = result.get("candidates", []) or []
            self._review_candidates = [(float(s), float(e)) for (s, e) in cand if float(e) > float(s)]
            self._review_video_path = str(result.get("input_path"))
            self._append_log(f"\n🔎 Scan complete — {len(self._review_candidates)} candidate clip(s).")
            self._populate_review_table()
            self.review_box.setVisible(True)
            self.btn_run.setEnabled(True)
        else:
            self.btn_run.setEnabled(True)
            self._append_log("\n✅ Completed.")
            if isinstance(result, dict) and result.get("output_path"):
                self._append_log(f"Output: {result['output_path']}")

    def _on_failed(self, err: str):
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False); self.btn_reveal.setEnabled(True)
        self._append_log("\n❌ Failed."); self._append_log(err)

    def _cleanup_thread(self):
        if self._worker: self._worker.deleteLater()
        self._worker = None
        if self._thread: self._thread.deleteLater()
        self._thread = None

    # ---------- Review UI helpers ----------

    def _populate_review_table(self):
        self.table.setRowCount(0)
        for (s, e) in self._review_candidates:
            row = self.table.rowCount()
            self.table.insertRow(row)

            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
            chk.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(row, 0, chk)

            def _fmt(x): return f"{x:.2f}"
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(_fmt(s)))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(_fmt(e)))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(_fmt(e - s)))

        has = self.table.rowCount() > 0
        self.btn_prev.setEnabled(has)
        self.btn_play.setEnabled(has)
        self.btn_stop.setEnabled(has)
        self.btn_next.setEnabled(has)
        self.btn_toggle_all.setEnabled(has)
        self.btn_finalize.setEnabled(has)
        if has:
            self.table.selectRow(0)

    def _current_row(self) -> int:
        idxs = self.table.selectionModel().selectedRows()
        return idxs[0].row() if idxs else -1

    def _on_row_change(self):
        self._stop_player_if_any()

    def _on_prev(self):
        r = self._current_row()
        if r > 0: self.table.selectRow(r - 1)

    def _on_next(self):
        r = self._current_row()
        if r >= 0 and r + 1 < self.table.rowCount():
            self.table.selectRow(r + 1)

    def _on_play(self):
        r = self._current_row()
        if r < 0 or not self._review_video_path:
            return
        s_item = self.table.item(r, 1)
        e_item = self.table.item(r, 2)
        if s_item is None or e_item is None:
            return  # missing cells
        try:
            s = float(s_item.text())
            e = float(e_item.text())
        except (TypeError, ValueError):
            return  # malformed cells
        self._stop_player_if_any()
        self._player_thread = QtCore.QThread(self)
        self._player = ClipPlayer(self._review_video_path, s, e, fps_hint=30)
        self._player.moveToThread(self._player_thread)
        self._player_thread.started.connect(self._player.play)
        self._player.frameReady.connect(self._on_frame)
        self._player.error.connect(self._append_log)
        self._player.finished.connect(self._player_thread.quit)
        self._player_thread.finished.connect(self._cleanup_player)
        self._player_thread.start()

    def _on_stop(self):
        self._stop_player_if_any()

    def _stop_player_if_any(self):
        if self._player:
            self._player.stop()
        # cleanup happens when the thread finishes

    def _cleanup_player(self):
        if self._player: self._player.deleteLater()
        self._player = None
        if self._player_thread: self._player_thread.deleteLater()
        self._player_thread = None

    def _on_toggle_all(self):
        # Determine if all are currently checked (treat missing items as unchecked)
        all_checked = True
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is None or item.checkState() != QtCore.Qt.Checked:
                all_checked = False
                break
        target = QtCore.Qt.Unchecked if all_checked else QtCore.Qt.Checked
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                item.setCheckState(target)

    def _on_finalize(self):
        if not self._review_video_path:
            QtWidgets.QMessageBox.information(self, "Nothing to render", "No scanned video available."); return
        cfg = self.pick_config.value()
        if not cfg or not Path(cfg).exists():
            QtWidgets.QMessageBox.warning(self, "Missing config", "Please select a valid config.yml."); return

        approved: List[Tuple[float, float]] = []
        for r in range(self.table.rowCount()):
            item0 = self.table.item(r, 0)
            item1 = self.table.item(r, 1)
            item2 = self.table.item(r, 2)
            if item0 is None or item0.checkState() != QtCore.Qt.Checked:
                continue
            if item1 is None or item2 is None:
                continue
            try:
                s = float(item1.text())
                e = float(item2.text())
                if e > s:
                    approved.append((s, e))
            except (TypeError, ValueError):
                continue  # skip malformed rows

        if not approved:
            QtWidgets.QMessageBox.information(self, "No clips selected", "Select at least one clip to render."); return

        self._append_log(f"[Finalize] Rendering {len(approved)} approved clip(s)…")
        self.btn_finalize.setEnabled(False); self.btn_play.setEnabled(False); self.btn_stop.setEnabled(False)
        self.btn_prev.setEnabled(False); self.btn_next.setEnabled(False)

        self._final_thread = QtCore.QThread(self)
        self._final_worker = FinalizeWorker(self._review_video_path, cfg, approved)
        self._final_worker.moveToThread(self._final_thread)
        self._final_thread.started.connect(self._final_worker.run)
        self._final_worker.progressed.connect(lambda p: self.progress.setValue(int(round(p*100))))
        self._final_worker.logged.connect(self._append_log)
        self._final_worker.finished.connect(self._on_finalize_done)
        self._final_worker.failed.connect(self._on_failed)
        self._final_worker.finished.connect(self._final_thread.quit); self._final_worker.failed.connect(self._final_thread.quit)
        self._final_thread.finished.connect(self._cleanup_finalize_thread)
        self._final_thread.start()

    def _on_finalize_done(self, result: object):
        self._append_log("\n✅ Finalize complete.")
        if isinstance(result, dict) and result.get("output_path"):
            self._append_log(f"Output: {result['output_path']}")
        self.btn_finalize.setEnabled(True)
        self.btn_play.setEnabled(True); self.btn_stop.setEnabled(True)
        self.btn_prev.setEnabled(True); self.btn_next.setEnabled(True)
        self.btn_reveal.setEnabled(True)

    def _cleanup_finalize_thread(self):
        if self._final_worker: self._final_worker.deleteLater()
        self._final_worker = None
        if self._final_thread: self._final_thread.deleteLater()
        self._final_thread = None

    # ---------- Rendering helpers ----------

    def _on_frame(self, qimg: QtGui.QImage, t_sec: float):
        # Only display frames if a player is active OR preview is enabled during scanning.
        if not (self._player or self.chk_preview.isChecked()):
            return
        self._last_qimage = qimg
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self.preview.setToolTip(f"t = {t_sec:.2f}s")

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        if self._last_qimage is not None:
            pix = QtGui.QPixmap.fromImage(self._last_qimage)
            scaled = pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.preview.setPixmap(scaled)

    def _append_log(self, text: str):
        self.logs.appendPlainText(text)
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def _apply_dark_theme(self):
        # Keep current style (our CheckStyle(Fusion)); don't overwrite it here.
        palette = QtGui.QPalette()
        bg = QtGui.QColor(21,22,26); panel = QtGui.QColor(15,16,19); text = QtGui.QColor(230,230,235); acc = QtGui.QColor(108,92,231)
        palette.setColor(QtGui.QPalette.Window, bg); palette.setColor(QtGui.QPalette.WindowText, text)
        palette.setColor(QtGui.QPalette.Base, panel); palette.setColor(QtGui.QPalette.AlternateBase, bg)
        palette.setColor(QtGui.QPalette.Text, text); palette.setColor(QtGui.QPalette.Button, bg)
        palette.setColor(QtGui.QPalette.ButtonText, text); palette.setColor(QtGui.QPalette.Link, acc)
        palette.setColor(QtGui.QPalette.Highlight, acc); palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            #AppTitle { font-size: 28px; font-weight: 700; letter-spacing: 0.3px; color: #ffffff}
            #Subtitle { color: #ACACB5; margin-top: 4px; }
            #Card { padding: 8px 10px; background: #15161A; border: 1px solid #2A2B31; border-radius: 16px; }
            QLabel { font-size: 14px; color: #ffffff}
            QLineEdit { padding: 8px 10px; border-radius: 10px; border: 1px solid #2A2B31; background: #0F1013; color: #ffffff }
            QLineEdit:focus { border: 1px solid #6C5CE7; }
            QPushButton { padding: 8px 14px; border-radius: 10px; font-weight: 600; background: #1B1C21; color: #E6E6EB; border: 1px solid #2A2B31; }
            QPushButton:hover { background: #22232A; }
            QPushButton#PrimaryButton { padding: 8px 10px; background: #6C5CE7; color: white; border: none; }
            QPushButton#PrimaryButton:hover { padding: 8px 10px; background: #7B6CFA; }
            /* Keep checkbox label text pure white but let our style handle the indicator visuals */
            QCheckBox { color: #ffffff }
            QProgressBar { min-height: 8px; background: #0f0f12; border-radius: 6px; }
            QProgressBar::chunk { border-radius: 6px; background-color: #6C5CE7; }
            #LogView { background: #0F1013; border: 1px solid #2A2B31; border-radius: 12px;
                       font-family: ui-monospace, Menlo, Monaco, Consolas, "Courier New", monospace;
                       font-size: 12px; padding: 10px; color: #E6E6EB; }
            #Preview { background: #0F1013; border: 1px dashed #2A2B31; border-radius: 12px; color: #ffffff }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #ffffff}
            QGroupBox { border: 1px solid #2A2B31; border-radius: 12px; margin-top: 10px; padding-top: 12px; margin-left: 8px; }
            QTableWidget { background: #0F1013; border: 1px solid #2A2B31; border-radius: 8px; color: #ffffff; }
        """)

def qt_main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    # Apply our reactive checkbox style on top of Fusion globally
    base = QtWidgets.QStyleFactory.create("Fusion")
    app.setStyle(CheckStyle(base))

    win = MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    qt_main()
