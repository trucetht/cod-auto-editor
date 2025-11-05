# cod_auto_editor/gui_qt.py
from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from .api import run_pipeline

APP_TITLE = "COD Auto Editor"

class Worker(QtCore.QObject):
    progressed = QtCore.pyqtSignal(float)
    logged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, input_path: str, config_path: str, cancel_event: threading.Event):
        super().__init__()
        self.input_path = input_path
        self.config_path = config_path
        self.cancel_event = cancel_event

    @QtCore.pyqtSlot()
    def run(self):
        try:
            def _log(msg: str): self.logged.emit(str(msg))
            def _prog(p: float): self.progressed.emit(max(0.0, min(1.0, float(p))))
            def _cancelled(): return self.cancel_event.is_set()
            result = run_pipeline(self.input_path, self.config_path, log=_log, progress=_prog, is_cancelled=_cancelled)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

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
        self.setWindowTitle(APP_TITLE); self.setMinimumSize(1040, 680)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[Worker] = None
        self._cancel_event = threading.Event(); self._latest_output: Optional[str] = None

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(18,18,18,18); root.setSpacing(14)
        root.addWidget(Header())

        card = QtWidgets.QFrame(); card.setObjectName("Card"); root.addWidget(card, 1)
        grid = QtWidgets.QGridLayout(card); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(12); grid.setContentsMargins(18,18,18,18)

        self.pick_input = PathPicker("Input video (.mp4)", "file")
        self.pick_config = PathPicker("Config (config.yml)", "file")
        grid.addWidget(self.pick_input, 0, 0, 1, 2)
        grid.addWidget(self.pick_config, 1, 0, 1, 2)

        self.btn_run = QtWidgets.QPushButton("Run Edit"); self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_cancel = QtWidgets.QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        self.btn_cancel.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_reveal = QtWidgets.QPushButton("Open Output Folder"); self.btn_reveal.setEnabled(False)
        self.btn_reveal.setCursor(QtCore.Qt.PointingHandCursor)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1); actions.addWidget(self.btn_reveal); actions.addWidget(self.btn_cancel); actions.addWidget(self.btn_run)
        grid.addLayout(actions, 2, 0, 1, 2)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setTextVisible(False)
        grid.addWidget(self.progress, 3, 0, 1, 2)

        self.logs = QtWidgets.QPlainTextEdit(); self.logs.setReadOnly(True); self.logs.setObjectName("LogView")
        self.logs.setPlaceholderText("Logs will appear here…"); grid.addWidget(self.logs, 4, 0, 1, 2)

        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(QtWidgets.QLabel("Tip: HDR preserve uses FFmpeg (10-bit HEVC HLG)."))
        foot.addStretch(1); link = QtWidgets.QLabel('<a href="https://ffmpeg.org/">FFmpeg</a>'); link.setOpenExternalLinks(True); foot.addWidget(link)
        root.addLayout(foot)

        self.btn_run.clicked.connect(self._on_run); self.btn_cancel.clicked.connect(self._on_cancel); self.btn_reveal.clicked.connect(self._on_reveal)

        for guess in ["input/match1.mp4"]:
            if Path(guess).exists(): self.pick_input.set_value(guess); break
        for guess in ["config.yml", "config.yaml"]:
            if Path(guess).exists(): self.pick_config.set_value(guess); break

        self._apply_dark_theme()

    def _on_run(self):
        video = self.pick_input.value(); cfg = self.pick_config.value()
        if not video or not Path(video).exists():
            QtWidgets.QMessageBox.warning(self, "Missing file", "Please select a valid input video."); return
        if not cfg or not Path(cfg).exists():
            QtWidgets.QMessageBox.warning(self, "Missing config", "Please select a valid config.yml."); return

        self.logs.clear(); self.progress.setValue(0); self._latest_output = None
        self.btn_run.setEnabled(False); self.btn_cancel.setEnabled(True); self.btn_reveal.setEnabled(False)
        self._cancel_event.clear()

        self._thread = QtCore.QThread(self)
        self._worker = Worker(video, cfg, self._cancel_event); self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(lambda p: self.progress.setValue(int(round(p*100))))
        self._worker.logged.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit); self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_cancel(self): self._cancel_event.set(); self._append_log("[UI] Cancel requested…")
    def _on_reveal(self):
        out_dir = Path("output").resolve()
        if sys.platform == "darwin": os.system(f'open "{out_dir}"')
        elif os.name == "nt": os.startfile(str(out_dir))
        else: os.system(f'xdg-open "{out_dir}"')

    def _on_finished(self, result: object):
        self.btn_run.setEnabled(True); self.btn_cancel.setEnabled(False); self.btn_reveal.setEnabled(True)
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

    def _append_log(self, text: str):
        self.logs.appendPlainText(text)
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

    def _apply_dark_theme(self):
        QtWidgets.QApplication.setStyle("Fusion")
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
            #Card { background: #15161A; border: 1px solid #2A2B31; border-radius: 16px; }
            QLabel { font-size: 14px; color: #ffffff}
            QLineEdit { padding: 8px 10px; border-radius: 10px; border: 1px solid #2A2B31; background: #0F1013; color: #E6E6EB; }
            QLineEdit:focus { border: 1px solid #6C5CE7; }
            QPushButton { padding: 8px 14px; border-radius: 10px; font-weight: 600; background: #1B1C21; color: #E6E6EB; border: 1px solid #2A2B31; }
            QPushButton:hover { background: #22232A; }
            QPushButton#PrimaryButton { background: #6C5CE7; color: white; border: none; }
            QPushButton#PrimaryButton:hover { background: #7B6CFA; }
            QProgressBar { min-height: 8px; background: #0f0f12; border-radius: 6px; }
            QProgressBar::chunk { border-radius: 6px; background-color: #6C5CE7; }
            #LogView { background: #0F1013; border: 1px solid #2A2B31; border-radius: 12px;
                       font-family: ui-monospace, Menlo, Monaco, Consolas, "Courier New", monospace;
                       font-size: 12px; padding: 10px; color: #E6E6EB; }
        """)

def qt_main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())
