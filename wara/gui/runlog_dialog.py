"""Pop-up for the API / Neutrons "Log run" button: shows the metadata and
statistics that will be recorded for the loaded file and asks the user for a
description. Writing the entry is done by :func:`wara.runlog.log_run`."""
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

from wara import runlog

from . import theme as T
from .widgets import header


class RunLogDialog(QDialog):
    """Ask for a run description; the recorded fields are shown read-only."""

    def __init__(self, source, metadata, stats, log_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Log run — {source}")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(460)
        self.resize(520, 520)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)
        lay.addWidget(header("DESCRIPTION"))
        self.ed_desc = QPlainTextEdit()
        self.ed_desc.setPlaceholderText(
            "What is this file? Source, geometry, shielding, settings, "
            "anything you want to remember…")
        self.ed_desc.setToolTip(
            "Free-text description saved with the entry (several lines allowed)")
        lay.addWidget(self.ed_desc, 2)

        lay.addWidget(header("RECORDED WITH THE ENTRY"))
        preview = runlog.format_entry(
            runlog.count_entries(runlog.current_log_file(log_dir)) + 1,
            "(your description)", source, metadata, stats)
        self.txt_preview = QPlainTextEdit(preview)
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setStyleSheet(f"font-family:{T.MONO_FAMILY};")
        self.txt_preview.setToolTip(
            "Metadata and statistics captured from the loaded file")
        lay.addWidget(self.txt_preview, 3)

        target = runlog.current_log_file(log_dir)
        self.lbl_target = QLabel(
            f"Saved to {target}  (up to {runlog.MAX_ENTRIES} entries per file)")
        self.lbl_target.setObjectName("stat_key"); self.lbl_target.setWordWrap(True)
        lay.addWidget(self.lbl_target)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.ed_desc.setFocus()

    def description(self):
        return self.ed_desc.toPlainText().strip()
