"""Pop-up for the API / Neutrons "Log run" button.

Two tabs: *New entry* asks for a description and previews the metadata and
statistics that will be recorded for the loaded file; *Log entries* reads back
the entries already in the run-log files. A run that is already in the log is
not logged twice: a warning is shown and the user may replace the old entry
instead. Reading and writing the files is done by :mod:`wara.runlog`."""
from datetime import datetime
from html import escape

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from wara import runlog

from . import theme as T
from .widgets import header

# Colours of the rendered entries.
_C_TITLE = T.ACCENT_CYAN       # "ENTRY n"
_C_SOURCE = T.ACCENT_AMBER     # tab the run was logged from
_C_SECTION = T.LOGO_GREEN      # Description / Metadata / Statistics
_C_KEY = T.TEXT_DIM
_C_VALUE = T.TEXT_PRIMARY
_C_YES = T.ACCENT_GREEN        # data folder present
_C_NO = T.ACCENT_RED           # data folder missing / not found


def _value_html(value):
    v = escape(str(value))
    if v.startswith("yes"):
        return f"<span style='color:{_C_YES}'>{v}</span>"
    if v in ("no", "not found"):
        return f"<span style='color:{_C_NO}'>{v}</span>"
    return f"<span style='color:{_C_VALUE}'>{v}</span>"


def _section_html(title, items):
    if not items:
        return ""
    rows = "".join(
        f"<tr><td style='color:{_C_KEY}; padding-right:14px'>{escape(str(k))}</td>"
        f"<td>{_value_html(v)}</td></tr>" for k, v in items.items())
    return (f"<div style='color:{_C_SECTION}; font-weight:700; margin-top:6px'>"
            f"{title}</div><table cellspacing='0' cellpadding='1' "
            f"style='margin-left:12px'>{rows}</table>")


def entry_html(entry, highlight=False):
    """One run-log entry (a dict as returned by :func:`wara.runlog.read_entries`)
    as coloured HTML. *highlight* frames it (used for the duplicate run)."""
    desc = escape(entry.get("description") or "(no description)")
    desc = desc.replace("\n", "<br>")
    border = T.ACCENT_AMBER if highlight else T.BORDER
    return (
        f"<table width='100%' cellspacing='0' cellpadding='8' "
        f"style='border:1px solid {border}; margin-bottom:10px; "
        f"background:{T.BG_PANEL}'><tr><td>"
        f"<span style='color:{_C_TITLE}; font-weight:700; font-size:15px'>"
        f"ENTRY {entry.get('number', '?')}</span>"
        f"<span style='color:{_C_KEY}'>&nbsp;&nbsp;{escape(entry.get('timestamp', ''))}"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;</span>"
        f"<span style='color:{_C_SOURCE}; font-weight:700'>"
        f"{escape(entry.get('source', ''))}</span>"
        f"<div style='color:{_C_SECTION}; font-weight:700; margin-top:6px'>"
        f"Description</div>"
        f"<div style='color:{_C_VALUE}; margin-left:12px'><i>{desc}</i></div>"
        + _section_html("Metadata", entry.get("metadata"))
        + _section_html("Statistics", entry.get("stats"))
        + "</td></tr></table>")


def _browser(tooltip):
    b = QTextBrowser()
    b.setOpenLinks(False)
    b.setStyleSheet(f"font-family:{T.MONO_FAMILY}; font-size:13px;")
    b.setToolTip(tooltip)
    return b


class RunLogDialog(QDialog):
    """Ask for a run description (tab 1) and browse the run log (tab 2).

    With ``metadata=None`` (nothing loaded) the dialog is browse-only: the
    *New entry* tab is disabled and only *Log entries* can be used."""

    def __init__(self, source, metadata=None, stats=None, log_dir=None,
                 parent=None):
        super().__init__(parent)
        self.source, self.metadata, self.stats = source, metadata, stats or {}
        self.log_dir = log_dir
        self.browse_only = metadata is None
        self.setWindowTitle(f"Run log — {source}" if self.browse_only
                            else f"Log run — {source}")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(560)
        self.resize(680, 620)
        # Existing entry for this run, or None: then Save is replaced by Replace.
        self.duplicate = (None if self.browse_only
                          else runlog.find_run(source, metadata, log_dir))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)
        self.tabs = QTabWidget()
        self.tabs.addTab(QWidget() if self.browse_only else self._build_new_tab(),
                         "New entry")
        self.tabs.addTab(self._build_read_tab(), "Log entries")
        self.tabs.setTabToolTip(0, "Load a run first to add an entry to the log"
                                if self.browse_only else
                                "Describe the loaded run and save it to the log")
        self.tabs.setTabToolTip(1, "Read the entries already saved in the run log")
        self._fit_tabbar(self.tabs)
        lay.addWidget(self.tabs)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.btn_save = bb.button(QDialogButtonBox.Save)
        self.btn_save.setToolTip("Append a new entry for this run to the log")
        self.btn_replace = QPushButton("Replace entry")
        self.btn_replace.setToolTip(
            "Overwrite the existing entry for this run with this description, "
            "metadata and statistics (it keeps its entry number)")
        self.btn_replace.setCursor(Qt.PointingHandCursor)
        bb.addButton(self.btn_replace, QDialogButtonBox.ActionRole)
        self.btn_replace.clicked.connect(self._on_replace)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.replace = False                  # set when Replace was clicked
        if self.browse_only:
            self.tabs.setTabEnabled(0, False)
            self.tabs.setCurrentIndex(1)
            self.btn_save.setVisible(False)
            self.btn_replace.setVisible(False)
            bb.button(QDialogButtonBox.Cancel).setText("Close")
            return
        self.btn_save.setEnabled(self.duplicate is None)
        self.btn_replace.setVisible(self.duplicate is not None)
        self.ed_desc.setFocus()

    @staticmethod
    def _fit_tabbar(tabw):
        """Size the tab bar with the bold stylesheet font so labels are not
        clipped (mirrors DiagnosticsDialog._fit_tabbar)."""
        bar = tabw.tabBar()
        f = bar.font()
        f.setPixelSize(14)
        f.setBold(True)
        bar.setFont(f)
        bar.setElideMode(Qt.ElideNone)
        bar.setExpanding(False)
        tabw.setStyleSheet(
            "QTabBar::tab { padding: 7px 24px; min-width: 96px; }"
            f"QTabBar::tab:disabled {{ color: {T.GRID}; background: {T.BG_DARK}; }}")

    # -- tab 1: new entry -------------------------------------------------------
    def _build_new_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 8, 6, 6); lay.setSpacing(6)
        lay.addWidget(header("DESCRIPTION"))
        self.ed_desc = QPlainTextEdit()
        self.ed_desc.setPlaceholderText(
            "What is this file? Source, geometry, shielding, settings, "
            "anything you want to remember…")
        self.ed_desc.setToolTip(
            "Free-text description saved with the entry (several lines allowed)")
        lay.addWidget(self.ed_desc, 2)

        lay.addWidget(header("RECORDED WITH THE ENTRY"))
        target = self._target_file()
        number = (self.duplicate[1]["number"] if self.duplicate is not None
                  else runlog.count_entries(target) + 1)
        preview = {"number": number,
                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005
                   "source": self.source, "description": "(your description)",
                   "metadata": self.metadata, "stats": self.stats}
        self.txt_preview = _browser(
            "Metadata and statistics captured from the loaded file")
        self.txt_preview.setHtml(entry_html(preview))
        lay.addWidget(self.txt_preview, 3)

        # Quiet (no pop-up) note when this run is already in the log.
        self.lbl_duplicate = QLabel()
        self.lbl_duplicate.setWordWrap(True)
        self.lbl_duplicate.setStyleSheet(f"color:{T.ACCENT_AMBER};")
        if self.duplicate is not None:
            path, e = self.duplicate
            self.lbl_duplicate.setText(
                f"⚠ This run is already logged: entry {e['number']} of "
                f"{path.name} ({e['timestamp']}). A second entry will not be "
                f"written — use Replace entry to overwrite it.")
            self.lbl_duplicate.setToolTip(
                "The run is identified by tab + date + run number (PIXIE runs) "
                "or tab + file path (trace files). See the Log entries tab.")
        self.lbl_duplicate.setVisible(self.duplicate is not None)
        lay.addWidget(self.lbl_duplicate)

        self.lbl_target = QLabel(
            f"Saved to {target}  (up to {runlog.MAX_ENTRIES} entries per file)")
        self.lbl_target.setObjectName("stat_key"); self.lbl_target.setWordWrap(True)
        lay.addWidget(self.lbl_target)
        return w

    def _target_file(self):
        if self.duplicate is not None:
            return self.duplicate[0]
        return runlog.current_log_file(self.log_dir)

    # -- tab 2: read the log ----------------------------------------------------
    def _build_read_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 8, 6, 6); lay.setSpacing(6)
        row = QHBoxLayout(); row.setSpacing(8)
        lbl = QLabel("Log file:"); lbl.setObjectName("stat_key")
        row.addWidget(lbl)
        self.cmb_file = QComboBox()
        self.cmb_file.setToolTip("Run-log file to read (newest last)")
        self._files = runlog.log_files(self.log_dir)
        for p in self._files:
            n = runlog.count_entries(p)
            self.cmb_file.addItem(f"{p.name}  ({n} entr{'y' if n == 1 else 'ies'})")
        row.addWidget(self.cmb_file, 1)
        lay.addLayout(row)
        self.txt_log = _browser("Entries of the selected log file, newest first")
        lay.addWidget(self.txt_log, 1)
        self.cmb_file.currentIndexChanged.connect(self._show_file)
        if self._files:
            idx = len(self._files) - 1
            if self.duplicate is not None:
                idx = self._files.index(self.duplicate[0])
            self.cmb_file.setCurrentIndex(idx)
            self._show_file(idx)
        else:
            self.txt_log.setHtml(
                f"<p style='color:{T.TEXT_DIM}'><i>The run log is empty.</i></p>")
        return w

    def _show_file(self, idx):
        if not 0 <= idx < len(self._files):
            return
        path = self._files[idx]
        dup = self.duplicate[1]["number"] if (
            self.duplicate is not None and self.duplicate[0] == path) else None
        entries = runlog.read_entries(path)
        html = "".join(entry_html(e, highlight=e["number"] == dup)
                       for e in reversed(entries))
        self.txt_log.setHtml(html or f"<p style='color:{T.TEXT_DIM}'>"
                                     f"<i>No entries in {path.name}.</i></p>")

    # -- result -----------------------------------------------------------------
    def _on_replace(self):
        self.replace = True
        self.accept()

    def description(self):
        return "" if self.browse_only else self.ed_desc.toPlainText().strip()

    def write(self):
        """Write the entry (append, or replace the duplicate when Replace was
        clicked) and return the file it went to. Raises ``OSError``."""
        if self.browse_only:
            return None
        if self.duplicate is not None:
            if not self.replace:
                return None           # never append a second entry for a run
            path, e = self.duplicate
            return runlog.replace_entry(path, e["number"], self.description(),
                                        self.source, self.metadata, self.stats)
        return runlog.log_run(self.description(), self.source, self.metadata,
                              self.stats, log_dir=self.log_dir)
