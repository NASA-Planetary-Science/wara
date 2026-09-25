"""Pop-up for the API / Neutrons "Log run" button.

Two tabs: *New entry* asks for a description and previews the metadata and
statistics that will be recorded for the loaded file; *Log entries* reads back
the entries already in the run-log files and can delete one. A run that is
already in the log is not logged twice: a warning is shown and the user may
replace the old entry instead. Reading and writing the files is done by
:mod:`wara.runlog`."""
from datetime import datetime
from html import escape

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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

_NEW_TAB, _LOG_TAB = 0, 1


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


def entry_html(entry, highlight=False, selected=False, link=False):
    """One run-log entry (a dict as returned by :func:`wara.runlog.read_entries`)
    as coloured HTML. *highlight* frames it amber (the duplicate run),
    *selected* frames it cyan (the entry picked for deletion) and *link* makes
    the title a clickable ``entry:N`` link with an ``entryN`` anchor."""
    desc = escape(entry.get("description") or "(no description)")
    desc = desc.replace("\n", "<br>")
    border = (T.ACCENT_CYAN if selected else
              T.ACCENT_AMBER if highlight else T.BORDER)
    width = 2 if (selected or highlight) else 1
    num = entry.get("number", "?")
    title = f"ENTRY {num}"
    if link:
        title = (f"<a name='entry{num}' href='entry:{num}' "
                 f"style='color:{_C_TITLE}; text-decoration:none'>{title}</a>")
    return (
        f"<table width='100%' cellspacing='0' cellpadding='8' border='{width}' "
        f"style='border-color:{border}; border-style:solid; margin-bottom:10px; "
        f"background:{T.BG_PANEL}'><tr><td>"
        f"<span style='color:{_C_TITLE}; font-weight:700; font-size:15px'>"
        f"{title}</span>"
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
    """Ask for a run description (tab 1) and browse / prune the run log (tab 2).

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
        self.replace = False                  # set when Replace was clicked

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)

        # Buttons first: the tabs' refresh code enables/disables them.
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.btn_save = bb.button(QDialogButtonBox.Save)
        self.btn_save.setToolTip("Append a new entry for this run to the log")
        self.btn_cancel = bb.button(QDialogButtonBox.Cancel)
        self.btn_replace = QPushButton("Replace entry")
        self.btn_replace.setToolTip(
            "Overwrite the existing entry for this run with this description, "
            "metadata and statistics (it keeps its entry number)")
        self.btn_replace.setCursor(Qt.PointingHandCursor)
        bb.addButton(self.btn_replace, QDialogButtonBox.ActionRole)
        self.btn_replace.clicked.connect(self._on_replace)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        # Delete sits on the left, away from Save; it only acts on Log entries.
        self.btn_delete = QPushButton("Delete entry")
        self.btn_delete.setObjectName("danger_btn")
        self.btn_delete.setCursor(Qt.PointingHandCursor)
        self.btn_delete.setToolTip(
            "Delete the selected entry from its log file (asks first); the "
            "entries after it are renumbered.\nPick the entry in the Entry list "
            "or click its title. Only available on the Log entries tab.")
        self.btn_delete.clicked.connect(self._on_delete)

        self.tabs = QTabWidget()
        self.tabs.addTab(QWidget() if self.browse_only else self._build_new_tab(),
                         "New entry")
        self.tabs.addTab(self._build_read_tab(), "Log entries")
        self.tabs.setTabToolTip(_NEW_TAB, "Load a run first to add an entry to the log"
                                if self.browse_only else
                                "Describe the loaded run and save it to the log")
        self.tabs.setTabToolTip(_LOG_TAB, "Read or delete the entries already "
                                          "saved in the run log")
        self._fit_tabbar(self.tabs)
        lay.addWidget(self.tabs)

        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(self.btn_delete)
        row.addWidget(bb, 1)
        lay.addLayout(row)

        if self.browse_only:
            self.tabs.setTabEnabled(_NEW_TAB, False)
            self.btn_save.setVisible(False)
            self.btn_replace.setVisible(False)
            self.btn_cancel.setText("Close")
        else:
            self._update_duplicate_ui()
            self.ed_desc.setFocus()
        self.tabs.currentChanged.connect(self._update_buttons)
        self.tabs.setCurrentIndex(_LOG_TAB if self.browse_only else _NEW_TAB)
        self._update_buttons()

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

    def _update_buttons(self, *_):
        """Save / Replace / Cancel belong to New entry and Delete to Log entries;
        the other tab's buttons are greyed out."""
        on_log = self.tabs.currentIndex() == _LOG_TAB
        if not self.browse_only:
            self.btn_save.setEnabled(not on_log and self.duplicate is None)
            self.btn_replace.setEnabled(not on_log)
            # (Browse-only keeps Close usable: it is the only button there.)
            self.btn_cancel.setEnabled(not on_log)
        self.btn_delete.setEnabled(on_log and self.selected_entry() is not None)

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
        # About four lines tall: the preview below gets the rest of the space.
        fm = self.ed_desc.fontMetrics()
        self.ed_desc.setFixedHeight(4 * fm.lineSpacing() + 16)
        lay.addWidget(self.ed_desc)

        lay.addWidget(header("RECORDED WITH THE ENTRY"))
        self._preview = {
            "number": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005
            "source": self.source, "description": "(your description)",
            "metadata": self.metadata, "stats": self.stats}
        self.txt_preview = _browser(
            "Metadata and statistics captured from the loaded file")
        lay.addWidget(self.txt_preview, 1)

        # Quiet (no pop-up) note when this run is already in the log.
        self.lbl_duplicate = QLabel()
        self.lbl_duplicate.setWordWrap(True)
        self.lbl_duplicate.setStyleSheet(f"color:{T.ACCENT_AMBER};")
        self.lbl_duplicate.setToolTip(
            "The run is identified by tab + date + run number (PIXIE runs) "
            "or tab + file path (trace files). See the Log entries tab.")
        lay.addWidget(self.lbl_duplicate)

        self.lbl_target = QLabel()
        self.lbl_target.setObjectName("stat_key"); self.lbl_target.setWordWrap(True)
        lay.addWidget(self.lbl_target)
        return w

    def _target_file(self):
        if self.duplicate is not None:
            return self.duplicate[0]
        return runlog.current_log_file(self.log_dir)

    def _update_duplicate_ui(self):
        """Sync the warning, preview number, target file and Replace button with
        ``self.duplicate`` (it changes when the duplicate entry is deleted)."""
        dup = self.duplicate
        self.btn_replace.setVisible(dup is not None)
        self.lbl_duplicate.setVisible(dup is not None)
        if dup is not None:
            path, e = dup
            self.lbl_duplicate.setText(
                f"⚠ This run is already logged: entry {e['number']} of "
                f"{path.name} ({e['timestamp']}). A second entry will not be "
                f"written — use Replace entry to overwrite it.")
        target = self._target_file()
        self._preview["number"] = (dup[1]["number"] if dup is not None
                                   else runlog.count_entries(target) + 1)
        self.txt_preview.setHtml(entry_html(self._preview))
        self.lbl_target.setText(
            f"Saved to {target}  (up to {runlog.MAX_ENTRIES} entries per file)")

    # -- tab 2: read / delete ---------------------------------------------------
    def _build_read_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 8, 6, 6); lay.setSpacing(6)
        row = QHBoxLayout(); row.setSpacing(8)
        lbl = QLabel("Log file:"); lbl.setObjectName("stat_key")
        row.addWidget(lbl)
        self.cmb_file = QComboBox()
        self.cmb_file.setToolTip("Run-log file to read (newest last)")
        row.addWidget(self.cmb_file, 1)
        lbl = QLabel("Entry:"); lbl.setObjectName("stat_key")
        row.addWidget(lbl)
        self.cmb_entry = QComboBox()
        self.cmb_entry.setMinimumWidth(190)
        self.cmb_entry.setToolTip(
            "Selected entry (framed in cyan) — the one Delete entry removes. "
            "You can also click an entry's title in the list below.")
        row.addWidget(self.cmb_entry)
        lay.addLayout(row)
        self.txt_log = _browser("Entries of the selected log file, newest first. "
                                "Click an entry's title to select it.")
        self.txt_log.anchorClicked.connect(self._on_anchor)
        lay.addWidget(self.txt_log, 1)

        self._files, self._entries = [], []
        self.cmb_file.currentIndexChanged.connect(lambda i: self._show_file(i))
        self.cmb_entry.currentIndexChanged.connect(self._on_entry_changed)
        start = None
        if self.duplicate is not None:
            start = (self.duplicate[0], self.duplicate[1]["number"])
        self._reload_files(start)
        return w

    def _reload_files(self, select=None):
        """Refill the file list; *select* = ``(path, number)`` to show."""
        self._files = runlog.log_files(self.log_dir)
        self.cmb_file.blockSignals(True)
        self.cmb_file.clear()
        for p in self._files:
            n = runlog.count_entries(p)
            self.cmb_file.addItem(f"{p.name}  ({n} entr{'y' if n == 1 else 'ies'})")
        self.cmb_file.blockSignals(False)
        if not self._files:
            self._entries = []
            self.cmb_entry.clear()
            self.txt_log.setHtml(
                f"<p style='color:{T.TEXT_DIM}'><i>The run log is empty.</i></p>")
            return
        idx = len(self._files) - 1
        if select is not None and select[0] in self._files:
            idx = self._files.index(select[0])
        self.cmb_file.blockSignals(True)
        self.cmb_file.setCurrentIndex(idx)
        self.cmb_file.blockSignals(False)
        self._show_file(idx, select[1] if select is not None else None)

    def _show_file(self, idx, number=None):
        """Show log file *idx* and select entry *number* (default: newest)."""
        if not 0 <= idx < len(self._files):
            return
        self._entries = list(reversed(runlog.read_entries(self._files[idx])))
        nums = [e["number"] for e in self._entries]
        self.cmb_entry.blockSignals(True)
        self.cmb_entry.clear()
        for e in self._entries:
            self.cmb_entry.addItem(f"{e['number']}  ·  {e['timestamp']}")
        if nums:
            self.cmb_entry.setCurrentIndex(nums.index(number) if number in nums
                                           else 0)
        self.cmb_entry.blockSignals(False)
        self._render_log()

    def _current_file(self):
        idx = self.cmb_file.currentIndex()
        return self._files[idx] if 0 <= idx < len(self._files) else None

    def selected_entry(self):
        """The entry dict selected on the Log entries tab, or None."""
        idx = self.cmb_entry.currentIndex()
        return self._entries[idx] if 0 <= idx < len(self._entries) else None

    def _render_log(self):
        path = self._current_file()
        if path is None:
            return
        dup = self.duplicate[1]["number"] if (
            self.duplicate is not None and self.duplicate[0] == path) else None
        sel = self.selected_entry()
        html = "".join(entry_html(e, highlight=e["number"] == dup,
                                  selected=e is sel, link=True)
                       for e in self._entries)
        bar = self.txt_log.verticalScrollBar()
        pos = bar.value()
        self.txt_log.setHtml(html or f"<p style='color:{T.TEXT_DIM}'>"
                                     f"<i>No entries in {path.name}.</i></p>")
        bar.setValue(pos)
        self._update_buttons()

    def _on_entry_changed(self, _idx):
        self._render_log()
        sel = self.selected_entry()
        if sel is not None:
            self.txt_log.scrollToAnchor(f"entry{sel['number']}")

    def _on_anchor(self, url):
        text = url.toString()
        if not text.startswith("entry:"):
            return
        nums = [e["number"] for e in self._entries]
        num = int(text.split(":", 1)[1])
        if num in nums:
            self.cmb_entry.setCurrentIndex(nums.index(num))

    def _confirm_delete(self, path, entry):
        desc = (entry["description"] or "(no description)").splitlines()[0]
        ans = QMessageBox.question(
            self, "Delete run-log entry",
            f"Delete entry {entry['number']} of {path.name}?\n\n"
            f"{entry['timestamp']} · {entry['source']}\n{desc}\n\n"
            "This cannot be undone. Later entries in the file are renumbered.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return ans == QMessageBox.Yes

    def _on_delete(self):
        path, e = self._current_file(), self.selected_entry()
        if path is None or e is None or not self._confirm_delete(path, e):
            return
        runlog.delete_entry(path, e["number"])
        if not self.browse_only:
            # The duplicate may be gone, or renumbered: look it up again.
            self.duplicate = runlog.find_run(self.source, self.metadata,
                                             self.log_dir)
            self._update_duplicate_ui()
        self._reload_files((path, e["number"]))
        self._update_buttons()

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
