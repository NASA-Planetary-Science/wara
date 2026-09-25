"""Run log: a plain-text notebook of the data files you have looked at.

Each call to :func:`log_run` appends one *entry* -- a user description plus the
metadata and statistics needed to recognise the file later -- to a text file in
a ``runlogs`` folder. A file holds at most ``max_entries`` (default 50)
entries; once full, the next entry starts a new file (``runlog_001.txt``,
``runlog_002.txt``, ...).

The folder is resolved, in order, from the ``log_dir`` argument, the
``WARA_RUNLOG_DIR`` environment variable, and finally ``runlogs/`` in the repo
root (next to ``data-path.txt``).

Example
-------
>>> from wara import runlog
>>> path = runlog.log_run("Cf-252 source, 10 cm", source="Neutrons",
...                       metadata={"File": "cf252.npz"},
...                       stats={"Traces": "12,000"})
>>> runlog.read_entries(path)[-1]["description"]
'Cf-252 source, 10 cm'
"""
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

MAX_ENTRIES = 50
FILE_PREFIX = "runlog_"
ENTRY_MARK = "=== ENTRY"
_FILE_RE = re.compile(rf"^{FILE_PREFIX}(\d+)\.txt$")
_RULE = "=" * 72


def default_log_dir():
    """The ``runlogs`` folder: ``$WARA_RUNLOG_DIR`` if set, else the repo root."""
    env = os.environ.get("WARA_RUNLOG_DIR")
    if env:
        return Path(env)
    return Path(files("wara")).parent / "runlogs"


def log_files(log_dir=None):
    """All run-log files in *log_dir*, oldest (lowest number) first."""
    d = Path(log_dir) if log_dir is not None else default_log_dir()
    if not d.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in d.iterdir()
             if (m := _FILE_RE.match(p.name))]
    return [p for _, p in sorted(found)]


def count_entries(path):
    """Number of entries in one run-log file (0 if it does not exist)."""
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith(ENTRY_MARK))


def current_log_file(log_dir=None, max_entries=MAX_ENTRIES):
    """The file the next entry goes to: the newest file while it has room,
    otherwise a new file numbered one higher."""
    d = Path(log_dir) if log_dir is not None else default_log_dir()
    existing = log_files(d)
    if existing and count_entries(existing[-1]) < max_entries:
        return existing[-1]
    n = int(_FILE_RE.match(existing[-1].name).group(1)) + 1 if existing else 1
    return d / f"{FILE_PREFIX}{n:03d}.txt"


def _fmt_section(title, items):
    if not items:
        return []
    width = max(len(str(k)) for k in items)
    # One line per item: fold any newline in a value so the file stays parseable.
    return [f"{title}:"] + [f"  {k!s:<{width}} : {' / '.join(str(v).splitlines())}"
                            for k, v in items.items()]


def format_entry(number, description, source, metadata=None, stats=None,
                 timestamp=None):
    """Render one entry as text (ends with a blank line)."""
    # Local wall-clock time: what a person reading the log expects.
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005
    desc = description.strip() or "(no description)"
    lines = [_RULE, f"{ENTRY_MARK} {number} | {ts} | {source}", _RULE,
             "Description:"]
    lines += [f"  {line}" for line in desc.splitlines()]
    lines += _fmt_section("Metadata", metadata or {})
    lines += _fmt_section("Statistics", stats or {})
    return "\n".join(lines) + "\n\n"


def log_run(description, source, metadata=None, stats=None, log_dir=None,
            max_entries=MAX_ENTRIES, timestamp=None):
    """Append an entry to the run log and return the file it was written to.

    Parameters
    ----------
    description : str
        Free-text description from the user (may span several lines).
    source : str
        Where the file was loaded, e.g. ``"API"`` or ``"Neutrons"``.
    metadata, stats : dict, optional
        ``label -> value`` pairs identifying the file (run, channel, path, ...)
        and summarising it (counts, live time, gates, ...). Written in order.
    log_dir : str or Path, optional
        Folder for the log files (see the module docstring for the default).
    max_entries : int
        Entries per file before a new file is started.
    timestamp : datetime, optional
        Time stamp of the entry (defaults to now).
    """
    if max_entries < 1:
        raise ValueError("max_entries must be >= 1")
    d = Path(log_dir) if log_dir is not None else default_log_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = current_log_file(d, max_entries)
    n = count_entries(path) + 1
    with path.open("a", encoding="utf-8") as f:
        f.write(format_entry(n, description, source, metadata, stats, timestamp))
    return path


def read_entries(path):
    """Parse a run-log file back into a list of dicts with ``number``,
    ``timestamp``, ``source``, ``description``, ``metadata`` and ``stats``."""
    entries, cur, section = [], None, None
    with Path(path).open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith(ENTRY_MARK):
                num, ts, src = (s.strip() for s in
                                line[len(ENTRY_MARK):].split("|", 2))
                cur = {"number": int(num), "timestamp": ts, "source": src,
                       "description": "", "metadata": {}, "stats": {}}
                entries.append(cur)
                section = None
            elif cur is None or line == _RULE or not line:
                continue
            elif line == "Description:":
                section = "description"
            elif line == "Metadata:":
                section = "metadata"
            elif line == "Statistics:":
                section = "stats"
            elif section == "description":
                sep = "\n" if cur["description"] else ""
                cur["description"] += sep + line[2:]
            elif section in ("metadata", "stats"):
                key, _, val = line.strip().partition(" : ")
                cur[section][key.strip()] = val
    return entries
