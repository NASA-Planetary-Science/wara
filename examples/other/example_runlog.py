"""
Example for the "Log run..." button of the API / Neutrons tabs (``wara.runlog``).

The GUI button asks for a description and appends it, together with metadata
and statistics of the loaded file, to a text file in ``runlogs/``. This script
does the same from Python for the bundled EJ-309 trace subset, then reads the
log back. Each log file holds up to 50 entries before a new one is started.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from wara import runlog
from wara.neutron_psd import NeutronTraces

npz = Path(__file__).resolve().parent.parent / "data" / "EJ309_neutrons_traces_subset.npz"
nt = NeutronTraces.from_npz(npz).compute()
e = nt.energy[nt.valid]

log_dir = Path(__file__).resolve().parent / "runlogs"
desc = "EJ-309 subset, default gates -- neutron/gamma bands well separated"
meta = {"File": npz.name, "Samples / trace": nt.time_ns.size}
stats = {"Traces": f"{nt.n_traces:,}",
         "Valid pulses": f"{int(nt.valid.sum()):,}",
         "Energy median (V·ns)": f"{np.median(e):.4g}"}

# A run is logged once: find_run() spots an existing entry for the same run
# (tab + date/run, or tab + file path) and replace_entry() overwrites it, so
# running this script again updates the entry instead of adding another.
dup = runlog.find_run("Neutrons", meta, log_dir=log_dir)
if dup is None:
    path = runlog.log_run(desc, "Neutrons", meta, stats, log_dir=log_dir)
    print(f"Wrote entry to {path}\n")
else:
    path, old = dup
    runlog.replace_entry(path, old["number"], desc, "Neutrons", meta, stats)
    print(f"Replaced entry {old['number']} of {path}\n")
print(path.read_text(encoding="utf-8").split(runlog.ENTRY_MARK)[-1])
last = runlog.read_entries(path)[-1]
print("Parsed back:", last["description"], "|", last["stats"])

# For a PIXIE run folder, run_folder_fields() collects the setup, the channels
# with data, the data-folder sizes and per-channel event / trace counts. Here a
# tiny fake run folder is built so the example runs without real data.

with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp) / "RUN-2026-09-24-00001"
    (run / "settings").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"setup": "ATLAS_NEUTRONS"}))
    counts = [0] * 16
    counts[1], counts[9] = 99272, 101984
    (run / "settings" / "RUN-stats-2026-09-24.json").write_text(
        json.dumps([{"module": 0, "output_counts": counts}]))
    meta, stats = runlog.run_folder_fields(run)
    print("\nRun folder metadata:", meta)
    print("Run folder statistics:", stats)

