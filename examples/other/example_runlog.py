"""
Example for the "Log run..." button of the API / Neutrons tabs (``wara.runlog``).

The GUI button asks for a description and appends it, together with metadata
and statistics of the loaded file, to a text file in ``runlogs/``. This script
does the same from Python for the bundled EJ-309 trace subset, then reads the
log back. Each log file holds up to 50 entries before a new one is started.
"""
from pathlib import Path

import numpy as np

from wara import runlog
from wara.neutron_psd import NeutronTraces

npz = Path(__file__).resolve().parent.parent / "data" / "EJ309_neutrons_traces_subset.npz"
nt = NeutronTraces.from_npz(npz).compute()
e = nt.energy[nt.valid]

path = runlog.log_run(
    "EJ-309 subset, default gates -- neutron/gamma bands well separated",
    source="Neutrons",
    metadata={"File": npz.name, "Samples / trace": nt.time_ns.size},
    stats={"Traces": f"{nt.n_traces:,}",
           "Valid pulses": f"{int(nt.valid.sum()):,}",
           "Energy median (V·ns)": f"{np.median(e):.4g}"},
    log_dir=Path(__file__).resolve().parent / "runlogs")

print(f"Wrote entry to {path}\n")
print(path.read_text(encoding="utf-8").split(runlog.ENTRY_MARK)[-1])
last = runlog.read_entries(path)[-1]
print("Parsed back:", last["description"], "|", last["stats"])
