"""
Fast, low-memory loading of an API run with ``read_parquet_file``.

``columns=`` reads only the listed parquet columns and the channel filter is
applied inside the reader, so unused columns (timestamps, the raw A-D corner
signals, ...) never reach memory. ``lists_as_arrow=True`` keeps the per-event
``Trace`` column Arrow-backed (one buffer instead of one numpy array per row);
``neutron_psd.trace_matrix`` accepts it directly.  This is what the GUI's API
tab does on load; "Apply to data" re-reads the full run so no column is lost
when saving.

Needs the run under a folder listed in ``data-path.txt``.
"""
import time

from wara import read_parquet_api
from wara.neutron_psd import trace_matrix

DATE, RUNNR, CH = "2026-09-25", 1, 1

t0 = time.perf_counter()
df = read_parquet_api.read_parquet_file(
    DATE, RUNNR, ch=CH,
    columns=["channel", "energy", "dt", "X2", "Y2", "Trace"],
    lists_as_arrow=True)
print(f"{len(df):,} events, columns {list(df.columns)} "
      f"in {time.perf_counter() - t0:.1f} s, "
      f"{df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

if "Trace" in df:
    traces, has = trace_matrix(df["Trace"])
    print(f"trace matrix {traces.shape} ({has.sum():,} events with a trace)")
