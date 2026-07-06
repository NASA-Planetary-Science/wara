"""File loading for the wara GUI — reuses wara.file_reader, mirroring the
suffix dispatch used by the legacy launcher (wara.param_handle)."""
from pathlib import Path

from wara import file_reader

# Filter string for the Open dialog.
OPEN_FILTER = (
    "Spectrum files (*.csv *.cnf *.txt *.mca *.spe);;All files (*)"
)


def load_spectrum_file(path):
    """Return a wara.spectrum.Spectrum for the given file path."""
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()

    if name.endswith("-lynx.csv"):
        return file_reader.read_lynx_csv(path)
    if name.endswith(".pha.txt"):
        return file_reader.read_multiscan(path)
    if suffix == ".csv":
        return file_reader.read_csv(path)
    if suffix == ".cnf":
        return file_reader.read_cnf(path)
    if suffix == ".txt":
        return file_reader.read_txt(path)
    if suffix == ".mca":
        return file_reader.read_mca(path)
    if suffix == ".spe":
        return file_reader.read_spe(path)
    raise ValueError(
        f"Unsupported file type: '{suffix}'. "
        "Supported: .csv, -lynx.csv, .cnf, .txt, .pha.txt, .mca, .spe"
    )
