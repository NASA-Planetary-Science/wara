"""
Initial parameter settings for GUI. It is used only when parameters are
passed directly in the command line.
"""

from pathlib import Path

from wara import peaksearch as ps
from wara import file_reader


def get_spect_search(commands):
    if commands["-o"] or commands["<file_name>"] is None:
        return None
    file_name = commands["<file_name>"]
    # The detector types below are accurate only for the example files.
    # Add a similar command for your own detector or modify the values below.
    if commands["--cebr"] or commands["--labr"]:
        fwhm_at_0 = 1.0
        ref_x = 420
        ref_fwhm = 20  # 41
    elif commands["--hpge"]:
        fwhm_at_0 = 0.1
        ref_x = 948
        ref_fwhm = 4.4
    else:
        fwhm_at_0 = float(commands["--fwhm_at_0"])
        ref_x = float(commands["--ref_x"])
        ref_fwhm = float(commands["--ref_fwhm"])

    if commands["--min_snr"] is None:
        min_snr = 5.0
    else:
        min_snr = float(commands["--min_snr"])

    path = Path(file_name)
    name_lower = path.name.lower()
    suffix_lower = path.suffix.lower()

    # Check composite suffixes before single-extension ones
    if name_lower.endswith("-lynx.csv"):
        spect = file_reader.read_lynx_csv(file_name)
    elif name_lower.endswith(".pha.txt"):
        spect = file_reader.read_multiscan(file_name)
    elif suffix_lower == ".csv":
        spect = file_reader.read_csv(file_name)
    elif suffix_lower == ".cnf":
        spect = file_reader.read_cnf(file_name)
    elif suffix_lower == ".txt":
        spect = file_reader.read_txt(file_name)
    elif suffix_lower == ".mca":
        spect = file_reader.read_mca(file_name)
    elif suffix_lower == ".spe":
        spect = file_reader.read_spe(file_name)
    else:
        raise ValueError(f"Unsupported file type: '{path.suffix}'. Supported: .csv, -lynx.csv, .cnf, .txt, .pha.txt, .mca, .spe")

    # peaksearch class
    search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
    return spect, search, ref_x, fwhm_at_0, ref_fwhm
