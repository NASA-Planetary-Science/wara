"""
Classes and functions to read different file types
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from wara import spectrum as sp
from wara import cnf_reader
import datetime


def process_df(df):
    """
    Process dataframe.
    Must have at least one header with one of the key words listed
    in name_lst.

    Parameters
    ----------
    df : pandas dataframe.
        dataframe containing counts or counts and energy.

    Returns
    -------
    unit : string.
        X-axis units e.g. channels, keV, MeV.
    cts_col : string.
        name of column header for counts.
    erg : string.
        name of column header for energies.
    """
    # remove white spaces and convert to lower case
    df.columns = df.columns.str.replace(" ", "").str.lower()
    ###
    name_lst = ["count", "counts", "cts", "data", "countrate(cps)"]
    e_lst = ["energy", "energies", "erg"]
    unit_dict = {"ev": "eV", "kev": "keV", "mev": "MeV", "gev": "GeV"}
    col_lst = list(df.columns)
    # cts_col = [s for s in col_lst if "counts" in s.lower()][0]
    cts_col = None
    erg = None
    unit = "keV_default"  # if no units given, default to keV
    for s in col_lst:
        s2 = re.split("[^a-zA-Z]", s)  # split by non alphabetic character
        s2 = [x for x in s2 if x]  # remove empty string
        if s in name_lst:
            cts_col = s
            continue
        for st in s2:
            if st in e_lst:
                erg = s
            if st in list(unit_dict.keys()):
                unit = unit_dict[st]
    return unit, cts_col, erg


def read_csv(file_name):
    """
    Read .csv file.
    Must have at least one header with one of the key words listed
    in name_lst.

    Parameters
    ----------
    file_name : string.
        file path.

    Returns
    -------
    e_units : string
        X-axis units e.g. channels, keV, MeV.
    spect : Spectrum instance.
        Spectrum object from wara.

    """
    df = pd.read_csv(file_name)

    unit, cts_col, erg = process_df(df)

    if cts_col is None:
        raise ValueError("No counts column found. Column must be named: counts, cts, data, or countrate(cps)")
    elif erg is None:
        # print("working with channel numbers")
        e_units = "channels"
        spect = sp.Spectrum(counts=df[cts_col], e_units=e_units)
        spect.x = spect.channels
    else:
        # print("working with energy values")
        e_units = unit
        spect = sp.Spectrum(counts=df[cts_col], energies=df[erg], e_units=e_units)
        spect.x = spect.energies

    return spect


def read_txt(filename):
    description = None
    plot_label = None
    date_created = None
    realtime = None
    livetime = None
    erg_cal = None
    start_idx = None
    with open(filename, "r", encoding="utf-8") as myfile:
        filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            parts = line.split()
            if not parts:
                continue
            if parts[0].lower() == "description:" and len(parts) > 1:
                description = " ".join(parts[1:])
            if parts[0].lower() == "label:" and len(parts) > 1:
                plot_label = " ".join(parts[1:])
            if parts[0].lower() == "date" and parts[1].lower() == "created:" and len(parts) > 2:
                date_created = " ".join(parts[2:])
            if parts[0].lower() == "real" and parts[1].lower() == "time" and len(parts) > 2:
                realtime = parts[3]
            if parts[0].lower() == "live" and parts[1].lower() == "time" and len(parts) > 2:
                livetime = parts[3]
            if parts[0].lower() == "energy" and parts[1].lower() == "calibration:":
                if len(parts) > 2:
                    erg_cal = " ".join(parts[2:])
                start_idx = i + 1
                break
    
    if start_idx is None:
        raise ValueError(
            f"Could not find 'Energy calibration:' header in file: {filename}. "
            "File may be malformed or in an unexpected format."
        )
    df = pd.read_csv(filename, skiprows=start_idx)
    unit, cts_col, erg = process_df(df)

    if realtime == "None":
        realtime = None
    else:
        realtime = float(realtime)
    if livetime == "None":
        livetime = None
    else:
        livetime = float(livetime)
    if description == "None":
        description = None
    if plot_label == "None":
        plot_label = None
    if date_created == "None":
        date_created = None
    if erg_cal == "None":
        erg_cal = None

    if cts_col is None:
        raise ValueError("No counts column found. Column must be named: counts, cts, data, or countrate(cps)")
    elif erg is None:
        e_units = "channels"
        spect = sp.Spectrum(
            counts=df[cts_col],
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            description=description,
            acq_date=date_created,
            energy_cal=erg_cal,
            label=plot_label,
        )
        spect.x = spect.channels
    else:
        # print("working with energy values")
        e_units = unit
        spect = sp.Spectrum(
            counts=df[cts_col],
            energies=df[erg],
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            description=description,
            acq_date=date_created,
            energy_cal=erg_cal,
            label=plot_label,
        )
    return spect


def read_cnf(filename):
    """
    Read CNF file.

    Parameters
    ----------
    filename : string.
        file path.

    Returns
    -------
    e_units : string
        X-axis units e.g. channels, keV, MeV.
    spect : Spectrum instance.
        Spectrum object from wara.

    """
    dict_cnf = cnf_reader.read_cnf_file(filename, write_output=False)

    counts = dict_cnf["Channels data"]
    energy = dict_cnf["Energy"]
    livetime = dict_cnf["Live time"]
    realtime = dict_cnf["Real time"]
    start_date_time = dict_cnf["Start time"]

    if energy is None:
        e_units = "channels"
        spect = sp.Spectrum(
            counts=counts,
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            acq_date=start_date_time,
        )
    else:
        erg_coeff = dict_cnf["Energy coefficients"]
        erg_eqn = f"{erg_coeff[0]} + {erg_coeff[1]}*ch + {erg_coeff[2]}*ch^2 + {erg_coeff[3]}*ch^3"
        e_units = dict_cnf["Energy unit"]
        spect = sp.Spectrum(
            counts=counts,
            energies=energy,
            e_units=e_units,
            livetime=livetime,
            realtime=realtime,
            acq_date=start_date_time,
            energy_cal=erg_eqn,
        )

    return spect


class ReadMCA:
    def __init__(self, file):
        """
        Read .mca file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.tag = None
        self.description = None
        self.gain = None
        self.threshold = None
        self.live_mode = None
        self.preset_time = None
        self.live_time = None
        self.real_time = None
        self.start_time = None
        self.serial_no = None
        self.counts = None
        if file[-3:].lower() != "mca":
            raise ValueError(f"Expected a .mca file, got: {file}")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()

        start_idx = None
        for i, line in enumerate(filelst):
            parts = line.lower().split()
            if parts[0] == "tag":
                self.tag = parts[-1]
            elif parts[0] == "description":
                self.description = parts[-1]
            elif parts[0] == "gain":
                try:
                    self.gain = float(parts[-1])
                except ValueError:
                    pass
            elif parts[0] == "threshold":
                try:
                    self.threshold = float(parts[-1])
                except ValueError:
                    pass
            elif parts[0] == "live_mode":
                try:
                    self.live_mode = float(parts[-1])
                except ValueError:
                    pass
            elif parts[0] == "preset_time":
                try:
                    self.preset_time = float(parts[-1])
                except ValueError:
                    pass
            elif parts[0] == "live_time":
                try:
                    self.live_time = float(parts[-1])
                except ValueError:
                    pass
            elif parts[0] == "real_time":
                try:
                    self.real_time = float(parts[-1])
                except ValueError:
                    pass
            elif parts[0] == "start_time":
                self.start_time = parts[-2] + " " + parts[-1]
            elif parts[0] == "serial_number":
                self.serial_no = parts[-1]
            elif parts[0] == "<<data>>":
                start_idx = i
                break
        if start_idx is None:
            raise ValueError(f"Could not find '<<DATA>>' section in file: {self.file}")
        self.counts = np.array(filelst[start_idx + 1 : -1], dtype=int)

def read_mca(file):
    mca = ReadMCA(file)
    spect = sp.Spectrum(
        counts=mca.counts,
        livetime=mca.live_time,
        realtime=mca.real_time,
        acq_date=mca.start_time,
        description=mca.description,
        label=mca.tag,
    )
    return spect
    


class ReadSPE:
    def __init__(self, file):
        """
        Read .Spe file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = Path(file)
        self.description = None
        self.detector = None
        self.detector_description = None
        self.version = None
        self.time_str = None
        self.time_s = None
        self.date = None
        self.start_time = None
        self.live_time = None
        self.real_time = None
        self.channels = None
        self.ROI = None
        self.counts = None
        self.erg_cal = None

        if self.file.suffix.lower() != ".spe":
            raise ValueError(f"Expected a .Spe file, got: {file}")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()
        start_idx = None
        end_idx = None
        for i, line in enumerate(filelst):
            parts = line.lower().split()
            if parts[0] == "$spec_id:":
                self.description = filelst[i + 1]
            if "det#" in parts:
                self.detector = parts[1]
            if "detdesc#" in parts:
                self.detector_description = parts[1:]
            if "ap#" in parts:
                self.version = parts[1:]
            if "$date_mea:" in parts:
                self.date = filelst[i + 1].split()[0]
                self.time_str = filelst[i + 1].split()[1]
                tme = datetime.datetime.strptime(self.time_str, "%H:%M:%S")
                self.time_s = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
            if "$meas_tim:" in parts:
                self.real_time = float(filelst[i + 1].split()[1])
                self.live_time = float(filelst[i + 1].split()[0])
            if "$data:" in parts:
                self.channels = int(filelst[i + 1].split()[1])
                start_idx = i
            if "$roi:" in parts:
                self.ROI = filelst[i + 1].split()[0]
                end_idx = i
            if "$ener_fit:" in parts:
                self.erg_cal = filelst[i + 1].split()
        if start_idx is None:
            raise ValueError(f"Could not find '$DATA:' section in file: {self.file}")
        if end_idx is None:
            raise ValueError(f"Could not find '$ROI:' section in file: {self.file}")
        self.counts = np.array(filelst[start_idx + 2 : end_idx - 1], dtype=int)

def read_spe(file):
    spe = ReadSPE(file)
    spect = sp.Spectrum(
        counts=spe.counts,
        livetime=spe.live_time,
        realtime=spe.real_time,
        acq_date=spe.date + " " + spe.time_str,
        description=spe.description,
        energy_cal=spe.erg_cal,
    )
    return spect


def read_lynx_csv(file_name):
    with open(file_name, "r") as myfile:
        filelst = myfile.readlines()

    istart = None
    for i, line in enumerate(filelst):
        parts = line.lower().split()
        if "channel," in parts and "counts" in parts:
            istart = i
            break
    if istart is None:
        raise ValueError(f"Could not find channel/counts header in file: {file_name}")
    df = pd.read_csv(file_name, skiprows=istart, dtype=float)
    df.columns = df.columns.str.replace(" ", "")  # remove white spaces
    df.columns = df.columns.str.lower()
    ###
    cols = ["channel", "energy(kev)", "counts"]  # as listed on lynx
    # print("working with energy values")
    e_units = "keV"
    spect = sp.Spectrum(counts=df[cols[2]], energies=df[cols[1]], e_units=e_units)
    return spect


class ReadLynxCsv:
    def __init__(self, file):
        """
        Read -lynx.csv file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.start_time = None
        self.energy_calibration = None
        self.live_time = None
        self.real_time = None
        self.elapsed_computational = None
        self.eunits = None
        self.counts = None
        self.count_rate = None
        self.energy = None
        self.spect = None
        self.nch = None
        if file[-9:].lower() != "-lynx.csv":
            raise ValueError(f"Expected a -lynx.csv file, got: {file}")
        self.parse_file()

    def parse_file(self):
        with open(self.file, "r") as myfile:
            filelst = myfile.readlines()
        for i, line in enumerate(filelst):
            parts = line.lower().split()
            if "start" in parts and "time," in parts:
                self.start_time = " ".join(parts[2:])
            if "energy" in parts and "calibration," in parts:
                self.energy_calibration = " ".join(parts[2:])
            if "live" in parts and "time" in parts:
                self.live_time = parts[-1] + parts[-2]
            if "real" in parts and "time" in parts:
                self.real_time = parts[-1] + parts[-2]
            if "elapsed" in parts and "computational," in parts:
                self.elapsed_computational = parts[-1]
            if "channel," in parts and "counts" in parts:
                istart = i
                break
        df = pd.read_csv(self.file, skiprows=istart, dtype=float)
        df.columns = df.columns.str.replace(" ", "")  # remove white spaces
        df.columns = df.columns.str.lower()
        cols = ["channel", "energy(kev)", "counts"]  # as listed on lynx
        e_units = "keV"
        self.spect = sp.Spectrum(
            counts=df[cols[2]],
            energies=df[cols[1]],
            e_units=e_units,
            livetime=float(self.live_time[:-4]),
        )
        self.spect.x = self.spect.energies
        self.counts = self.spect.counts.sum()
        self.count_rate = self.counts / float(self.live_time[0:-4])
        self.nch = self.spect.counts.shape[0]


def read_multiscan(file):
    description = None
    start_time = None
    energy_calibration = None
    live_time = None
    real_time = None
    eunits = None
    if file[-8:].lower() != ".pha.txt":
        raise ValueError(f"Expected a .pha.txt file, got: {file}")

    with open(file, "r") as myfile:
        filelst = myfile.readlines()
    istart = None
    cols = None
    for i, line in enumerate(filelst):
        parts = line.lower().strip().split(",")
        if "name" in parts and len(parts) > 1:
            description = parts[1]
        if "time started" in parts:
            start_time = ",".join(parts[1:]).strip('"')
        if "live time when finished" in parts:
            tme = datetime.datetime.strptime(parts[1], "%H:%M:%S.%f")
            live_time = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
        if "real time when finished" in parts:
            tme = datetime.datetime.strptime(parts[1], "%H:%M:%S.%f")
            real_time = tme.hour * 60 * 60 + tme.minute * 60 + tme.second
        if "energy equation" in parts:
            energy_calibration = parts[1]
            eunits = parts[1].split("+")[0][-3:]
        if ["channel", "energy", "counts"] == parts:
            istart = i
            cols = parts
            break
    if istart is None:
        raise ValueError(f"Could not find 'channel,energy,counts' header in file: {file}")
    df = pd.read_csv(file, skiprows=istart, dtype=float)
    df.columns = cols
    spect = sp.Spectrum(
        counts=df["counts"],
        energies=df["energy"],
        description=description,
        e_units=eunits,
        livetime=live_time,
        realtime=real_time,
        energy_cal=energy_calibration,
        acq_date=start_time,
    )
    return spect

class ReadMultiScanTlist:
    def __init__(self, file):
        """
        Read MultiScan .txt Tlist file.
        
        Parameters
        ----------
        file : sting
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.energy_flag = False # default
        self.df = None
        
    def read_file(self):
        split_data = []
        if self.file[-3:] == "txt":
            with open(self.file, mode="r") as f:
                file_lst = f.readlines()
            for line in file_lst:
                parts = line.strip().split()
                split_data.append(parts)
        try:
            cols = ["channel", "ts"]
            df = pd.DataFrame(columns=cols, data=split_data, dtype=np.float64)
            self.df = df
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not parse time-list data from {self.file}") from e
   

class ReadCaenListMode:
    def __init__(self, file):
        """
        Read CAEN .txt list mode data file.

        Parameters
        ----------
        file : string.
            file path.

        Returns
        -------
        None.

        """
        self.file = file
        self.header0 = None
        self.header1 = None
        self.header2 = None
        self.header3 = None
        self.header4 = None
        self.idx_start = None  # start of data after header
        self.df = None
        self.read_file()
        self.parse_header()

    def read_file(self):
        with open(self.file, "r") as myfile:
            self.filelst = [line.rstrip() for line in myfile]

    def parse_header(self):
        for i, line in enumerate(self.filelst):
            parts = line.lower().split(":")
            if parts[0] == "header0":
                self.header0 = int(parts[1])
            if parts[0] == "header1":
                self.header1 = int(parts[1])
            if parts[0] == "header2":
                self.header2 = int(parts[1])
            if parts[0] == "header3":
                self.header3 = int(parts[1])
            if parts[0] == "header4":
                self.header4 = int(parts[1])
                self.idx_start = i
                break

    def parse_data(self):
        data = []
        for i, line in enumerate(self.filelst[self.idx_start :]):
            parts = line.split()
            data.append(parts)
        data = np.array(data[1:], dtype=int)
        cols = ["ts (ns)", "channel", "flag"]
        self.df = pd.DataFrame(columns=cols, data=data)
