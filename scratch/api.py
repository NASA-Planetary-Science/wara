from datetime import datetime
from importlib import resources
import pickle
import sys

import numpy as np

import scipy.constants as SC
from ROOTS.data import DataManager

def vel(E, m):
    """Calculate the velocity [m/s] from E and m (both in MeV)"""
    gamma = 1 + E / m
    beta = np.sqrt(1 - 1 / gamma / gamma)
    return beta * SC.c


# define constants
E_alpha = 3.5  # MeV
m_alpha = SC.physical_constants["alpha particle mass energy equivalent in MeV"][0]
v_alpha = vel(E_alpha, m_alpha)

E_neutron = 14.1  # MeV
m_neutron = SC.physical_constants["neutron mass energy equivalent in MeV"][0]
v_neutron = vel(E_neutron, m_neutron)


def fit_z_location(df, meta, limit, detectors, zpos=-1.15, ion_energy=55.0):
    """Move the largest peak to the position of zpos

    This calculates the largest point in the Z-histogram for both
    detectors and moves them to zpos by adjusting the timing offsets

    It doesn't change the dataframe, but just updated the z-align
    variables in the metadata (which are not saved until meta.save() is called).

    """
    X = np.linspace(-10, 10, 2000)
    iteration_max = 15

    for det in detectors:
        iteration = 0

        # initial value for while loop
        dt = 1.0

        # avoid oscillating between two values by slowly reducing the offset we apply
        rate = 1.0

        tmp_df = limit.apply_lmit(df, det)

        while (abs(dt) > 0.2e-9) and (iteration < iteration_max):
            # set to something small so that while loop will stop if detector is not selected
            iteration += 1
            hist, _ = np.histogram(tmp_df["Z"], bins=2000, range=[-10, 10])
            if hist.sum():
                N = hist.argmax()
                zmax = X[N]
                dz = zmax - zpos
                # estimate dt
                dt = dz / v_neutron
                meta["z-align"][det] -= dt * rate
                print(
                    f"{det:>10}: shifting by {dt:9.4g}  zmax:{zmax:9.4f} offset:{meta['z-align'][det]:9.4g}"
                )
            else:
                dt = 0  # stop while loop
            calcXYZ(tmp_df, meta, ion_energy=ion_energy)
            rate *= 0.9
        if iteration == iteration_max:
            print("[orange3]WARNING[/] stopped z-align: iteration limit reached")


def map_alpha_XY(df, fix_edges=False):
    """Some events have xy locations outside of the actual window, so
    we bin it first and find a min and max that fit most of the
    values and use these to define the corners of the alpha
    detector.

    This modifies the dataframe in place, so it doesn't have to be returned.

    This needs to be checked and improved
    """

    if "A" in df.columns:
        # intial calculation of x, y
        df["x"] = (df["B"] + df["C"] - df["D"] - df["A"]) / (
            df["A"] + df["B"] + df["C"] + df["D"]
        )
        # 1.305 is the factor to make alpha detector measured positions a square
        df["y"] = (
            1.305
            * (df["B"] + df["A"] - df["D"] - df["C"])
            / (df["A"] + df["B"] + df["C"] + df["D"])
        )
        if fix_edges:
            # load position correction spline fit lookup tables
            with (resources.files("ROOTS") / "Data" / "DX.ilut").open("rb") as f:
                DX = pickle.load(f)
            with (resources.files("ROOTS") / "Data" / "DY.ilut").open("rb") as f:
                DY = pickle.load(f)

            # energy amplitudes
            df["s"] = df["A"] + df["B"] + df["C"] + df["D"]

            # simulation mean energy amplitude of center values:
            center_amp_sim = 1330847.619223938

            # finding mean energy amplitude of center events from real data to nomralize energy amplitude
            X_real = np.array(df["x"])
            Y_real = np.array(df["y"])
            amp_real = np.array(df["s"])
            amp_freq_real = np.array(amp_real)[
                (np.abs(X_real) ** 2 + np.abs(Y_real) ** 2 < 0.25)
            ]
            center_amp_real = np.mean(amp_freq_real)

            # normalizing energy amplitude
            normalizing_factor = center_amp_sim / center_amp_real
            df["s"] = df["s"] * normalizing_factor

            # convert to polar coordinates
            df["phi"] = np.arctan2(df["y"], df["x"])
            df["R"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)

            # use LinearNDInterpolate spline fit
            df["dx"] = DX(df["phi"], df["s"])
            df["dy"] = DY(df["phi"], df["s"])
            df = df.replace(np.nan, 0)

            df["new_x"] = df["x"] + df["dx"]
            df["new_y"] = df["y"] + df["dy"]

            #        #set borders such that only events outside it get their position corrected
            #        #do not use until these values are improved
            #        #filters can also be changed however desired
            #        x_filter1 = df.x > 0.65
            #        x_filter2 = df.x < -0.65
            #        y_filter1 = df.y > 0.65
            #        y_filter2 = df.y < -0.65
            #        df["new_x"] = df["x"]
            #        df["new_x"][x_filter1 | x_filter2 | y_filter1 | y_filter2] = df["x"] + df["dx"]
            #        df["new_y"] = df["y"]
            #        df["new_y"][x_filter1 | x_filter2 | y_filter1 | y_filter2] = df["y"] + df["dy"]

            df["x"] = df["new_x"]
            df["y"] = df["new_y"]

    else:
        # need to convert from old [0, 1] range to new [-1, 1] range
        df["x"] = (df["x"] - 0.5) * 2
        df["y"] = (df["y"] - 0.5) * 2

    for column in ["x", "y"]:
        val, edges = np.histogram(df[column], bins=1000)
        # find midpoint of bins
        edges = 0.5 * (edges[1:] + edges[:-1])
        noise_level = val.max() * 25 / 100
        region = edges[val > noise_level]
        mymin = region.min()
        mymax = region.max()

        # scale to cm of alpha detector, e.g. -2.4cm to 2.4 cm
        # flip direction of the X axis, so that an XY image uses the same coordinate system
        # as if you are standing in front of the experiment
        if column == "x":
            df[f"{column.upper()}_alpha"] = -(
                (df[column] - mymin) / (mymax - mymin) * 0.048 - 0.024
            )
        else:
            df[f"{column.upper()}_alpha"] = (df[column] - mymin) / (
                mymax - mymin
            ) * 0.048 - 0.024

    return df


def calcXYZ(
    df, meta, cleanup: bool = True, simulation: bool = False, ion_energy: float = 0.0
):
    """Modifies a pandas dataframe to calculate XYZ position

    The dataframe should have a 'dt', 'X_alpha', and 'Y_alpha' column
    where dt is the measured difference in time between the alpha and
    the gamma x, y are the xy coordinates on the alpha detectors, from
    e.g. X=(sum of two corners)/(sum of four corners) that is, these
    are between 0 and 1

    It also should have a channel column indicating what channel was used.

    This modifies the dataframe in place, so it doesn't have to be returned.

    Parameters
    ----------

    df : pandas.DataFrame
       DataFrame to hold xyz data
    meta : ROOTS.metadata.Metadata
       Holds the detector position and other information
    cleanup
       Flag to remove intermediate steps in the calculation, for debugging only
    simulation
       Skip offset correction for simulated data
    ion_energy
       Beam energy in keV

    """

    if "surface" in df:
        simulation = True

    assert "dt" in df
    assert "X_alpha" in df
    assert "Y_alpha" in df
    if simulation:
        df["File time"] = datetime.now().timestamp()
    assert "File time" in df
    assert len(df) > 0
    assert df["File time"].iloc[0] > datetime(2018, 1, 1).timestamp(), (
        "File-time column Does not seem to be a unix time stamp (rerun ROOTS-bin2pandas?)"
    )
    if not simulation:
        assert "channel" in df.columns

    # use an abritraty timestamp to get the detector locations
    scintillator_dz = meta["scintillator distance"]
    v_com_direction = np.array(meta["ion beam axis"])

    E, v_cm = np.load(resources.files("ROOTS") / "Data" / "center-of-mass.npy")
    v_com = np.interp(ion_energy, xp=E, fp=v_cm, left=0)
    v_com_vector = v_com * v_com_direction

    # Adjust case for consistency
    z_align = {k.lower(): v for k, v in meta["z-align"].items()}
    det_in_meta = {k.lower(): v for k, v in meta["detectors"].items()}

    # add the Z location of the scintillator
    df["Z_alpha"] = scintillator_dz  # in m

    # solve quadratic equation to calculate t_alpha
    # see ipython notebook atap-roots-analysis/center_of_mass/Center_of_mass.ipynb for details

    df["a"] = v_alpha**2 - v_com**2
    df["b"] = 2 * (
        df["X_alpha"] * v_com_vector[0]
        + df["Y_alpha"] * v_com_vector[1]
        + df["Z_alpha"] * v_com_vector[2]
    )
    df["c"] = -(
        df["X_alpha"] * df["X_alpha"]
        + df["Y_alpha"] * df["Y_alpha"]
        + df["Z_alpha"] * df["Z_alpha"]
    )

    df["t_alpha"] = (
        (-df["b"] + np.sqrt(df["b"] * df["b"] - 4 * df["a"] * df["c"])) / 2 / df["a"]
    )

    if "dt_orig" in df.columns:
        df["dt"] = df["dt_orig"].copy()
    else:
        df["dt_orig"] = df["dt"].copy()
    df["dt"] = df["dt"] + df["t_alpha"]

    # correct for detector timing
    if not simulation:
        for k, v in z_align.items():
            mask = df["channel"] == det_in_meta[k]["channel"]
            if mask.sum() > 0:
                df.loc[mask, "dt"] = df.loc[mask, "dt"] - float(v)

    # define unit vector for direction of alpha particle
    # (we already call it vn for neutron, since that's what we need in a bit)
    df["vn_x"] = (df["X_alpha"] - df["t_alpha"] * v_com_vector[0]) / df["t_alpha"]
    df["vn_y"] = (df["Y_alpha"] - df["t_alpha"] * v_com_vector[1]) / df["t_alpha"]
    df["vn_z"] = (df["Z_alpha"] - df["t_alpha"] * v_com_vector[2]) / df["t_alpha"]

    # calculate time of flight of neutron
    # we get this from the following equations
    #
    # N is the xyz location of the neutron interaction
    # D is the location of the gamma detector
    # G is the vector pointing from N to G
    # vn is the velocity vector of the neutron
    # v_com is the center of mass velocity
    # c is the speed of light (=v_gamma)
    #
    # 1: dt = dt_measured + t_alpha = t_neutron + t_gamma
    # 2: G = D-N
    # 3: |G| = t_gamma * c
    # 4: N = t_n *(v_n + v_com)
    #
    # we already took care of t_alpha by redefining dt, so eq1 reads
    # 1:  dt = t_neutron +  t_gamma
    #
    # from the rest we get:
    #  |D - t_n (v_n + v_com)|^2 = (dt-t_n)^2 c^2
    # =>
    #  c^2 dt^2 - 2 c^2 dt t_n + c^2 t_n^2 = D^2 - 2 <D * (v_n+v_com)> t_n + t_n^2 (v_n+v_com)^2
    #
    # this is a simple quadratic equation with two solutions.
    #
    # t_neutron = [-b +- sqrt(b^2 -4ac)]/(2a)
    # We are looking for the the solution with the "-" sqrt (TODO: double check, pytest seems to indicate this is true)
    #
    #
    # for helpers we define some new columns that we delete later

    # change to the direction and speed of the neutron and add v_com, since that is how they always show up
    df["vn_x"] = -df["vn_x"] / v_alpha * v_neutron + v_com_vector[0]
    df["vn_y"] = -df["vn_y"] / v_alpha * v_neutron + v_com_vector[1]
    df["vn_z"] = -df["vn_z"] / v_alpha * v_neutron + v_com_vector[2]

    # save position in df
    if not simulation:
        df["locX"] = np.nan
        df["locY"] = np.nan
        df["locZ"] = np.nan
        for k, v in meta["detectors"].items():
            mask = df["channel"] == meta["detectors"][k]["channel"]
            if mask.sum() > 0:
                df.loc[mask, "locX"] = v["position"][0]
                df.loc[mask, "locY"] = v["position"][1]
                df.loc[mask, "locZ"] = v["position"][2]
        mask = df["locX"] == np.nan
        if mask.sum() > 0:
            print(
                "[red]ERROR[/] calcXYZ: Undefined locations in df. This should not happen"
            )
            sys.exit()

    df["a"] = SC.c * SC.c - (
        df["vn_x"] * df["vn_x"] + df["vn_y"] * df["vn_y"] + df["vn_z"] * df["vn_z"]
    )
    df["b"] = -(
        2 * SC.c * SC.c * df["dt"]
        - 2
        * (df["locX"] * df["vn_x"] + df["locY"] * df["vn_y"] + df["locZ"] * df["vn_z"])
    )
    df["c"] = SC.c * SC.c * df["dt"] * df["dt"] - (
        df["locX"] * df["locX"] + df["locY"] * df["locY"] + df["locZ"] * df["locZ"]
    )

    df["t_neutron"] = (
        (-df["b"] - np.sqrt(df["b"] * df["b"] - 4 * df["a"] * df["c"])) / 2 / df["a"]
    )

    # calculate position
    df["X"] = df["vn_x"] * df["t_neutron"]
    df["Y"] = df["vn_y"] * df["t_neutron"]
    df["Z"] = df["vn_z"] * df["t_neutron"]

    if cleanup:
        cleanup_dataframe(df)
    return df


def calcXYZ_DuckDB(data: DataManager, meta, cleanup: bool = True, simulation: bool = False, ion_energy: float = 0.0):
    """Modified a duckDB table to calculate XYZ positions.

    The table should have a 'dt', 'X_alpha', and 'Y_alpha' column
    where dt is the measured difference in time between the alpha and
    the gamma x, y are the xy coordinates on the alpha detectors, from
    e.g. X=(sum of two corners)/(sum of four corners) that is, these
    are between 0 and 1

    It also should have a channel column indicating what channel was used.

    This modifies the table in place, so it doesn't have to be returned.

    Parameters
    ----------

    data: ROOTS.data.DataManager
       Holds duckdb connection with the table of interest
    meta : ROOTS.metadata.Metadata
       Holds the detector position and other information
    cleanup
       Flag to remove intermediate steps in the calculation, for debugging only
    simulation
       Skip offset correction for simulated data
    ion_energy
       Beam energy in keV
    
    """
    # (TODO: Eventually replace original calcXYZ function with DuckDB)

    if data.column_exists("surface"):
        simulation = True
    elif not data.column_exists("dt"):
        print("dt does not exist")
    elif not data.column_exists("X_alpha"): 
        print("X_alpha does not exist")
    elif not data.column_exists("Y_alpha"):
        print("Y_alpha does not exist")
    elif not data.column_exists("File time"):
        print("File time does not exist")
    

    # use an abritraty timestamp to get the detector locations
    scintillator_dz = meta["scintillator distance"]
    v_com_direction = np.array(meta["ion beam axis"])

    E, v_cm = np.load(resources.files("ROOTS") / "Data" / "center-of-mass.npy")
    v_com = np.interp(ion_energy, xp=E, fp=v_cm, left=0)
    v_com_vector = v_com * v_com_direction

    data.connection.execute("ALTER TABLE data ADD COLUMN Z_alpha FLOAT")
    data.connection.execute(f"UPDATE data SET Z_alpha = {scintillator_dz}")

    data.connection.execute("ALTER TABLE data ADD COLUMN a1 FLOAT")
    data.connection.execute(f"UPDATE data SET a1 = {v_alpha**2 - v_com**2}")

    data.connection.execute("ALTER TABLE data ADD COLUMN b1 FLOAT")
    data.connection.execute(f"""UPDATE data SET b1 = 2 * ( X_alpha * {v_com_vector[0]} + Y_alpha * {v_com_vector[1]} + Z_alpha * {v_com_vector[2]})""")

    data.connection.execute("ALTER TABLE data ADD COLUMN c1 FLOAT")
    data.connection.execute("UPDATE data SET c1 = -(X_alpha * X_alpha + Y_alpha * Y_alpha + Z_alpha * Z_alpha)")

    data.connection.execute("ALTER TABLE data ADD COLUMN t_alpha FLOAT")
    data.connection.execute("UPDATE data SET t_alpha = (-b1 + sqrt(b1 * b1 - 4 * a1 * c1))/(2 * a1)")

    if data.column_exists("dt_orig"):
        data.connection.execute("UPDATE data SET dt = dt_orig")
    else:
        data.connection.execute("UPDATE data SET dt_orig = dt")
    data.connection.execute("UPDATE data SET dt = dt + t_alpha")


    if not simulation:
        for k, v in meta['z-align'].items():
            data.connection.execute(f"""
                UPDATE data
                SET dt = dt - {float(v)}
                WHERE channel = {meta['detectors'][k]['channel']}
            """)

    data.connection.execute("ALTER TABLE data ADD COLUMN vn_x FLOAT")
    data.connection.execute("ALTER TABLE data ADD COLUMN vn_y FLOAT")
    data.connection.execute("ALTER TABLE data ADD COLUMN vn_z FLOAT")
    data.connection.execute(f"UPDATE data SET vn_x = (X_alpha - t_alpha * {v_com_vector[0]}) / t_alpha")
    data.connection.execute(f"UPDATE data SET vn_y = (Y_alpha - t_alpha * {v_com_vector[1]}) / t_alpha")
    data.connection.execute(f"UPDATE data SET vn_z = (Z_alpha - t_alpha * {v_com_vector[2]}) / t_alpha")

    data.connection.execute(f"UPDATE data SET vn_x = -vn_x / {v_alpha} * {v_neutron} + {v_com_vector[0]}")
    data.connection.execute(f"UPDATE data SET vn_y = -vn_y / {v_alpha} * {v_neutron} + {v_com_vector[1]}")
    data.connection.execute(f"UPDATE data SET vn_z = -vn_z / {v_alpha} * {v_neutron} + {v_com_vector[2]}")

    if not simulation:
        data.connection.execute("UPDATE data SET locX = NULL")
        data.connection.execute("UPDATE data SET locY = NULL")
        data.connection.execute("UPDATE data SET locZ = NULL")
        for k, v in meta["detectors"].items():
            data.connection.execute(f"""
                UPDATE data
                SET locX = {v['position'][0]},
                locY = {v['position'][1]},
                locZ = {v['position'][2]},
                WHERE channel = {meta['detectors'][k]['channel']}
            """)
        null_count = data.connection.execute("SELECT COUNT(*) FROM data WHERE locX IS NULL").fetchone()[0]
        if null_count > 0:
            print("[red]ERROR[/] calcXYZ: Undefined locations in df. This should not happen")
            # sys.exit()

    data.connection.execute(f"UPDATE data SET a1 = {SC.c * SC.c} - (vn_x * vn_x + vn_y * vn_y + vn_z * vn_z)")
    data.connection.execute(f"UPDATE data SET b1 = -(2 * {SC.c * SC.c} * dt - 2 * (locX * vn_x + locY * vn_y + locZ * vn_z))") 
    data.connection.execute(f"UPDATE data SET c1 = ({SC.c * SC.c}) * dt * dt - (locX * locX + locY * locY + locZ * locZ)") 
    data.connection.execute("ALTER TABLE data ADD COLUMN t_neutron FLOAT")
    data.connection.execute("UPDATE data SET t_neutron = ((-b1 - sqrt(b1 * b1 - 4 * a1 * c1 ))/ 2 / a1) ")
    data.connection.execute("UPDATE data SET X = vn_x * t_neutron, Y = vn_y * t_neutron, Z = vn_z * t_neutron")


def cleanup_dataframe(df):
    for col in [
        "x",
        "y",
        "s",
        "phi",
        "R",
        "dx",
        "dy",
        "new_x",
        "new_y",
        "a",
        "b",
        "c",
        "t_neutron",
        "vn_x",
        "vn_y",
        "vn_z",
        "Z_alpha",
        "locX",
        "locY",
        "locZ",
    ]:
        if col in df:
            df.drop(inplace=True, columns=[col])
