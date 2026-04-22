"""
Example use of tlist module.
The following example comes from an experiment with a pulsed DT neutron generator
and a CeBr detector. The pulse period is 1000 us
"""

from wara import tlist
from wara import file_reader

file = "../data/gui_test_data_tlist.txt"
ms = file_reader.ReadMultiScanTlist(file)
ms.read_file()
df = ms.df
png = tlist.Tlist(df_raw=df, period=1000)

png.plot_time_hist()
