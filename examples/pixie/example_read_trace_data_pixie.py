"""
Example read binary data from PIXIE
"""
from wara import helper_api
import matplotlib.pyplot as plt

date = "2025-04-18"
runnr = 8
df = helper_api.read_trace_data(date=date, runnr=runnr)

plt.figure()
plt.plot(df.trace.iloc[0])