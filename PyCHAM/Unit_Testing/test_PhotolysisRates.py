'''module for testing the PhotolysisRates.py module of PyCHAM, needs to be called when inside the Unit_Testing folder'''
# will test the output of the PhotolysisRates.py module when MCM photoloysis rates
# are being estimated using natural sunlight source

# import the module to be tested
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
dirpath = os.getcwd() # get current path
sys.path.append(str(dirpath+'/PyCHAM')) # add path to system path
import PhotolysisRates


time_array = np.linspace(0.0, 86400.0, 24*6) # array of times (s)
lat = 51.51
lon = 0.13
TEMP = 298.15
act_flux_path = 'no'
DayOfYear = 171
photo_par_file = str(dirpath + '/PyCHAM/photofiles/MCMv3.2')
Jlen = 62

res = np.zeros((Jlen, len(time_array))) # empty results array
count = 0 # count on times
for time in time_array: # loop through times
	res[:, count] = PhotolysisRates.PhotolysisCalculation(time, lat, lon, TEMP, 
						act_flux_path, DayOfYear, photo_par_file, Jlen)
	count += 1

for i in range(Jlen):
	plt.semilogy(time_array/3600.0, res[i, :], label=str(i))
plt.legend()
plt.show()