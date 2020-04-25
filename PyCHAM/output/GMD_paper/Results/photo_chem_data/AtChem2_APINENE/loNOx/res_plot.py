'''code to plot results from AtChem2 and compare against PyCHAM results'''
import numpy as np
import sys
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# AtChem2 part
# open saved files
Atfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/AtChem2/36env/lib/python3.6/site-packages/AtChem2/model/output/CH4_scheme/speciesConcentrations.output'
inputs = open(Atfname, mode='r') # open results
# read the file and store everything into a list
in_list = inputs.readlines()
inputs.close() # close file
for i in range(len(in_list)):
	if i==0: # obtain headers
		comp_names = [] # empty list
		comp_nam = (in_list[i].split(' ')) # list of component names
		for ii in comp_nam:
			if len(ii.strip())==0: # omit white space
				continue
			else:
				comp_names.append(ii.strip())

		# create empty array to hold results
		gconc = np.empty((len(in_list)-1, len(comp_names)))

	else:
		compC0 = (in_list[i].split(' ')) # list of component concentrations
		comp_num = 0 # number of components at this time
		for ii in compC0:
			if len(ii.strip())==0: # omit white space
				continue
			else:
				gconc[i-1, comp_num] = ii.strip()
				comp_num += 1

# check that first column is time, otherwise throw error and exit
if str(comp_names[0]) != 't' and str(comp_names[0]) != 'time':
	sys.exit('Error, first column is not titled t or time, instead it is called: ' + str(comp_names[0]))
# ----------------------------------------------------------------------------------------
# PyCHAM2 part
# file name
Pyfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/PyCHAM_Gitw/PyCHAM/output/CH4_scheme/AtChem2_1'

# name of file where experiment constants saved (number of size bins and whether wall 
# included)
fname = str(Pyfname+'/constants')
const_in = open(fname)
const = {} # prepare to create dictionary
for line in const_in.readlines():

	# convert to python list
	dlist = []
	for i in line.split(',')[1::]:
		if str(line.split(',')[0]) == 'num_sb':
			dlist.append(int(i))
		if str(line.split(',')[0]) == 'num_speci':
			dlist.append(int(i))
		if str(line.split(',')[0]) == 'mw' or  str(line.split(',')[0]) == 'mv':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			dlist.append(float(i))
		if str(line.split(',')[0]) == 'spec_namelist':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			i = i.strip('\'')
			dlist.append(str(i))
	const[str(line.split(',')[0])] = dlist

num_sb = const['num_sb'] # number of size bins
num_speci = const['num_speci'] # number of species
y_mw = const['mw']
y_MV = const['mv']
PyCHAM_names = const['spec_namelist']

# name of file where concentration (molecules/cc (air)) results saved
fname = str(Pyfname+'/y')
y = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw times
fname = str(Pyfname+'/t')
t_array = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# ----------------------------------------------------------------------------------------
# comparative statistics

frac_dev = np.empty((len(t_array), len(comp_names)-1)) # empty fractional deviation matrix

# if AtChem values don't synchronise with PyCHAM, then interpolate, note we assume the 
# time arrays have the same units
if len(gconc[:,0])!=len(t_array):
	interp_flag = 1
elif np.abs(sum(gconc[:,0]-t_array))>1.0e-3:
	interp_flag = 1
else:
	interp_flag = 0

if interp_flag == 1:
	gconc_int = np.empty((len(t_array), len(comp_names)))
	# adopt PyCHAM times
	gconc_int[:,0] = t_array

	# loop through the component names from the AtChem array
	for i in range( 1, len(comp_names[1::])): # loop through AtChem components:
		# interpolate to PyCHAM times
		gconc_int[:, i] = np.interp(gconc_int[:, 0], gconc[:, 0], gconc[:, i])

else:
	gconc_int = gconc

# empty array for fractional deviation
frac_dev = np.zeros((len(t_array), len(comp_names[1::])))
# loop through the component names from the AtChem array
for i in range(1, len(comp_names[1::])): # loop through AtChem components:
	# find index of corresponding component in PyCHAM results
	ind = PyCHAM_names.index(comp_names[i])
	nz_ind = np.where(gconc_int[:, i]!=0)
	frac_dev[nz_ind, i-1] = ((y[nz_ind, ind]-
							gconc_int[nz_ind, i])/np.max(gconc_int[nz_ind, i]))*100.0



# ----------------------------------------------------------------------------------------
# make plot with all gas-phase concentrations shown
compnum = 0 # count on components
fig, (ax0,ax1) = plt.subplots(2, 1, figsize=(8,6))
for i in comp_names[1::]: # loop through components (excluding time in column 0)
	if str(i)=="CH3O2" or str(i)=="HO2" or str(i)=="O" or str(i)=="CO":
		continue
	ax0.plot(t_array/3600.0, frac_dev[:,compnum], label=str(i))
	compnum += 1
# plt.title('PyCHAM-AtChem2 Fractional Deviation of Gas-phase Concentration')
ax0.set_ylabel(r'Fractional Deviation (%)', fontsize=10)
ax0.yaxis.set_tick_params(size=12)
ax0.xaxis.set_tick_params(size=12)
ax0.legend(fontsize=10)


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# AtChem2 part
# open saved files
Atfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/AtChem2/36env/lib/python3.6/site-packages/AtChem2/model/output/apinene_scheme/speciesConcentrations.output'


inputs = open(Atfname, mode='r') # open results
# read the file and store everything into a list
in_list = inputs.readlines()
inputs.close() # close file
for i in range(len(in_list)):
	if i==0: # obtain headers
		comp_names = [] # empty list
		comp_nam = (in_list[i].split(' ')) # list of component names
		for ii in comp_nam:
			if len(ii.strip())==0: # omit white space
				continue
			else:
				comp_names.append(ii.strip())

		# create empty array to hold results
		gconc = np.empty((len(in_list)-1, len(comp_names)))

	else:
		compC0 = (in_list[i].split(' ')) # list of component concentrations
		comp_num = 0 # number of components at this time
		for ii in compC0:
			if len(ii.strip())==0: # omit white space
				continue
			else:
				gconc[i-1, comp_num] = ii.strip()
				comp_num += 1

# check that first column is time, otherwise throw error and exit
if str(comp_names[0]) != 't' and str(comp_names[0]) != 'time':
	sys.exit('Error, first column is not titled t or time, instead it is called: ' + str(comp_names[0]))

# ----------------------------------------------------------------------------------------
# PyCHAM2 part
# file name
Pyfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/PyCHAM_Gitw/PyCHAM/output/apinene_scheme/AtChem2_1'

# name of file where experiment constants saved (number of size bins and whether wall 
# included)
fname = str(Pyfname+'/constants')
const_in = open(fname)
const = {} # prepare to create dictionary
for line in const_in.readlines():

	# convert to python list
	dlist = []
	for i in line.split(',')[1::]:
		if str(line.split(',')[0]) == 'num_sb':
			dlist.append(int(i))
		if str(line.split(',')[0]) == 'num_speci':
			dlist.append(int(i))
		if str(line.split(',')[0]) == 'mw' or  str(line.split(',')[0]) == 'mv':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			dlist.append(float(i))
		if str(line.split(',')[0]) == 'spec_namelist':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			i = i.strip('\'')
			dlist.append(str(i))
	const[str(line.split(',')[0])] = dlist

num_sb = const['num_sb'] # number of size bins
num_speci = const['num_speci'] # number of species
y_mw = const['mw']
y_MV = const['mv']
PyCHAM_names = const['spec_namelist']

# name of file where concentration (molecules/cc (air)) results saved
fname = str(Pyfname+'/y')
y = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw times
fname = str(Pyfname+'/t')
t_array2 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)
# add starting time of day (s)
t_array2 = t_array2+gconc[0, 0]

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# comparative statistics

frac_dev2 = np.empty((len(t_array2), len(comp_names)-1)) # empty fractional deviation matrix

# if AtChem values don't synchronise with PyCHAM, then interpolate, note we assume the 
# time arrays have the same units
if len(gconc[:,0])!=len(t_array2):
	interp_flag = 1
elif np.abs(sum(gconc[:,0]-t_array2))>1.0e-3:
	interp_flag = 1
else:
	interp_flag = 0

if interp_flag == 1:
	gconc_int = np.empty((len(t_array2), len(comp_names)))
	# adopt PyCHAM times
	gconc_int[:,0] = t_array2

	# loop through the component names from the AtChem array
	for i in range( 1, len(comp_names[1::])): # loop through AtChem components:
		# interpolate to PyCHAM times
		gconc_int[:, i] = np.interp(gconc_int[:, 0], gconc[:, 0], gconc[:, i])

else:
	gconc_int = gconc

# empty array for fractional deviation
frac_dev2 = np.zeros((len(t_array2), len(comp_names[1::])))
# loop through the component names from the AtChem array
for i in range(1, len(comp_names[1::])): # loop through AtChem components:
	# find index of corresponding component in PyCHAM results
	ind = PyCHAM_names.index(comp_names[i])
	nz_ind = np.where(gconc_int[:, i]!=0)
	frac_dev2[nz_ind, i-1] = ((y[nz_ind, ind]-
							gconc_int[nz_ind, i])/np.max(gconc_int[nz_ind, i]))*100.0



# ----------------------------------------------------------------------------------------
# make plot with all gas-phase concentrations shown
compnum = 0 # count on components

for i in comp_names[1::]: # loop through components (excluding time in column 0)
	if str(i)=="CH3O2" or str(i)=="HO2" or str(i)=="O":
		continue
	ax1.plot(t_array2/3600.0, frac_dev2[:,compnum], label=str(i))
	compnum += 1
# plt.title('PyCHAM-AtChem2 Fractional Deviation of Gas-phase Concentration')
ax1.set_ylabel(r'Fractional Deviation (%)', fontsize=10)
ax1.yaxis.set_tick_params(size=12)
ax1.xaxis.set_tick_params(size=12)
ax1.legend(fontsize=10)
ax1.set_xlabel(r'Time of Day (hours)', fontsize=10)

fig.savefig('GasChemVer1.png')
# plt.show()

# plotting individual concentrations
# plt.plot(t_array2/3600.0, ((y[:, 119]-gconc_int[:, 7])),'r')
# plt.plot(t_array2/3600.0, y[:, 119],'r')
# plt.plot(t_array2/3600.0, gconc_int[:, 7],'--b')
# plt.plot(t_array2/3600.0, gconc_int[:, 1],'--b')
plt.show()

# plt.show()