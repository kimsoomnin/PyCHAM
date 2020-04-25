'''code to make plot that illustrates effect of temporal resolution on gas-phase photochemistry using the alpha-pinene ozonolysis in presence of NOx simulation using for photo_chem_res_plot'''
import numpy as np
import sys
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# PyCHAM with 60 s, computer time: 1805.663383 s
# file name
Pyfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/paper_GMD/GMD_paper/Results/photo_chem_data/PyCHAM_time_res/PyCHAM_time_res60s'

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
		if str(line.split(',')[0]) == 'Cfactor':
			dlist.append(float(i))
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
Cfactor = const['Cfactor']
PyCHAM_names = const['spec_namelist']

# name of file where concentration (molecules/cc (air)) results saved
fname = str(Pyfname+'/y')
y60 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw times
fname = str(Pyfname+'/t')
t_array60 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# ----------------------------------------------------------------------------------------
# PyCHAM with 600 s, computer time: 205.249932 s
# file name
Pyfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/paper_GMD/GMD_paper/Results/photo_chem_data/PyCHAM_time_res/PyCHAM_time_res600s'

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
		if str(line.split(',')[0]) == 'Cfactor':
			dlist.append(float(i))
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
Cfactor = const['Cfactor']
PyCHAM_names = const['spec_namelist']

# name of file where concentration (molecules/cc (air)) results saved
fname = str(Pyfname+'/y')
y600 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw times
fname = str(Pyfname+'/t')
t_array600 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# ----------------------------------------------------------------------------------------
# PyCHAM with 6000 s, computer time: 45.661775000000006 s
# file name
Pyfname = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/paper_GMD/GMD_paper/Results/photo_chem_data/PyCHAM_time_res/PyCHAM_time_res6000s'

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
		if str(line.split(',')[0]) == 'Cfactor':
			dlist.append(float(i))
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
Cfactor = const['Cfactor']
PyCHAM_names = const['spec_namelist']

# name of file where concentration (molecules/cc (air)) results saved
fname = str(Pyfname+'/y')
y6000 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw times
fname = str(Pyfname+'/t')
t_array6000 = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)
# ----------------------------------------------------------------------------------------

# make plot with all gas-phase concentrations shown
compnum = 0 # count on components
fig, (ax0) = plt.subplots(1, 1, figsize=(8,6))


ax0.plot(t_array600/3600.0, (y600[:,312]-y60[0::10,312])/np.max(np.abs(y60[0::10,312])), label=r'$\mathrm{\alpha}$-pinene $\mathrm{6x10^{2}\, s}$')
ax0.plot(t_array6000[0:-1]/3600.0, (y6000[0:-1,312]-y60[0::100,312])/np.max(np.abs(y60[0::100,312])), label=r'$\mathrm{\alpha}$-pinene $\mathrm{6x10^{3}\, s}$')
ax0.plot(t_array600/3600.0, (y600[:,1]-y60[0::10,1])/np.max(np.abs(y60[0::10,1])), label=r'$\mathrm{O_3}\, \mathrm{6x10^{2}\, s}$')
ax0.plot(t_array6000[0:-1]/3600.0, (y6000[0:-1,1]-y60[0::100,1])/np.max(np.abs(y60[0::100,1])), label=r'$\mathrm{O_3}\, \mathrm{6x10^{3}\, s}$')
ax0.plot(t_array600/3600.0, (y600[:,7]-y60[0::10,7])/np.max(np.abs(y60[0::10,7])), label=r'$\mathrm{OH}\, \mathrm{6x10^{2}\, s}$')
ax0.plot(t_array6000[0:-1]/3600.0, (y6000[0:-1,7]-y60[0::100,7])/np.max(np.abs(y60[0::100,7])), label=r'$\mathrm{OH}\, \mathrm{6x10^{2}\, s}$')

ax0.set_ylabel(r'Deviation (%)', fontsize=14)
ax0.set_xlabel(r'Time of day (hours)', fontsize=14)
ax0.yaxis.set_tick_params(size=14)
ax0.xaxis.set_tick_params(size=14)
ax0.legend(fontsize=12)
fig.savefig('photo_chem_time_res.png')
plt.show()