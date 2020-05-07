'''code to plot results from temporal resolution tests, please call from inside the GMD_paper/Results folder of PyCHAM'''
# important that we can exemplify the sensitivity of model estimates to the resolution of 
# the maximum integration time step and the operator-split time step, therefore this code
# is responsible for recording inputs to tests and plotting corresponding results

import numpy as np
import matplotlib.pyplot as plt
import os

# for moving-centre, the maximum integration time step is adapted depending on the 
# structure's tolerance for volume change - note this is additionally to the 
# integrator's tolerance for concentration changes and therefore maybe redundant

# parameters to assess sensitivity over:
# 128 size bins & 8 size bins, Riverside (polydisperse) distribution, with pconc 
# given in coag_resol_test_res_plot.py
# and unimodal distribution, 
# one equation stating the one semi-volatile component with initial concentration
# above zero and constant concentration thereafter 
# acceptable volume change of 1%, 2%, 4%, 8%, 16%, 32%, 64%, 128%, 256% and 
# changing by one size bin and two size bins after an integration

# 8 size bin, polydisperse --------------------------

# get current working directory - will only work if inside the GMD_paper/Results directory of
# PyCHAM
cwd = os.getcwd() # get current path

output_by_sim = str(cwd + '/tr_tests_data/mov_cen_sens2tr_no_opsplt_chng1sb_128sb_tempconst_seeded')

# name of file where experiment constants saved
fname = str(output_by_sim + '/model_and_component_constants')

const_in = open(fname)
const = {} # prepare to create dictionary
for line in const_in.readlines():

	# convert to python list
	dlist = []
	for i in line.split(',')[1::]:
		if str(line.split(',')[0]) == 'number_of_size_bins':
			dlist.append(int(i))
		if str(line.split(',')[0]) == 'number_of_components':
			dlist.append(int(i))
		if str(line.split(',')[0]) == 'molecular_weights_g/mol_corresponding_to_component_names' or  str(line.split(',')[0]) == 'molecular_volumes_cm3/mol':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			dlist.append(float(i))
		if str(line.split(',')[0]) == 'component_names':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			i = i.strip('\'')
			dlist.append(str(i))
		if str(line.split(',')[0]) == 'factor_for_multiplying_ppb_to_get_molec/cm3_with_time':
			i = i.strip('\n')
			i = i.strip('[')
			i = i.strip(']')
			i = i.strip(' ')
			dlist.append(float(i))
			
	const[str(line.split(',')[0])] = dlist

num_sb = int((const['number_of_size_bins'])[0]) # number of size bins
num_speci = int((const['number_of_components'])[0]) # number of species
# conversion factor to change gas-phase concentrations from molecules/cc 
# (air) into ppb 
Cfactor = const['factor_for_multiplying_ppb_to_get_molec/cm3_with_time']

# withdraw times (s)
fname = str(output_by_sim+'/time')
t_array = np.loadtxt(fname, delimiter=',', skiprows=1)

# withdraw concentrations
fname = str(output_by_sim+'/concentrations_all_components_all_times_gas_particle_wall')
y = np.loadtxt(fname, delimiter=',', skiprows=1)

# withdraw number-size distributions (# particles/cc (air))
fname = str(output_by_sim+'/particle_number_concentration_dry')
N = np.loadtxt(fname, delimiter=',', skiprows=1)

# withdraw size bin bounds, represented by radii (um)
fname = str(output_by_sim+'/size_bin_bounds')
sbb = np.loadtxt(fname, delimiter=',', skiprows=1)

# particle sizes (um)
fname = str(output_by_sim+'/size_bin_radius')
x = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header

timehr = t_array/3600.0
# plot of component in gas-, particle- and wall-phases
# plt.plot(timehr, y[:, 1], 'r')
# plt.plot(timehr, (y[:, 4::3].sum(axis=1)-y[:, -2])/Cfactor, 'b')
# plt.plot(timehr, y[:, -2]/Cfactor, 'g')
# plt.plot(timehr, (y[:, 1::3].sum(axis=1))/Cfactor, '--k')
# plt.show()

# dN/dlog10(Dp) at start and finish
dN_start = N[0, :]/(np.log10(sbb[1::]*2.0)-np.log10(sbb[0:-1]*2.0))
dN_finis = N[-1, :]/(np.log10(sbb[1::]*2.0)-np.log10(sbb[0:-1]*2.0))

# number of points in moving average
np_mvav = 3
dN_start_av = np.zeros((num_sb-np_mvav)) # dN/dlog10(Dp) moving average
rad_start = np.zeros((num_sb-np_mvav)) # particle centre (um) moving average

dN_finis_av = np.zeros((num_sb-np_mvav)) # dN/dlog10(Dp) moving average
rad_finis = np.zeros((num_sb-np_mvav)) # particle centre (um) moving average

for i in range(np_mvav):

	if np_mvav-(i+1)>0:
		# moving average number concentration
		dN_start_av += (dN_start[i:-(np_mvav-(i+1))])/np_mvav
		dN_finis_av += (dN_finis[i:-(np_mvav-(i+1))])/np_mvav
		# moving average particle centre (um)
		rad_start += x[0, i:-(np_mvav-(i+1))]/np_mvav
		rad_finis += x[-1, i:-(np_mvav-(i+1))]/np_mvav
	else: # reach the inclusion of the final element of the original array
		dN_start_av += (dN_start[i::])/np_mvav
		dN_finis_av += (dN_finis[i::])/np_mvav
		rad_start += x[0, i::]/np_mvav
		rad_finis += x[-1, i::]/np_mvav


# plot of number-size distribution
plt.loglog(rad_start*2.0, dN_start_av, 'k')
plt.loglog(rad_finis*2.0, dN_finis_av, '--r')
plt.show()

# plot of water-concentration
yp = y[:, 3:-3] # just particle-phase concentrations
plt.semilogy(yp[:, 1::3].sum(axis=1), 'xk')
plt.semilogy(y[:, 1]*2.0, 'or')
plt.semilogy(y[:, -2]*2.0, '+g')
plt.show()

# plot of radii
plt.semilogy(x[-1, :], 'xk')
plt.semilogy(rad_finis, 'or')
plt.show()


