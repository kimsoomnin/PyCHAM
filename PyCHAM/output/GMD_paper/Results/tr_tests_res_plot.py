'''code to plot results from temporal resolution tests, please call from inside the GMD_paper/Results folder of PyCHAM'''
# important that we can exemplify the sensitivity of model estimates to the resolution of 
# the maximum integration time step and the operator-split time step, therefore this code
# is responsible for recording inputs to tests and plotting corresponding results

import numpy as np
import matplotlib.pyplot as plt
import os
from retrieve_PyCHAM_outputs import retrieve_outputs as retr

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

# 128 size bin, polydisperse, no partitioning --------------------------

# get current working directory - assumes module called inside the GMD_paper/Results 
# directory of PyCHAM
cwd = os.getcwd()

# outputs from 128 size bin, no partitioning with variable operator-split time step and
# only coagulation affecting particles


# empty dictionary of results from each simulation
num_sb_dict = {'num_sb0' : [], 'num_sb1' : [], 'num_sb2' : [], 'num_sb3' : []}
num_speci_dict = {'num_speci0' : [], 'num_speci1' : [], 'num_speci2' : [], 'num_speci3' : []}
Cfac_dict = {'Cfac0' : [], 'Cfac1' : [], 'Cfac2' : [], 'Cfac3' : []}
y_dict = {'y0' : [], 'y1' : [], 'y2' : [], 'y3' : []}
N_dict = {'N0' : [], 'N1' : [], 'N2' : [], 'N3' : []}
sbb_dict = {'sbb0' : [], 'sbb1' : [], 'sbb2' : [], 'sbb3' : []}
x_dict = {'x0' : [], 'x1' : [], 'x2' : [], 'x3' : []}
thr_dict = {'thr0' : [], 'thr1' : [], 'thr2' : [], 'thr3' : []}

(num_sb_dict['num_sb0'], num_speci_dict['num_speci0'], Cfac_dict['Cfac0'], y_dict['y0'], N_dict['N0'], sbb_dict['sbb0'], x_dict['x0'], thr_dict['thr0']) = retr(str(cwd + '/tr_tests_data/mov_cen_sens2tr_no_opsplt_128sb_tempconst_seeded'))
(num_sb_dict['num_sb1'], num_speci_dict['num_speci1'], Cfac_dict['Cfac1'], y_dict['y1'], N_dict['N1'], sbb_dict['sbb1'], x_dict['x1'], thr_dict['thr1']) = retr(str(cwd + '/tr_tests_data/mov_cen_sens2tr_12hr_opsplt_128sb_tempconst_seeded'))
(num_sb_dict['num_sb2'], num_speci_dict['num_speci2'], Cfac_dict['Cfac2'], y_dict['y2'], N_dict['N2'], sbb_dict['sbb2'], x_dict['x2'], thr_dict['thr2']) = retr(str(cwd + '/tr_tests_data/mov_cen_sens2tr_6hr_opsplt_128sb_tempconst_seeded'))
(num_sb_dict['num_sb3'], num_speci_dict['num_speci3'], Cfac_dict['Cfac3'], y_dict['y3'], N_dict['N3'], sbb_dict['sbb3'], x_dict['x3'], thr_dict['thr3']) = retr(str(cwd + '/tr_tests_data/mov_cen_sens2tr_3hr_opsplt_128sb_tempconst_seeded'))

for resi in range(len(num_sb_dict)):
	N = N_dict[str('N' + str(resi))]
	sbb = sbb_dict[str('sbb' + str(resi))]
	num_sb = num_sb_dict[str('num_sb' + str(resi))]
	x = x_dict[str('x' + str(resi))]
	
	# dN/dlog10(Dp) at start and finish
	dN = np.zeros((N.shape[0], N.shape[1]))
	for ti in range(N.shape[0]):
		dN[ti, :] = N[ti, :]/(np.log10(sbb[1::]*2.0)-np.log10(sbb[0:-1]*2.0))
	
	# number of points in moving average
	np_mvav = 3
	dN_av = np.zeros((dN.shape[0], num_sb-np_mvav)) # dN/dlog10(Dp) moving average
	# diameter at size bin centre moving average (um)
	D_av = np.zeros((1, num_sb-np_mvav))
	
	for i in range(np_mvav): # loop through number concentration points to get average
	
		# moving average number concentration
		dN_av[:, :] += (dN[:, i:(num_sb-1)-(np_mvav-(i+1))])/np_mvav
		# diameter (um) at centre of moving average size bins
		D_av[0, :] += x[0, i:(num_sb-1)-(np_mvav-(i+1))]*2.0/np_mvav
	
	if resi == 0:
		# remember benchmark values
		dN_av_bench = dN_av
		continue # no need to plot root-mean square deviation of benchmark against itself
	# plot of root-mean square deviation
	plt.loglog(D_av[0, :], ((((dN_av_bench-dN_av)**2.0).sum(axis=0))/dN_av.shape[0])**0.5, label=str(resi))

plt.legend()
plt.show()