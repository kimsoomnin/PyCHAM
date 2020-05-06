'''module to track particle number size distribution using moving centre size structure (p. 416 of Jacobson 2000)'''

import numpy as np
import ipdb
import scipy.constants as si

def mov_cen_main(n0, s0, Cn, rho, sbn, nc, MW, x, Vol0, t, tinc_count, MV):



	# ---------------------------------------------------------------
	# input:
	
	# n0 - particle number concentration per size bin before time step
	# (# particle/cc (air)) (excluding wall)
	# s0 - volume bounds per size bin (um3) (number of size bins +1) (molecules/cc (air))
	# Cn - particle phase concentration per component per size bin
	# (molecules/cc (air)), with rows representing
	# components and columns size bins (including wall as final size bin)
	# rho - particle phase component densities (g/cc (particle))
	# sbn - number of size bins
	# nc - number of components
	# MW - molar weight of components (g/mol)
	# x - original particle size bin radii (cm)
	# Vol0 - original volume of size bins (um3) (excluding wall)
	# t - integration time (s)
	# tinc_count - count on number of time steps since time interval last required 
	# decreasing
	# MV - molar volume (cc/mol)
	# ---------------------------------------------------------------
	# output:
	
	# n1 - end of time step particle number concentration per size bin
	# (# particle/cc (air))
	# m1 - end of time step mass per size bin (g/m3 (air))
	# rad - new radius (um)
	# redt - flag to say whether time step needs reducing due to excess size bin changes
	# tnew - integration time to use on next step
	# ---------------------------------------------------------------
	
	NA = si.Avogadro # Avogadro's number (molecules/mol)
		
	Mrho = ((rho*1.0e-12)/MW[:, 0]).reshape(nc, 1) # molar density (mol/um3)
	
	Vnew = np.zeros((sbn))
	ish = n0>1.0e-10
	nmolC = np.zeros((nc, ish.sum()))
	
	# number of moles of each component in a single particle (mol/cc (air))
	nmolC[:,:] = ((Cn[:, ish]/(NA*n0[ish])))
	MVrep = np.repeat(MV, nmolC.shape[1], axis=1)

	# new volume of single particle per size bin (um3) including volume of water
	Vnew[ish] = np.sum(nmolC*MV*1.0e12, axis=0)
	
	Vnew[n0<=1.0e-10] = Vol0[0::][n0<=1.0e-10]
	
	# truth array of size bins where particles have moved up a size bin
	ind_moveup = Vnew>s0[1::]

	ind_moveup2 = np.asarray(np.where(ind_moveup==1))+1 # ready for excess growth check

	# truth array of size bins where particles have moved down a size bin
	ind_movedn = Vnew<s0[0:-1]
	# ensure we consider only size bins with particles inside
	ind_movedn = (Vnew>0.0)*ind_movedn
	ind_movedn2 = np.asarray(np.where(ind_movedn==1))-1 # ready for excess shrink check
		
	
	
	# return if particles have grown too large (beyond uppermost boundary)
	if (ind_moveup2>(sbn-1)).sum()>0:
		
		print('largest particles exceed uppermost volume bound')
		print('Vnew = ', Vnew)
		print('s0 = ', s0)
		print('n0 = ', n0)
		redt = 1
		tnew = t/2.0 # new time for integration on next step (s)
		
		# prepare for output
		if len(n0)>1:
			n0 = np.squeeze(n0)
		else: # if just one size bin present
			holder = n0
			n0 = np.zeros((1))
			n0[0] = holder

		return(0, 0, 0, 0, 0, 0, 0, 0)
		
	
	
	# matrix of components and array of particle numbers that stay in same size bin
	num_molec_stay = np.zeros((nc, sbn))
	num_molec_stay[:, (ind_moveup+ind_movedn)==0] = Cn[:, (ind_moveup+ind_movedn)==0]
	num_part_stay = np.zeros((1, sbn))
	num_part_stay[0, (ind_moveup+ind_movedn)==0] = n0[(ind_moveup+ind_movedn)==0]
		
	
	# matrix of components and array of particle numbers that go up a size bin
	num_molec_up = np.zeros((nc, sbn))
	num_molec_up[:, np.squeeze(ind_moveup2)] = np.squeeze(Cn[:, ind_moveup])
	num_part_up = np.zeros((1, sbn))
	num_part_up[0, np.squeeze(ind_moveup2)] = n0[ind_moveup]
	
	# matrix of components and array of particle numbers that go down a size bin
	num_molec_dn = np.zeros((nc, sbn))
	num_molec_dn[:, np.squeeze(ind_movedn2)] = np.squeeze(Cn[:, ind_movedn])
	num_part_dn = np.zeros((1, sbn))
	num_part_dn[0, np.squeeze(ind_movedn2)] = n0[ind_movedn]
	
	# combine above matrices to complete rebinning
	num_molec_new = num_molec_stay+num_molec_up+num_molec_dn # molecules/cc (air)
	num_part_new = num_part_stay+num_part_up+num_part_dn # particle/cc (air)
	
	
	# need to find new volumes of single particles (um3)
	# total volume of components 
	# ((um3 (all particles)/cc (air))/(particle number/cc (air))) 
	# calculation is:
	# divide number of molecules/cc (air) by Na to get moles/cc(air), then 
	# multiply by ug3/mol (MV[:,0]*1.0e12) to get ug3 (of each component)/cc (air),
	# then sum volume of components per size bin to get ug3 (all particles)/cc (air)
	MVrep = np.repeat(MV, num_molec_new.shape[1], axis=1)
	
	Vtot = (np.sum((num_molec_new/6.0221409e+23)*(MVrep*1.0e12), 0)) # (um3)
	isb = num_part_new[0, :]>0.0
	# then divide by particle number (#/cc (air)) to get volume of single particles
	# per size bin (um3)
	Vsing = np.zeros(len(Vtot))
	Vsing[isb] = Vtot[isb]/num_part_new[0, isb]
	isb2 = Vtot
	
	# new radius per size bin (um)
	rad = ((3.0*Vsing)/(4.0*np.pi))**(1.0/3.0)
	
	# if zero particles left in a size bin add a tiny number to prevent dividing by zero
	isb = num_part_new<1.0e-25
	num_part_new[isb] = 1.0e-40 # very low no. particles
	num_molec_new[:, isb[0,:]] = 1.0e-40 # very low concentration of components
	# fill radius array elements for bins without particles with central radii (um)
	rad[0::][isb[0, :]] = x[0::][isb[0, :]]
	# fill volume array elements for bins without particles with central volume (um3)
	Vsing[isb[0,:]] = ((4.0/3.0)*np.pi)*(rad[isb[0,:]]**3.0)
	
	# flag to show no reduction in time step needed
	redt = 0

	tnew = t
	
	# prepare for output by ensuring number of dimensions is one for num_part_new
	if sbn>1:
		num_part_new = np.squeeze(num_part_new)
	else: # if just one size bin present
		holder = num_part_new
		num_part_new = np.zeros((1))
		num_part_new[0] = holder
	
	return(num_part_new, Vsing, np.ravel(np.transpose(num_molec_new)), rad, 
			redt, t, tnew)