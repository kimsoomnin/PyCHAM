'''function to initiate concentrations of components, obtain MCM reaction rate constants and produce the reaction coefficient file'''

import numpy as np
import eqn_parser
import scipy.constants as si
import math
from water_calc import water_calc

def init_conc_func(num_speci, Comp0, init_conc, TEMP, RH, 
					reac_coef, filename, PInit, time, lat, lon, Pybel_objects,
					testf, pconc, act_flux_path, dydt_trak, end_sim_time, save_step, 
					rindx, pindx, num_eqn, nreac, nprod, DayOfYear, 
					spec_namelist, Compt, seed_name, const_comp, const_infl, seed_mw,
					core_diss, nuc_comp):
		
	# -----------------------------------------------------------
	# inputs:
	
	# Comp0 - chemical scheme names of components present at start of experiment
	# TEMP - temperature in chamber at start of experiment (K)
	# RH - relative humidity in chamber (dimensionless fraction 0-1)
	# PInit - initial pressure (Pa)
	# init_SMIL - SMILES of components present at start of experiment (whose 
	# concentrations are given in init_conc)
	# testf - flag for whether in normal mode (0) or testing mode (1/2)
	# pconc - initial concentration of particles (# particles/cc (air))
	# act_flux_path - path to actinic flux file for indoor photolysis, 
	# 					'no' if none provided
	# dydt_trak - chemical scheme name of components for which user wants the tendency to  
	#			change tracked
	# end_sim_time - total simulation time (s)
	# save_step - recording frequency (s)
	# num_eqn - number of equations
	# DayOfYear - day of year for natural light calculation (integer 1-365)
	# spec_namelist - list of components' names in chemical equation file
	# Compt - name of component injected after start of experiment
	# seed_name - name of core component (input by user)
	# const_comp - names of components with constant gas-phase concentration
	# const_infl - names of components with constant influx
	# seed_mw - molecular weight of seed material (g/mol)
	# core_diss - dissociation constant of seed material
	# nuc_comp - name of nucleating component (input by user, or defaults to 'core')
	# -----------------------------------------------------------

	if testf==1: # testing mode
		# return dummies
		return(0,0,0,0,0,0,0,0)

	NA = si.Avogadro # Avogadro's number (molecules/mol)
	# empty array for storing species' concentrations, must be an array
	y = np.zeros((num_speci))
	y_mw = np.zeros((num_speci, 1)) # species' molecular weight (g/mol)
	# empty array for storing index of interesting gas-phase components
	y_indx_plot = []
	
	# convert concentrations
	# total number of molecules in 1 cc air using ideal gas law.  R has units cc.Pa/K.mol
	ntot = PInit*(NA/(8.3144598e6*TEMP))
	# one billionth of number of molecules in chamber unit volume
	Cfactor = ntot*1.0e-9 # ppb-to-molecules/cc
	

	# prepare dictionary for tracking tendency to change of user-specified components
	dydt_vst = {}

	# insert initial concentrations where appropriate
	for i in range (len(Comp0)):
    	# index of where initial species occurs in list of components
		y_indx = spec_namelist.index(Comp0[i])
		y[y_indx] = init_conc[i]*Cfactor # convert from ppb to molecules/cc (air)
		# remember index for plotting gas-phase concentrations later
		y_indx_plot.append(y_indx)
		
		
	# empty array for storing index of components with constant concentration
	const_compi = []
	for i in range (len(const_comp)):
		# index of where constant components occur in list of components
		y_indx = spec_namelist.index(const_comp[i])
		const_compi.append(y_indx) # remember their index
	
	# empty array for storing index of components with constant influx
	const_infli = []
	num_const_infl = len(const_infl)
	const_infli = np.zeros((num_const_infl))

	for i in range (num_const_infl):
		# index of where constant components occur in list of components
		y_indx = spec_namelist.index(const_infl[i])
		const_infli[i] = y_indx # remember their index
		

	# get index of user-specified components for tracking their dydt due to model 
	# mechanisms
	if len(dydt_trak)>0:
		
		dydt_traki = [] # empty list for indices of these components
		# total number of recording steps (same as for other recorded metrics)
		num_rec_steps = math.ceil(end_sim_time/save_step)
		
		for i in range (len(dydt_trak)):
			reac_index = [] # indices of reactions involving this species
			# index of where initial species occurs in SMILE string
			y_indx = spec_namelist.index(dydt_trak[i])

			# remember index for plotting gas-phase concentrations later
			dydt_traki.append(int(y_indx))
			# search through reactions to see where this component is reactant or product
			for ri in range(num_eqn):
				if sum(rindx[ri,0:nreac[ri]]==y_indx)>0:
					reac_index.append(int(ri)) # append reaction index
			for ri in range(num_eqn): # repeat for products
				if sum(pindx[ri,0:nprod[ri]]==y_indx)>0:
					reac_index.append(int(ri)) # append reaction index
					
	
			# save reaction indices in dictionary value for this component,
			# when creating empty rec_array, add two rows onto the end for particle- and 
			# wall-partitioning, respectively
			rec_array = np.zeros((num_rec_steps+2, len(reac_index)+2))
			rec_array[0,0:-2] = (reac_index)
			dydt_vst[y_indx] = rec_array

		dydt_vst['comp_index'] = dydt_traki
		
		# call on write_dydt_rec in eqn_parser to generate the module that will process
		# the tendency to change during the simulation
		eqn_parser.write_dydt_rec()
	
	for i in range(num_speci): # loop through all species
		y_mw[i] = Pybel_objects[i].molwt # molecular weight (g/mol)
	
	# ------------------------------------------------------------------------------------
	# account for water's properties
	H2Oi = num_speci # index for water
	num_speci += 1 # update number of species to account for water
	
	# update gas-phase concentration (molecules/cc (air)) and vapour pressure
	# of water (log10(atm))
	[C_H2O, Psat_water, H2O_mw] = water_calc(TEMP, RH, 6.02214129e+23)
	
	# append empty element to y and y_mw to hold water values
	y = np.append(y, C_H2O)
	y_mw = (np.append(y_mw, H2O_mw)).reshape(-1, 1)
	spec_namelist.append('H2O') # append water's name to component name list

	# ------------------------------------------------------------------------------------
	# account for seed properties - note that even if no seed particle, this code ensures
	# that an index is provided for core material
	
	# if seed particles present and made of a 'core' material
	if sum(sum(pconc))>0.0 and seed_name == 'core':
		# append core gas-phase concentration (molecules/cc (air)) and molecular 
		# weight (g/mol) (needs to have a 1 length in second dimension for the kimt 
		# calculations)
		y = np.append(y, 1.0e-40) 
		y_mw = (np.append(y_mw, seed_mw)).reshape(-1, 1)
		corei = num_speci # index of core component
		num_speci += 1 # update number of species to account for core material
		spec_namelist.append('core') # append core's name to component name list
	# if nucleating component formed of core component
	if nuc_comp[0] == 'core':
		if sum(sum(pconc))>0.0 and seed_name == 'core':
			nuci = corei
		else:
			y = np.append(y, 1.0e-40) 
			y_mw = (np.append(y_mw, seed_mw)).reshape(-1, 1)
			nuci = num_speci # index of core component
			num_speci += 1 # update number of species to account for core material
			spec_namelist.append('core') # append core's name to component name list
	else:
		nuci = -1 # filler
		
	# if seed particles made of a non-'core' material
	if seed_name != 'core':
		# append core gas-phase concentration (molecules/cc (air)) and molecular weight 
		# (g/mol) (needs to have a 1 length in second dimension for the kimt calculations)
		corei = spec_namelist.index(seed_name) # index of core component
	if sum(sum(pconc)) == 0.0: # no seed particle case
		corei = -1 # filler
		core_diss = 1.0 # ensure no artefact in Raoult term due to this filler
	
	# get index of component with latter injections
	if len(Compt)>0:
		inj_indx = np.zeros((len(Compt)))
		for i in range(len(Compt)):
			# index of where initial species occurs in SMILE string
			inj_indx[i] = spec_namelist.index(Compt[i])
	else:
		inj_indx = np.zeros((1)) # dummy
	# ensure inj_indx is integer type
	inj_indx = inj_indx.astype('int')
	
	return (y, H2Oi, y_mw, num_speci, Cfactor, y_indx_plot, corei, dydt_vst, 
				spec_namelist, inj_indx, const_compi, const_infli, core_diss,
				Psat_water, nuci)