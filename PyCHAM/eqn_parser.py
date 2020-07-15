''' module for parsing the equation file, including converting MCM names of components into SMILES strings - see the README for how to format the equation and xml files'''
# utilising python's inherent ability to readily parse inputs into code, this module
# reads in and prepares the chemical reactions

import os
import re
import collections
import sys
import datetime
import numpy as np
import pybel
import formatting
import xmltodict # for opening and converting xml files to python dictionaries
import ipdb
from eqn_interr import eqn_interr

# ----------Extraction of eqn info----------
# Extract the mechanism information
def extract_mechanism(filename, xmlname, PInit, testf, RH, 
						start_sim_time, lat, lon, act_flux_path, DayOfYear, 
						chem_scheme_markers, photo_par_file):

	
	# inputs: ----------------------------------------------------------------------------
	# testf - flag for operating in normal mode (0) or testing mode (1)
	# chem_scheme_markers - markers for different sections of the chemical scheme, 
	#						default input is for the kinetic pre-processor (KPP) format
	# photo_par_file - path (from PyCHAM home directory) to file containing photolysis
	#					information (absorption cross sections and quantum yields)
	# ------------------------------------------------------------------------------------
	
	if testf == 1: # for just testing mode
		return(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    
    print('Now parsing the equation information ... \n')
    
	# open the chemical scheme file
	f_open_eqn = open(filename, mode='r')
	
	# read the file and store everything into a list
	# reaction rates have units /s
	total_list_eqn = f_open_eqn.readlines()
	f_open_eqn.close()
	if (f_open_eqn.closed == False):
		print('IOError')
		print('Eqn file not closed')
		sys.exit()
	
	naked_list_eqn = [] # empty list for gas-phase equation reactions
	naked_list_peqn = [] # empty list for other equation reactions
	RO2_names = [] # empty list for peroxy radicals
	rrc = [] # empty list for reaction rate coefficients
	rrc_name = [] # empty list for reaction rate coefficient labels

	eqn_flag = 0 # don't collate reaction equations until seen
	pr_flag = 0 # don't collate peroxy radicals until seen
	RO2_count = 0 # count on number of lines considered in peroxy radical list
	
	# obtain lists for reaction rate coefficients, peroxy radicals and equation reactions
	# using markers for separating chemical scheme elements
	for line in total_list_eqn:
		
		line1 = line.strip() # remove bounding white space
		
		# --------------------------------------------------------------------------------
		# generic reaction rate coefficients part
		# look out for start of generic reaction rate coefficients
		# could be generic reaction coefficient if just one = in line
		if len(line1.split('='))==2:

			# dont consider if start of peroxy radical list
			if (line1.split('=')[0]).strip() != chem_scheme_markers[1]:
				# don't consider if a chemical scheme reaction
				if ((line1.split('=')[0]).strip())[0] != chem_scheme_markers[0]:
					# remove end characters
					line2 = line1.replace(str(chem_scheme_markers[6]), '')
					# remove all white space
					line2 = line2.replace(' ', '')
					# convert fortran-type scientific notation to python type
					line2 = formatting.SN_conversion(line2)
					# ensure rate coefficient is python readable
					line2 = formatting.convert_rate_mcm(line2)
					rrc.append(line2.strip())
					# get just name of reaction rate coefficient
					rrc_name.append((line2.split('=')[0]).strip())
			
		# --------------------------------------------------------------------------------
		# peroxy radical part
		# now start logging peroxy radicals
		if (re.match(chem_scheme_markers[1], line1) != None):
			pr_flag = 1
		if RO2_count == float(chem_scheme_markers[5]): # no longer log peroxy radicals
			pr_flag=0
		if (pr_flag==1):
			# get the elements in line separated by peroxy radical separator
			line2 = line1.split(chem_scheme_markers[2])
			RO2_count += 1 # count on number of lines considered in peroxy radical list
			
			for line3 in line2: # loop through elements in line
				if len(line3.split('='))>1: # in case of RO2 = ...
					line3 = (line3.split('='))[1]
				if len(line3.split(';'))>1: # in case of RO2 list finishing with ...;
					line3 = (line3.split(';'))[0]
				if len(line3.split('&'))>1: # in case of RO2 list finishing with &
					line3 = (line3.split('&'))[0]
				# don't include white space or ampersands
				if (line3.strip() == '' or line3.strip() == '&'):
					continue
				else: # if not these, then first strip surrounding marks
					if line3[0:len(chem_scheme_markers[3])] == chem_scheme_markers[3]:
						line3 = line3[len(chem_scheme_markers[3])::]
					if line3[-len(chem_scheme_markers[4]):-1] == chem_scheme_markers[4]:
						line3 = line3[0:-len(chem_scheme_markers[4])]
					RO2_names.append(line3.strip())
		# --------------------------------------------------------------------------------
		# gas-phase reaction equation part
		if (re.match(chem_scheme_markers[0], line1) != None):
			naked_list_eqn.append(line1) # store reaction equations
		# aqueous-phase reaction equation part
		if (re.match(chem_scheme_markers[7], line1) != None):
			naked_list_peqn.append(line1) # store reaction equations
		
			
		# --------------------------------------------------------------------------------
	
	# format the equation list
	
	
	# first loop through equation list to concatenate any multiple lines that hold just
	# one equation, note this depends on a symbol representing the start and end of
	# and equation (provided by chem_scheme_markers)
	for iline in range (0, len(naked_list_eqn)):
		
		# keep track of when we reach end of naked_list_eqn (required as naked_list_eqn
		# may shorten)
		if (iline+1)==len(naked_list_eqn):
			break
			
		# if one equation already on one line
		if (re.match(chem_scheme_markers[0], naked_list_eqn[iline])!=None and re.match(chem_scheme_markers[0], naked_list_eqn[iline+1])!=None):
			continue
		# if spread across more than one line
		else:
			con_count = 1 # count number of lines that need concatenating
			iline2 = iline # index for further lines
			while con_count!=0:
				iline2 += 1 # move onto next line
				naked_list_eqn[iline] = str(naked_list_eqn[iline]+naked_list_eqn[iline2])
				if re.match(chem_scheme_markers[0], naked_list_eqn[iline2+1])!=None:
					# remove concatenated line(s) from original list
					naked_list_eqn = naked_list_eqn[0:iline]+naked_list_eqn[iline2+1::]
					con_count = 0 # finish concatenating
	
	num_eqn = [len(naked_list_eqn), len(naked_list_peqn)] # get number of equations
	
	
	# --open and initialise the xml file for converting chemical names to SMILES-----
	with open(xmlname) as fd:
		doc = xmltodict.parse(fd.read())

	a = doc['mechanism']['species_defs']['species']
	spec_numb = list(('0',) * len(a))
	spec_name = list(('0',) * len(a))
	spec_smil = list(('0',) * len(a))
	
	for i in range(len(a)):
		spec_numb[i] = a[i]['@species_number']
		spec_name[i] = a[i]['@species_name']
		if "smiles" in a[i]:
			spec_smil[i] = a[i]['smiles']
		elif spec_name[i][0]=='O' or spec_name[i][0]=='H':
			 spec_smil[i] = '['+spec_name[i]+']'
		else:
			 spec_smil[i] = spec_name[i] 
    
	species_step = 0 # log the number of unique species
	max_no_reac = 0.0 # log maximum number of reactants in a reaction
	max_no_prod = 0.0 # log maximum number of products in a reaction
	
	species_step = 0 # ready for equation loop
	
	# initialising lists
	
	# matrix to record indices of reactants (cols) in each equation (rows)
	rindx = np.zeros((num_eqn, 1)).astype(int)
	rindx_p = np.zeros((num_eqn, 1)).astype(int)
	# matrix to record indices of products (cols) in each equation (rows)
	pindx = np.zeros((num_eqn, 1)).astype(int)
	pindx_p = np.zeros((num_eqn, 1)).astype(int)
	# matrix to record stoichometries of reactants (cols) in each equation (rows)
	rstoi = np.zeros((num_eqn, 1))
	rstoi_p = np.zeros((num_eqn, 1))
	# matrix to record stoichometries of products (cols) in each equation (rows)
	pstoi = np.zeros((num_eqn, 1))
	pstoi_p = np.zeros((num_eqn, 1))
	# arrays to store number of reactants and products in gas-phase equations
	nreac = np.empty(num_eqn[0], dtype=np.int8)
	nprod = np.empty(num_eqn[0], dtype=np.int8)
	nreac_p = np.empty(num_eqn[1], dtype=np.int8)
	nprod_p = np.empty(num_eqn[1], dtype=np.int8)
	# list for equation reaction rate coefficients
	reac_coef = []
	reac_coef_p = []
	# list for components' SMILE strings
	spec_list = []
	# list of Pybel objects
	Pybel_objects = []
	# a new list for the name strings of species presented in the scheme (not SMILES)
	spec_namelist = []
	
	# get equation information for gas-phase reactions
	[rindx, rstoi, pindx, pstoi, reac_coef, spec_namelist, spec_list, 
			Pybel_objects, nreac, nprod, species_step] = eqn_interr(num_eqn[0], naked_list_eqn, 
				rindx, rstoi, pindx, pstoi, 
				rate_coeff_start_mark[0], reac_coef, spec_namelist, spec_name, spec_smil,
				spec_list, Pybel_objects, species_step, nreac, nprod)
	# get equation information for aqueous-phase reactions
	[rindx_p, rstoi_p, pindx_p, pstoi_p, reac_coef_p, spec_namelist, spec_list, 
			Pybel_objects, nreac_p, nprod_p, species_step] = eqn_interr(num_eqn[1], naked_list_peqn, 
				rindx_p, rstoi_p, pindx_p, pstoi_p, 
				rate_coeff_start_mark[7], reac_coef_p, spec_namelist, spec_name, spec_smil,
				spec_list, Pybel_objects, species_step, nreac_p, nprod_p)
	
	print(rindx_p)
	print(rstoi_p)
	print(pindx_p)
	print(pstoi_p)
	ipdb.set_trace()
	
	if len(spec_list)!=len(spec_namelist):
		sys.exit('Error: inside eqn_parser, length of spec_list is different to length of spec_namelist and the SMILES in the former should align with the chemical scheme names in the latter')	
	
	# number of columns in rindx and pindx
	reacn = rindx.shape[1]
	prodn = pindx.shape[1]  
	
	# create a 2 column array, the first column with the RO2 list index of any RO2 species
	# that appear in the species list, the second column for its index in the species list
	RO2_indices = write_RO2_indices(spec_namelist, RO2_names)
	
	# automatically generate the Rate_coeffs module that will allow rate coefficients to
	# be calculated inside ode_gen module
	# now create reaction rate file (reaction rates are set up to have units /s)
	write_rate_file(reac_coef, rrc, rrc_name, testf)

	# number of photolysis reactions, if this relevant
	cwd = os.getcwd() # address of current working directory
	if photo_par_file == str(cwd + '/PyCHAM/photofiles/MCMv3.2'):
		Jlen = 62 # for MCM (default name of photolysis parameters)
	else: # need to find out number of photolysis reactions
		# use Fortran indexing to be consistent with MCM photochemical reaction numbers
		Jlen = 1 
		# open file to read
		f = open(str(photo_par_file), 'r')
		for line in f: # loop through line
			if line.strip() == str('J_'+str(Jlen) + '_axs'):
				Jlen += 1

	# print the brief info for the simulation to the screen
	print('Briefing:')
	print('Total number of gas-phase equations: %i' %(num_eqn[0]))
	print('Total number of aqueous-phase equations: %i' %(num_eqn[1]))
	print('Total number of species found in chemical scheme file: %i\n' %(species_step))
	
	# outputs: 
	
	# rindx  - matrix to record indices of reactants (cols) in each equation (rows)
	# pindx - indices of equation products (cols) in each equation (rows)
	# rstoi - matrix to record stoichometries of reactants (cols) in each equation (rows)
	# pstoi - matrix to record stoichometries of products (cols) in each equation (rows)
	# reac_coef - list for equation reaction rate coefficients
	# spec_list - list for components' SMILE strings
	# Pybel_objects - list of Pybel objects
	# species_step - number of species
	# num_eqn - number of equations
	# nreac - number of reactants in each equation
	# max_no_jaci - number of columns for Jacobian index matrix
	# nprod - number of products per equation
	# prodn - number of columns in pindx
	# reacn - rindx number of columns
	# spec_namelist - list of component names used in the chemical reaction file
	
	return (rindx, pindx, rstoi, pstoi, reac_coef, spec_list, Pybel_objects, num_eqn, 
			species_step, RO2_indices, nreac,
			nprod, prodn, reacn, spec_namelist, Jlen)



# This function generates a python script that calculate rate coef. numerically
# main part by Dave (/s)
def write_rate_file(reac_coef, rrc, rrc_name, testf):
	if testf==0:
		f = open('PyCHAM/Rate_coeffs.py', mode='w')
	if testf==2:
		f = open('Rate_coeffs.py', mode='w')
	f.write('\'\'\'module for calculating gas-phase reaction rate coefficients, automatically generated by eqn_parser\'\'\'\n')
	f.write('\n')
	f.write('##################################################################################################### \n') # python will convert \n to os.linesep
	f.write('# Python function to hold expressions for calculating rate coefficients for a given equation number # \n') # python will convert \n to os.linesep
	f.write('#    Copyright (C) 2017  David Topping : david.topping@manchester.ac.uk                             # \n')
	f.write('#                                      : davetopp80@gmail.com                                       # \n')
	f.write('#    Personal website: davetoppingsci.com                                                           # \n')
	f.write('#                                                                                                   # \n')
	f.write('#                                                                                                   # \n')
	f.write('#                                                                                                   # \n')
	f.write('##################################################################################################### \n')    
	f.write('# Minor modified by XSX\n')
	f.write('# File Created at %s\n' %(datetime.datetime.now()))
	f.write('\n')
	f.write('import numpy\n')
	f.write('import PhotolysisRates\n')
	f.write('\n')

	# following part is the function (there should be an indent at the start of each line)
	# suggest using 1 Tab
	f.write('def evaluate_rates(RO2, H2O, TEMP, lightm, time, lat, lon, act_flux_path, DayOfYear, M, N2, O2, photo_par_file, Jlen):\n')
	f.write('\n')
	f.write('	# ------------------------------------------------------------------------')
	f.write('	# inputs:\n')
	f.write('	# M - third body concentration (molecules/cc (air))\n')
	f.write('	# N2 - nitrogen concentration (molecules/cc (air))\n')
	f.write('	# O2 - oxygen concentration (molecules/cc (air))\n')
	f.write('	# RO2: specified by the chemical scheme. eg: subset of MCM\n')
	f.write('	# H2O, TEMP: given by the user\n')
	f.write('	# lightm: given by the user and is 0 for lights off and 1 for on\n')
	f.write('	# reaction rate coefficients and their names parsed in eqn_parser.py \n')
	f.write('	# Jlen - number of photolysis reactions')
	f.write('\n')
	f.write('	# calculate reaction rates with given by chemical scheme\n')
	# code to calculate rate coefficients given by chemical scheme file
	for line in rrc:
		f.write('	%s \n' %line)
	f.write('\n')
	f.write('	# estimate and append photolysis rates\n')
	f.write('	J = PhotolysisRates.PhotolysisCalculation(time, lat, lon, TEMP, act_flux_path, DayOfYear, photo_par_file, Jlen)\n')
	f.write('\n')
	f.write('	if lightm == 0:\n')
	f.write('		J = [0]*len(J)\n')

	# calculate the rate coef. numerically for each equation
	f.write('	rate_values = numpy.zeros(%i)\n' %(len(reac_coef)))
	# BE NOTIFIED!!!: before writing the script, 'reac_coef' must be converted to 
	# python-compatible format
	f.write('	# reac_coef has been formatted so that python can recognize it\n')
	for eqn_key in range (len(reac_coef)):
		f.write('	rate_values[%s] = %s\n' %(eqn_key, reac_coef[eqn_key]))
	f.write('	\n')
	f.write('	return rate_values\n')
	f.close()

# function to automatically generate a module that is used to record the tendency
# of components (components specified by the user, their index given by rec_comp_index) 
# to change in response to box model 
# mechanisms - gets called inside ode_gen on each time step
def write_dydt_rec():

	f = open('PyCHAM/dydt_rec.py', mode='w')
	f.write('\'\'\'module for calculating and recording change tendency of components, automatically generated by eqn_parser\'\'\'\n')
	f.write('\n')
	f.write('# File Created at %s\n' %(datetime.datetime.now()))
	f.write('\n')
	f.write('import numpy as np \n')
	f.write('\n')
	# following part is the function (there should be an indent at the start of each line)
	# suggest 1 Tab
	f.write('def dydt_rec(y, rindx, rstoi, reac_coef, pindx, nprod, step, dydt_vst, nreac, num_sb, num_speci, pconc, core_diss, Psat, kelv_fac, kimt, kwgt, Cw, act_coeff):\n')
	f.write('	# loop through components to record the tendency of change \n')
	f.write('	for compi in dydt_vst.get(\'comp_index\'): \n')
	f.write('		# open relevant dictionary value \n')
	f.write('		dydt_rec = dydt_vst.get(compi) \n')
	f.write('		# keep count on relevant reactions \n')
	f.write('		reac_count = 0 \n')
	f.write('		# loop through relevant reactions \n')
	f.write('		for i in dydt_rec[0,0:-2]: # final two rows for particle- and wall-partitioning \n')
	f.write('			i = int(i) # ensure index is integer # this necessary because the dydt_rec array is float (the tendency to change records beneath its first row are float) \n')
	f.write('			# estimate gas-phase change tendency for every reaction involving this component \n')
	f.write('			gprate = ((y[rindx[i, 0:nreac[i]]]**rstoi[i, 0:nreac[i]]).prod())*reac_coef[i] \n')
	f.write('			# identify whether this component reacted or produced\n')
	f.write('			if sum(rindx[i, 0:nreac[i]]==compi)>0: \n')
	f.write('				dydt_rec[step+1, reac_count] -= ((gprate)) #*3600)/np.abs(y[compi]))*100.0 \n')
	f.write('			if sum(pindx[i, 0:nprod[i]]==compi)>0: \n')
	f.write('				dydt_rec[step+1, reac_count] += ((gprate)) #*3600)/np.abs(y[compi]))*100.0 \n')
	f.write('			reac_count += 1 \n')
	f.write('		# now estimate and record tendency to change due to particle- and wall-partitioning  \n')
	f.write('		# particle-partitioning \n')
	f.write('		for ibin in range(num_sb-1): # size bin loop\n')
	f.write('			Csit = y[num_speci*(ibin+1):num_speci*(ibin+2)]\n')
	f.write('			conc_sum = np.zeros((1)) \n')
	f.write('			if pconc>0.0: # if seed particles present \n')
	f.write('				conc_sum[0] = ((Csit[0:-1].sum())+Csit[-1]*core_diss)\n')
	f.write('			else: \n')
	f.write('				conc_sum[0] = Csit.sum() \n')
	f.write('			# prevent numerical error due to division by zero \n')
	f.write('			ish = conc_sum==0.0 \n')
	f.write('			conc_sum[ish] = 1.0e-40 \n')
	f.write('			# particle surface gas-phase concentration (molecules/cc (air)) \n')
	f.write('			Csit = (Csit[compi]/conc_sum)*Psat[compi, 0]*kelv_fac[ibin]*act_coeff[compi, 0] \n')
	f.write('			# partitioning rate (molecules/cc.s) \n')
	f.write('			dydt_all = kimt[compi, ibin]*(y[compi]-Csit) \n')
	f.write('			# gas-phase change (molecules/cc/s) \n')
	f.write('			dydt_rec[step+1, reac_count] -= dydt_all \n')
	f.write('		# wall-partitioning \n')
	f.write('		if (kwgt)>1.0e-10: \n')
	f.write('			# concentration at wall (molecules/cc (air)) \n')
	f.write('			Csit = y[num_speci*num_sb:num_speci*(num_sb+1)] \n')
	f.write('			Csit = (Psat[:,0]*(Csit/Cw)*act_coeff[compi, 0])\n')
	f.write('			dydt_all = (kwgt)*(y[compi]-Csit[compi]) \n')
	f.write('			# gas-phase change (molecules/cc/s) \n')
	f.write('			dydt_rec[step+1, reac_count+1] -= dydt_all \n')
	f.write('		\n')
	f.write('	return(dydt_vst) \n')
	f.close()				
						
	
# This function defines RO2 which is given by an MCM file
# this function is used when certain reaction rate coefficients are a function a RO2
# and is called on by the extract_mechanism function above,
# whilst the resulting RO2 index is used inside rate_valu_calc.py
def write_RO2_indices(smiles_array, RO2_names):
    
    # store the names of RO2 species which are present in the equation file
    # get a list of INDICES of RO2 that present in the equation file 
    # (or total species dict)
    # empty list for RO2_indices
    RO2_indices0 = []
    RO2_indices = []
    
    for name in RO2_names:
        
        if name in smiles_array:
            # get the RO2 index
            index0 = RO2_names.index(name)
            RO2_indices0.append(index0)
            # get the smiles_array index for this RO2 species
            index1 = smiles_array.index(name)
            RO2_indices.append(index1)
    
    # Ensure elements in RO2_indices are int (iterable)
    RO2_indices0 = (np.asarray(RO2_indices0, dtype=int)).reshape(-1, 1)
    RO2_indices = (np.asarray(RO2_indices, dtype=int)).reshape(-1, 1)
    RO2_indices = np.hstack((RO2_indices0, RO2_indices))
    
    return (RO2_indices)

# function to convert reaction rate coefficients to commands, and therefore quantities
# that will be used by Rate_coeffs.py
def write_rrc(rrc, rrc_name):

	f = open('PyCHAM/rrc_calc.py', mode='w')
	f.write('\'\'\'module for calculating reaction rate coefficients, automatically generated by eqn_parser\'\'\'\n')
	f.write('\n')
	f.write('# File Created at %s\n' %(datetime.datetime.now()))
	f.write('\n')
	f.write('import numpy as numpy \n')
	f.write('import PhotolysisRates \n')
	f.write('\n')
	# following part is the function (there should be an indent at the start of each line)
	# suggest 1 Tab
	f.write('def rrc_calc(rrc_name, TEMP, H2O, M, N2, O2, time, lat, lon, act_flux_path, DayOfYear):\n')
	f.write('	\n')
	f.write('	# reaction rate coefficients obtained from user\'s chemical scheme file \n')
	f.write('	rrc_constants = [] # empty list to hold rate coefficients\n')
	for line in rrc:
		f.write('	%s \n' %line)
	f.write('	\n')
	f.write('	for i in rrc_name: \n')
	f.write('		rrc_constants.append(locals()[i]) \n')
	f.write('	\n')
	f.write('	# estimate and append photolysis rates \n')
	f.write('	j = PhotolysisRates.PhotolysisCalculation(time, lat, lon, TEMP, act_flux_path, DayOfYear) \n')
	f.write('	rrc_constants.append(j) \n')
	f.write('	# append photolysis rate label \n')
	f.write('	rrc_name.append(\'J\') \n')
	f.write('	return(rrc_constants, rrc_name) \n')
	f.close()
	
	return