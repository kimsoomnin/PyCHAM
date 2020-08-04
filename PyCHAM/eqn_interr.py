'''module to interrogate equations to withdraw essential information for solution'''
# code to extract the equation information for chemical reactions required for 
# their solution in PyCHAM 

import numpy as np
import re
import formatting
import pybel
import sys
import ipdb

def eqn_interr(num_eqn, naked_list_eqn, rindx, rstoi, pindx, pstoi, 
				chem_scheme_markers, reac_coef, spec_namelist, spec_name, spec_smil,
				spec_list, Pybel_objects, nreac, nprod, comp_num, phase):
				
	# inputs: ----------------------------------------------------------------------------
	# num_eqn - number of equations (scalar)
	# naked_list_eqn - equations in strings
	# rindx - to hold indices of reactants
	# rstoi - to hold stoichiometries of reactants
	# pindx - to hold indices of products
	# pstoi - to hold stoichiometries of products
	# chem_scheme_markers - markers for separating sections of the chemical scheme
	# reac_coef - to hold reaction rate coefficients
	# spec_namelist - name strings of components present in the scheme (not SMILES)
	# spec_name - name string of components in xml file (not SMILES)
	# spec_smil - SMILES from xml file
	# spec_list - SMILES of components present in scheme
	# Pybel_objects - list containing pybel objects
	# nreac - to hold number of reactions per equation
	# nprod - number of products per equation
	# comp_num - number of unique components in reactions across all phases
	# phase - marker for the phase being considered: 0 for gas, 1 for particulates
	# ------------------------------------------------------------------------------------
	
	max_no_reac = 0.0 # log maximum number of reactants in a reaction
	max_no_prod = 0.0 # log maximum number of products in a reaction

	# Loop through equations line by line and extract the required information
	for eqn_step in range(num_eqn):
	
		line = naked_list_eqn[eqn_step] # extract this line
		
		# work out whether equation or reaction rate coefficient part comes first
		eqn_start = str('.*\\' +  chem_scheme_markers[10])
		rrc_start = str('.*\\' +  chem_scheme_markers[9])
		# get index of these markers, note span is the property of the match object that
		# gives the location of the marker
		eqn_start_indx = (re.match(eqn_start, line)).span()[1]
		rrc_start_indx = (re.match(rrc_start, line)).span()[1]
		
		if eqn_start_indx>rrc_start_indx:
			eqn_sec = 1 # equation is second part
		else:
			eqn_sec = 0 # equation is first part
		
		# split the line into 2 parts: equation and rate coefficient
		# . means match with anything except a new line character., when followed by a * 
		# means match zero or more times (so now we match with all characters in the line
		# except for new line characters, so final part is stating the character(s) we 
		# are specifically looking for, \\ ensures the marker is recognised
		if eqn_sec == 1:
			eqn_markers = str('\\' +  chem_scheme_markers[10]+ '.*\\' +  chem_scheme_markers[11])
		else: # end of equation part is start of reaction rate coefficient part
			eqn_markers = str('\\' +  chem_scheme_markers[10]+ '.*\\' +  chem_scheme_markers[9])

		# extract the equation as a string ([0] extracts the equation section and 
		# [1:-1] removes the bounding markers)
		eqn = re.findall(eqn_markers, line)[0][1:-1].strip()
		
		eqn_split = eqn.split()
		eqmark_pos = eqn_split.index('=')
		# with stoich number; rule out the photon
		reactants = [i for i in eqn_split[:eqmark_pos] if i != '+' and i != 'hv']
		products = [t for t in eqn_split[eqmark_pos+1:] if t != '+'] # with stoich number
		
		# record maximum number of reactants across all equations
		max_no_reac = np.maximum(len(reactants), max_no_reac)
		# record maximum number of products across all equations
		max_no_prod = np.maximum(len(products), max_no_prod)

		# append columns if needed
		while max_no_reac > np.minimum(rindx.shape[1], rstoi.shape[1]): 
			rindx = np.append(rindx, (np.zeros((num_eqn, 1))).astype(int), axis=1)
			rstoi = np.append(rstoi, (np.zeros((num_eqn, 1))), axis=1)
		while max_no_prod > np.minimum(pindx.shape[1], pstoi.shape[1]): 
			pindx = np.append(pindx, (np.zeros((num_eqn, 1))).astype(int), axis=1)
			pstoi = np.append(pstoi, (np.zeros((num_eqn, 1))), axis=1)

		# .* means occurs anywhere in line and, first \ means second \ can be interpreted 
		# and second \ ensures recognition of marker
		rate_coeff_start_mark = str('\\' +  chem_scheme_markers[9])
		# . means match with anything except a new line character, when followed by a * 
		# means match zero or more times (so now we match with all characters in the line
		# except for new line characters, \\ ensures the marker
		# is recognised
		if eqn_sec == 1: # end of reaction rate coefficient part is start of equation part
			rate_coeff_end_mark = str('.*\\' +  chem_scheme_markers[10])
		else: # end of reaction rate coefficient part is end of line
			rate_coeff_end_mark = str('.*\\' +  chem_scheme_markers[11])
		
		# rate coefficient starts and end punctuation
		rate_regex = str(rate_coeff_start_mark + rate_coeff_end_mark)
		# rate coefficient expression in a string
		rate_ex = re.findall(rate_regex, line)[0][1:-1].strip()

		# convert fortran-type scientific notation to python type
		rate_ex = formatting.SN_conversion(rate_ex)
		# convert the rate coefficient expressions into Python readable commands
		rate_ex = formatting.convert_rate_mcm(rate_ex)
		if (rate_ex.find('EXP') != -1):
			print(rate_ex)
			sys.exit()
		
		# store the reaction rate coefficient for this equation 
		# (/s once any inputs applied)
		reac_coef.append(rate_ex)
		
		# extract the stoichiometric number of the specii in current equation
		reactant_step = 0
		product_step = 0
		stoich_regex = r"^\d*\.\d*|^\d*"
		numr = len(reactants) # number of reactants in this equation
		
		
		# left hand side of equations (losses)
		for reactant in reactants:
				
			if (re.findall(stoich_regex, reactant)[0] != ''):
				stoich_num = float(re.findall(stoich_regex, reactant)[0])
				# name with no stoich number
				name_only = re.sub(stoich_regex, '', reactant)
			elif (re.findall(stoich_regex, reactant)[0] == ''):
				stoich_num = 1.0
				name_only = reactant
			
			# store stoichometry
			rstoi[eqn_step, reactant_step] = stoich_num
			
			if name_only not in spec_namelist: # if new component encountered
				spec_namelist.append(name_only) # add to chemical scheme name list
			
				# convert MCM chemical names to SMILES
				if name_only in spec_name:
					# index where xml file name matches reaction component name
					name_indx = spec_name.index(name_only)
					name_SMILE = spec_smil[name_indx] # SMILES of component
				else:
					sys.exit(str('Error: inside eqn_parser, chemical scheme name '+str(name_only)+' not found in xml file'))
			
				spec_list.append(name_SMILE) # list SMILE names
				name_indx = comp_num # allocate index to this species
				# Generate pybel
				Pybel_object = pybel.readstring('smi', name_SMILE)
				# append to Pybel object list
				Pybel_objects.append(Pybel_object)
				
				comp_num += 1 # number of unique species
				

			else: # if it's a species already encountered it will be in spec_list
				# existing index
				name_indx = spec_namelist.index(name_only)
			
			# store reactant index
			# check if index already present - i.e. component appears more than once
			if sum(rindx[eqn_step, 0:reactant_step]==int(name_indx))>0:
				# get pre-existing index of this component
				exist_indx = np.where(rindx[eqn_step, 0:reactant_step]==(int(name_indx)))
				# add to pre-existing stoichiometry
				rstoi[eqn_step, exist_indx] += rstoi[eqn_step, reactant_step]
				rstoi[eqn_step, reactant_step] = 0 # remove stoichiometry added above
				reactant_step -= 1 # ignore this duplicate product
			else:
				rindx[eqn_step, reactant_step] = int(name_indx)

			reactant_step += 1
			
		# number of reactants in this equation
		nreac[eqn_step] = int(reactant_step)
		
		# right hand side of equations (gains)
		for product in products:

			if (re.findall(stoich_regex, product)[0] != ''):
				stoich_num = float(re.findall(stoich_regex, product)[0])
				name_only = re.sub(stoich_regex, '', product) # name with no stoich number

			elif (re.findall(stoich_regex, product)[0] == ''):
				stoich_num = 1.0
				name_only = product
			
			# store stoichometry
			pstoi[eqn_step, product_step] = stoich_num
			
			if name_only not in spec_namelist: # if new component encountered
				spec_namelist.append(name_only)
				
				# convert MCM chemical names to SMILES
				# index where xml file name matches reaction component name
				if name_only in spec_name:
					name_indx = spec_name.index(name_only)
					name_SMILE = spec_smil[name_indx]
				else:
					sys.exit(str('Error: inside eqn_parser, chemical scheme name '+str(name_only)+' not found in xml file'))
				
				spec_list.append(name_SMILE) # list SMILE string of parsed species
				name_indx = comp_num # allocate index to this species
				# Generate pybel
				
				Pybel_object = pybel.readstring('smi', name_SMILE)
				# append to Pybel object list
				Pybel_objects.append(Pybel_object)
				
				comp_num += 1 # number of unique species
				
			

			else: # if it's a species already encountered
				# index of component already listed
				name_indx = spec_namelist.index(name_only)
				
			# store product index
			# check if index already present - i.e. component appears more than once
			if sum(pindx[eqn_step, 0:product_step]==int(name_indx))>0:
				exist_indx = np.where(pindx[eqn_step, 0:product_step]==(int(name_indx))) # get pre-existing index of this component
				# add to pre-existing stoichometry
				pstoi[eqn_step, exist_indx] += pstoi[eqn_step, product_step]
				pstoi[eqn_step, product_step] = 0 # remove stoichometry added above
				product_step -= 1 # ignore this duplicate product
			else:
				pindx[eqn_step, product_step] = int(name_indx)
			product_step += 1
		
		# number of products in this equation
		nprod[eqn_step] = int(product_step)

	return(rindx, rstoi, pindx, pstoi, reac_coef, spec_namelist, spec_list, 
			Pybel_objects, nreac, nprod, comp_num)