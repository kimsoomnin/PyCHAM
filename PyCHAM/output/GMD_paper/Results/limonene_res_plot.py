'''Code to plot results for limonene oxidation example'''
# aim is to exemplify the coupled integration of gas-phase chemistry and partitioning to
# particles and wall

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as si
from matplotlib.colors import LinearSegmentedColormap # for customised colormap
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker # set colormap tick labels to standard notation
import ipdb

# ----------------------------------------------------------------------------------------
# import results files, all used the limonene MCM PRAM sheme (limonene_MCM_PRAM.txt) 

# open saved files
output_by_sim = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/paper_GMD/GMD_paper/Results/limonene_output'

# name of file where experiment constants saved
fname = str(output_by_sim+'/model_and_component_constants')

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
		if str(line.split(',')[0]) == 'factor_for_multiplying_ppb_to_get_molec/cm3':
			dlist.append(float(i))
			
	const[str(line.split(',')[0])] = dlist

num_sb = int((const['number_of_size_bins'])[0]) # number of size bins
num_speci = int((const['number_of_components'])[0]) # number of species
y_mw = const['molecular_weights_g/mol_corresponding_to_component_names']
y_MV = const['molecular_volumes_cm3/mol']
PyCHAM_names = const['component_names']
# conversion factor to change gas-phase concentrations from molecules/cc 
# (air) into ppb
Cfactor = float((const['factor_for_multiplying_ppb_to_get_molec/cm3'])[0])

# name of file where concentration (ppb for gas-phase, molecules/cc (air) for others) 
# results saved
fname = str(output_by_sim+'/concentrations_all_components_all_times_gas_particle_wall')
y = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header

# withdraw times
fname = str(output_by_sim+'/time')
t_array = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header

# withdraw size bin bounds
fname = str(output_by_sim+'/size_bin_bounds')
sbb = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header

if num_sb>1:
	# name of file where concentration (# particles/cc (air)) results saved
	fname = str(output_by_sim+'/particle_number_concentration_dry')
	N = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header
	if num_sb==2: # just one particle size bin (wall included in num_sb)
		N = np.array(N.reshape(len(t_array), num_sb-1))
		
# name of file where particle size results saved
fname = str(output_by_sim+'/size_bin_radius')
x = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header
if t_array.ndim==0: # occurs if only one time step saved
	x = np.array(x.reshape(1, num_sb-1))
if num_sb==2: # just one particle size bin (wall included in num_sb)
	x = np.array(x.reshape(len(t_array), num_sb-1))
	
# ----------------------------------------------------------------------------------------
# get indices of components with a nitrate group
nitr_ind = np.empty(0, dtype='int')
for i in range(len(PyCHAM_names)):
	if len(PyCHAM_names[i])<3:
		continue
	if PyCHAM_names[i] == 'HNO3': # don't include inorganic nitrate
		continue
	else:
		for i2 in range(len(PyCHAM_names[i])):
			if PyCHAM_names[i][i2:i2+3] == 'NO3':
				nitr_ind = np.append(nitr_ind, i)

# ----------------------------------------------------------------------------------------

# prepare plot

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10,7))

# parasite axis for particle-phase plot
par1 = ax1.twinx() # first parasite axis
par2 = ax1.twinx() # second parasite axis

def make_patch_spines_invisible(ax):
	ax.set_frame_on(True)
	ax.patch.set_visible(False)
	for sp in ax.spines.values():
		sp.set_visible(False)

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)
# ----------------------------------------------------------------------------------------
# gas-phase concentrations

# get index of components of interest
comp_names = ['O3', 'NO2', 'LIMONENE', 'N2O5', 'NO3']
comp_index = []
for i in range(len(comp_names)):
	index = PyCHAM_names.index(comp_names[i])
	comp_index.append(index)

# plot gas-phase concentrations of O3 and NO2
ax0.semilogy(t_array/3600.0, y[:, comp_index[0]], label = 'O3')
ax0.semilogy(t_array/3600.0, y[:, comp_index[1]], label = 'NO2')
ax0.semilogy(t_array/3600.0, y[:, comp_index[2]], label = 'LIMONENE')
ax0.semilogy(t_array/3600.0, y[:, comp_index[3]], label = 'N2O5')
ax0.semilogy(t_array/3600.0, y[:, comp_index[4]], label = 'NO3')
ax0.set_ylim(1.0e-2, 1.0e2)
ax0.set_ylabel(r'Gas-phase concentration (ppb)', fontsize=12)
ax0.yaxis.set_tick_params(size=14)
ax0.set_xlabel(r'Time through simulation (hours)', fontsize=12)
ax0.yaxis.set_tick_params(size=14)

ax0.legend(fontsize=10)
ax0.text(x=(t_array/3600.0)[0], y=2e2, s='a)', size=12)

# ----------------------------------------------------------------------------------------
# particle-phase properties

# number size distribution (dN/dlog10(Dp))

# don't use the first boundary as it's zero, so will error when log10 taken
log10D = np.log10(sbb[1::]*2.0)
if num_sb>2:
	# note, can't append zero to start of log10D to cover first size bin as the log10 of the
	# non-zero boundaries give negative results due to the value being below 1, so instead
	# assume same log10 distance as the next pair
	log10D = np.append((log10D[0]-(log10D[1]-log10D[0])).reshape(1, 1), log10D.reshape(-1,1), axis=0)
	# radius distance covered by each size bin (log10(um))
	dlog10D = (log10D[1::]-log10D[0:-1]).reshape(1, -1)
if num_sb==2: # number of size bins includes wall
	# assume lower radius bound is ten times smaller than upper
	dlog10D = (log10D-np.log10((sbb[1::]/10.0)*2.0)).reshape(1, -1)

# repeat size bin diameter widths over times
dlog10D = np.repeat(dlog10D, N.shape[0], axis=0)

# prepare number size distribution contours (/cc (air))
dNdlog10D = np.zeros((N.shape[0], N.shape[1]))
dNdlog10D = N/dlog10D

# transpose number concentration results, so time on x axis and diameter on y
dNdlog10D = dNdlog10D.transpose()

# mask the nan values so they're not plotted
z = np.ma.masked_where(np.isnan(dNdlog10D), dNdlog10D)

# customised colormap (https://www.rapidtables.com/web/color/RGB_Color.html)
colors = [(0.60, 0.0, 0.70), (0, 0, 1), (0, 1.0, 1.0), (0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]  # R -> G -> B
n_bin = 100  # Discretizes the colormap interpolation into bins
cmap_name = 'my_list'
# Create the colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

# set contour levels
levels = (MaxNLocator(nbins = 100).tick_values(np.min(z[~np.isnan(z)]), 
		np.max(z[~np.isnan(z)])))

# associate colours and contour levels
norm1 = BoundaryNorm(levels, ncolors = cm.N, clip=True)

# contour plot with times along x axis and particle diameters along y axis
p1 = ax1.pcolormesh(t_array/3600.0, (sbb*2*1e3), z[:, :], cmap=cm, norm=norm1)


ax1.set_ylabel('Diameter (nm)', size=12)
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
ax1.set_xlabel(r'Time through simulation (hours)', fontsize=12)

# function for doing colorbar tick labels in standard notation
def fmt(x, pos):
	a, b = '{:.1e}'.format(x).split('e')
	b = int(b)
	return r'${} \times 10^{{{}}}$'.format(a, b)

cb = plt.colorbar(p1, format=ticker.FuncFormatter(fmt), pad=0.25)
cb.ax.tick_params(labelsize=14)   

# colour bar label
cb.set_label('dN/dlog10(D) $\mathrm{(cm^{-3})}$', size=12, rotation=270, labelpad=20)

ax1.text(x=(t_array[0]/3600.0), y=(sbb[-1]*2*1e3)*1.05, s='b)', size=12)

# ----------------------------------------------------------------------------------------
# organic aerosol mass concentration in particles (ug/m3)
final_i = 0
# check whether water and/or core is present
if PyCHAM_names[-2] == 'H2O': # if both present
	final_i = 2
if PyCHAM_names[-1] == 'H2O': # if just water
	final_i = 1

SOAvst = 0.0
NO3vst = np.zeros((y.shape[0], 1))
for i in range(1, num_sb):
	# calculate SOA (*1.0E-12 to convert from g/cc (air) to ug/m3 (air) assuming a density
	# of 1.0 g/cc)
	SOAvst += (((y[:,num_speci*i:num_speci*(i+1)-final_i]/si.N_A)*y_mw[0:-final_i]*1.0e12).sum(axis=1))
	# get just the concentrations of nitrate components in this size bin 
	# (molecules/cc (air))
	yNO3 = y[:, num_speci*i:num_speci*(i+1)-final_i]
	
	for i2 in nitr_ind:
		NO3vst[:, 0] += ((yNO3[:, i2]/si.N_A)*y_mw[i2]*1.0e12)
	
# mass concentration of organics on wall
# (*1.0E-12 to convert from g/cc (air) to ug/m3 (air) assuming a density
# of 1.0 g/cc)
Owall = (((y[:, num_speci*num_sb:num_speci*(num_sb+1)-final_i]/si.N_A)*y_mw[0:-final_i]*1.0e12).sum(axis=1))

# log10 of maximum in SOA
SOAmax = int(np.log10(max(SOAvst)))
Owallmax = int(np.log10(max(Owall)))
SOAmax = max(SOAmax, Owallmax)
# transform SOA so no standard notation required
SOAvst = SOAvst/(10**(SOAmax))
# transform Owall so no standard notation required
Owall = Owall/(10**(SOAmax))
# transform NO3vst so no standard notation required
NO3vst = NO3vst/(10**(SOAmax))

p5, = par2.plot(t_array/3600.0, SOAvst, '--k', label = '[SOA]')
p6, = par2.plot(t_array/3600.0, Owall, '-.k', label = '[wall organic]')
p7, = par2.plot(t_array/3600.0, NO3vst, ':k', label = '[particle RONO2]')
par2.set_ylabel(str('[organics]x ' + str(10**(SOAmax)) + ' ($\mathrm{\mu g\, m^{-3}})$'), rotation=270, size=12, labelpad=25)
par2.yaxis.set_tick_params(labelsize=12)

# ----------------------------------------------------------------------------------------
# total particle number concentration

p3, = par1.plot(t_array/3600.0, N.sum(axis=1), '-k', label = '[N]')
par1.set_ylabel('[N] ($\mathrm{cm^{-3}})$', rotation=270, size=12, labelpad=25)

plt.legend(fontsize=12, handles=[p3, p5, p6, p7] ,loc=2)
fig.savefig('limonene_res_plot.png')
plt.show()