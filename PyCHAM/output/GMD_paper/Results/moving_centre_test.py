'''module to test the moving-centre method of PyCHAM against Fig. 13.8 of Jacobson (2005), needs the file name of results manually inserting'''
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------------------
# test volume-size distribution to get the initial distribution given in Fig. 13.8 of
# Jacobson (2005)
# 
# # size bin bounds as if using the logarithmic spacing option in Size_distributions
# num_bins = 20
# rad_bounds = 10.0**(np.linspace(np.log10(0.035), 
# 						np.log10(50), num=(num_bins+1)))
# rwid = (rad_bounds[1::]-rad_bounds[0:-1]) # width of size bins (um)
# x_output = rad_bounds[0:-1]+rwid/2.0 # particle radius (um)
# 
# Dp = rad_bounds*2.0 # size bin bounds in diameter (um)
# Dpcen = x_output*2.0 # diameter at size bin centres (um)
# Dpsp = (np.log10(Dp))[1::]-(np.log10(Dp))[0:-1] # spacing of logarithm of diameters
# # for 19 size bins
# # num = np.array((1000.0, 1500.0, 300.0, 350.0, 200.0, 40.0, 6.0, 1.0, 0.9, 0.6, 0.4, 1.2e-1, 2.5e-2, 3.0e-3, 3.0e-4, 2.0e-5, 5.0e-7, 0.0, 0.0))
# num = np.array((750.0, 750.0, 1500.0, 500.0, 50.0, 200.0, 50.0, 2.0, 0.7, 0.9, 0.6, 0.4, 1.2e-1, 2.5e-2, 3.0e-3, 3.0e-4, 2.0e-5, 5.0e-7, 0.0, 0.0))
# # num = np.array((800.0, 1000.0, 1100.0, 500.0, 220.0, 100.0, 100.0, 160.0, 170.0, 80.0, 30.0, 10.0, 3.0, 1.0, 0.4, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06, 0.029, 0.015, 6.0e-3, 2.0e-3, 5.0e-4, 1.8e-4, 5.0e-5, 1.0e-5, 2.0e-6, 2.0e-7, 0.0, 0.0, 0.0, 0.0)) # particle number per size bin (#/cm3)
# vol = num*(4.0/3.0)*np.pi*(Dpcen/2.0)**3.0 # particle volume per size bin
# 
# # two-point moving average
# dDp2p = Dpsp[0:-1]+Dpsp[1::]
# dV2p = vol[0:-1]+vol[1::]
# dv0 = dV2p/dDp2p
# fig, (ax0) = plt.subplots(1, 1, figsize=(12,6))
# ax0.loglog(Dp[1:-1], dv0, '-x', label='Initial')
# ax0.set_xlabel(r'Particle Diameter ($\mathit{D}_{\rm{p}},\, \rm{\mu}$m)', fontsize=10)
# ax0.set_ylabel(r'$\rm{d}\mathit{v}\, (\mu \rm{m^{3}\, cm^{-3}})/\rm{dlog_{10}}\mathit{D}_{\rm{p}}}$', fontsize=10)
# plt.legend()
# plt.show()

# when testing against Fig. 3 of Zhang et al. (1999): doi.org/10.1080/027868299304039

from scipy import stats # Import the scipy.stats module
import numpy as np
import matplotlib.pyplot as plt

# sulphuric acid condensation rate:
#um3/cm3/12h -> um3/cm3/s
SArate = 5.5/43200.0
# molecular volume (um3/molecule)
MV = 1.72702238e-10
# molecular density rate (molecules/cm3/s)
SArate = SArate/MV


# lower and upper size (um)
lowersize = 0.0005
uppersize = 500.0
num_bins = 100

rad_bounds = 10.0**(np.linspace(np.log10(lowersize), 
						np.log10(uppersize), num=(num_bins+1)))
rwid = (rad_bounds[1::]-rad_bounds[0:-1]) # width of size bins (um)
x_output = rad_bounds[0:-1]+rwid/2.0 # particle radius (um)

# nucleation mode:

pconc = 2400.0
mean = 0.020
std = np.log(1.2)
loc = 0.0
scale = np.exp(np.log(mean))


# number fraction-size distribution - enforce high resolution to ensure size
# distribution of seed particles fully captured
hires = 10**(np.linspace(np.log10(lowersize), np.log10(uppersize), 1000))
pdf_output = stats.lognorm.pdf(hires, std, loc, scale)
pdf_out_nuc = np.interp(x_output, hires, pdf_output)

# number concentration of all size bins (# particle/cc (air))
Nperbin_nuc = (pdf_out_nuc/sum(pdf_out_nuc))*pconc

# volume concentration (um3/cm3)
volC = ((4.0/3.0)*np.pi*x_output**3.0*Nperbin_nuc)
# print(volC.sum())

# plt.loglog(x_output*2.0, volC /(np.log10(rad_bounds[1::]*2.0)-np.log10(rad_bounds[0:-1]*2.0)), '-o')
# plt.show()

# accumulation mode:

mean = 0.06
pconc = 3600.0
std = np.log(1.8)
loc = 0.0
scale = np.exp(np.log(mean))

# number fraction-size distribution - enforce high resolution to ensure size
# distribution of seed particles fully captured
pdf_output = stats.lognorm.pdf(hires, std, loc, scale)
pdf_out_accum = np.interp(x_output, hires, pdf_output)

# number concentration of all size bins (# particle/cc (air))
Nperbin_accum = (pdf_out_accum/sum(pdf_out_accum))*pconc

# volume concentration (um3/cm3)
volC = ((4.0/3.0)*np.pi*x_output**3.0*Nperbin_accum)
# print(volC.sum())

# plt.loglog(x_output*2.0, volC/(np.log10(rad_bounds[1::]*2.0)-np.log10(rad_bounds[0:-1]*2.0)), '-x')
# plt.show()


# coarse mode:

mean = 1.10
pconc = 1.9
std = np.log(2.1)
loc = 0.0
scale = np.exp(np.log(mean))

# number fraction-size distribution - enforce high resolution to ensure size
# distribution of seed particles fully captured
hires = 10**(np.linspace(np.log10(lowersize), np.log10(uppersize), 1000))
pdf_output = stats.lognorm.pdf(hires, std, loc, scale)
pdf_out_coarse = np.interp(x_output, hires, pdf_output)

# number concentration of all size bins (# particle/cc (air))
Nperbin_coarse = (pdf_out_coarse/sum(pdf_out_coarse))*pconc

# volume concentration (um3/cm3)
volC = ((4.0/3.0)*np.pi*x_output**3.0*Nperbin_coarse)
# print(volC.sum())

# plt.semilogx(x_output*2.0, volC/(np.log10(rad_bounds[1::]*2.0)-np.log10(rad_bounds[0:-1]*2.0)), '-x')
# plt.show()

# total Nperbin:
Nperbin = Nperbin_nuc+Nperbin_accum+Nperbin_coarse       
# for 60 size bin (end of 26/03/2020)
pconc = 1.16383683e-09, 1.92056619e-08, 2.71060857e-07, 3.29004771e-06, 3.42385154e-05, 3.05054120e-04, 2.33694312e-03, 1.53416501e-02, 8.62989910e-02, 4.17133172e-01, 1.72787397e+00, 6.13795633e+00, 1.88927649e+01, 6.57726632e+01, 4.29565603e+02, 1.44581796e+03, 1.31684627e+03, 6.46880800e+02, 6.02811207e+02, 6.22440308e+02, 5.55824957e+02, 4.25764094e+02, 2.79736100e+02, 1.57668428e+02, 7.62455944e+01, 3.16466322e+01, 1.13277894e+01, 3.55912770e+00, 1.07037986e+00, 4.12324741e-01, 2.71131328e-01, 2.38872567e-01, 2.12133765e-01, 1.74412466e-01, 1.30635014e-01, 8.89082618e-02, 5.49534790e-02, 3.08479087e-02, 1.57287821e-02, 7.28154702e-03, 3.06199328e-03, 1.16954972e-03, 4.05482925e-04, 1.27750950e-04, 3.65510483e-05, 9.48899420e-06, 2.24030961e-06, 4.80094262e-07, 9.33130530e-08, 1.65133160e-08, 2.65022633e-09, 3.85854270e-10, 5.11607550e-11, 6.14822787e-12, 6.70707488e-13, 6.66201928e-14, 5.99383937e-15, 4.90069037e-16, 3.64608667e-17, 2.45540542e-18
# for 100 size bin (end of 26/03/2020)
pconc = 3.70992583e-10, 2.07147971e-09, 1.09448455e-08, 5.47207518e-08, 2.58885598e-07, 1.15898303e-06, 4.90975158e-06, 1.96813604e-05, 7.46558313e-05, 2.67969421e-04, 9.10163583e-04, 2.92527167e-03, 8.89662503e-03, 2.56033316e-02, 6.97235594e-02, 1.79669451e-01, 4.38107088e-01, 1.01088781e+00, 2.20724142e+00, 4.56113074e+00, 8.94783895e+00, 1.72274772e+01, 3.82820692e+01, 1.15749306e+02, 3.51078381e+02, 7.43349399e+02, 9.71278277e+02, 7.99449088e+02, 5.02761780e+02, 3.65499291e+02, 3.55625222e+02, 3.71436839e+02, 3.73704589e+02, 3.56248638e+02, 3.21375076e+02, 2.74335688e+02, 2.21597057e+02, 1.69378875e+02, 1.22510272e+02, 8.38521077e+01, 5.43135498e+01, 3.32977857e+01, 1.93279926e+01, 1.06318594e+01, 5.55503390e+00, 2.77358807e+00, 1.34410350e+00, 6.56386676e-01, 3.48125791e-01, 2.20336436e-01, 1.71404120e-01, 1.52949483e-01, 1.43579234e-01, 1.34731485e-01, 1.23643965e-01, 1.10069044e-01, 9.47791559e-02, 7.88681413e-02, 6.34009818e-02, 4.92325858e-02, 3.69281916e-02, 2.67551881e-02, 1.87241144e-02, 1.26571955e-02, 8.26449792e-03, 5.21240532e-03, 3.17543004e-03, 1.86857068e-03, 1.06208448e-03, 5.83110795e-04, 3.09232775e-04, 1.58402595e-04, 7.83756800e-05, 3.74578215e-05, 1.72920220e-05, 7.71065030e-06, 3.32107048e-06, 1.38167825e-06, 5.55235687e-07, 2.15520863e-07, 8.08058174e-08, 2.92642342e-08, 1.02370018e-08, 3.45899022e-09, 1.12892909e-09, 3.55897236e-10, 1.08373480e-10, 3.18758454e-11, 9.05608550e-12, 2.48518799e-12, 6.58745598e-13, 1.68661424e-13, 4.17112461e-14, 9.96390854e-15, 2.29903332e-15, 5.12388714e-16, 1.10304288e-16, 2.29363109e-17, 4.60673102e-18, 8.93717039e-19
# for 100 size bins slightly lower number concentration in nucleation and accumulation mode
pconc = 3.33893325e-10, 1.86433174e-09, 9.85036097e-09, 4.92486766e-08, 2.32997038e-07, 1.04308472e-06, 4.41877642e-06, 1.77132244e-05, 6.71902482e-05, 2.41172479e-04, 8.19147224e-04, 2.63274450e-03, 8.00696252e-03, 2.30429984e-02, 6.27512035e-02, 1.61702506e-01, 3.94296379e-01, 9.09799025e-01, 1.98651716e+00, 4.10500881e+00, 8.05269675e+00, 1.54965724e+01, 3.43493030e+01, 1.03419646e+02, 3.12902542e+02, 6.61990286e+02, 8.65092074e+02, 7.12923638e+02, 4.49792395e+02, 3.28328330e+02, 3.19982012e+02, 3.34287259e+02, 3.36333910e+02, 3.20623813e+02, 2.89237652e+02, 2.46902268e+02, 1.99437607e+02, 1.52441415e+02, 1.10259931e+02, 7.54679637e+01, 4.88837955e+01, 2.99703273e+01, 1.73984416e+01, 9.57306621e+00, 5.00526853e+00, 2.50346912e+00, 1.21851663e+00, 6.01135066e-01, 3.25124236e-01, 2.11275334e-01, 1.68026457e-01, 1.51758067e-01, 1.43181565e-01, 1.34605884e-01, 1.23606427e-01, 1.10058428e-01, 9.47763149e-02, 7.88674218e-02, 6.34008094e-02, 4.92325467e-02, 3.69281832e-02, 2.67551864e-02, 1.87241141e-02, 1.26571954e-02, 8.26449791e-03, 5.21240532e-03, 3.17543004e-03, 1.86857068e-03, 1.06208448e-03, 5.83110795e-04, 3.09232775e-04, 1.58402595e-04, 7.83756800e-05, 3.74578215e-05, 1.72920220e-05, 7.71065030e-06, 3.32107048e-06, 1.38167825e-06, 5.55235687e-07, 2.15520863e-07, 8.08058174e-08, 2.92642342e-08, 1.02370018e-08, 3.45899022e-09, 1.12892909e-09, 3.55897236e-10, 1.08373480e-10, 3.18758454e-11, 9.05608550e-12, 2.48518799e-12, 6.58745598e-13, 1.68661424e-13, 4.17112461e-14, 9.96390854e-15, 2.29903332e-15, 5.12388714e-16, 1.10304288e-16, 2.29363109e-17, 4.60673102e-18, 8.93717039e-19

Vperbin = (4.0/3.0)*np.pi*x_output**3.0*Nperbin

print('plotting dV/dlog10(Dp)')
plt.semilogx(x_output*2.0, Vperbin/(np.log10(rad_bounds[1::]*2.0)-np.log10(rad_bounds[0:-1]*2.0)), '-')
plt.show()
print('plotting dN/dlog10(Dp)')
plt.semilogx(x_output*2.0, Nperbin/(np.log10(rad_bounds[1::]*2.0)-np.log10(rad_bounds[0:-1]*2.0)), '-')
plt.show()

# ----------------------------------------------------------------------------------------
# once happy with the initial volume-size distribution, input the bounds and particle
# number concentration to 'moving_centre_inputs'
# then analyse results with code below

# Jacobson (2005) result
# diameters (um)
Jacob_x = np.array((0.1, 0.11, 0.2, 0.28, 0.4, 0.41, 16.0, 16.1, 20.0, 21.0, 22.0, 26.0, 30.0, 70.0, 100.0))
Jacob_y = np.array((0.4, 0.4, 1.25, 110.0, 300.0, 1.0e-10, 1.0e-10, 3.0e7, 2.0e6, 3.0e5, 4.0e5, 1.1e4, 1.1e3, 2.0, 0.3))

# open saved files
output_by_sim = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/PyCHAM_Gitw/PyCHAM/output/empty_chem/test_31815_coag_100diss_100dens_200p00g'

# withdraw times (s)
fname = str(output_by_sim+'/t')
t_array = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw number-size distributions (# particles/cc (air))
fname = str(output_by_sim+'/N_wet')
N = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header)

# withdraw radii at size bin centre (um)
fname = str(output_by_sim+'/x')
x = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header)

# withdraw size bin bounds, represented by radii (um)
fname = str(output_by_sim+'/sbb')
sbb = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)
# correct for extension of upper limit
sbb[-1] = sbb[-1]*1.0e-2
# correct for zeroing of lower limit
sbb[0] = 1.0e-40

Dpbb = sbb*2.0 # diameters at size bin boundaries (um)
dDp = (np.log10(Dpbb))[1::]-(np.log10(Dpbb))[0:-1] # difference in the log10 of bin bounds
# volume concentration of particles per size bin (um3/cc (air))
dV = ((4.0/3.0)*np.pi*x**3.0)*N

# volumes normalised by size bin width (um3/cm)
dv0 = dV[0,:]/dDp # initial time
dv1 = dV[-1,:]/dDp # final time step
# two-point moving average
# combined difference in the log10(diameter) of neighbouring bins (um)
dDp2p = dDp[0:-1]+dDp[1::]
# combined volume concentration of neighbouring bins (um3/cc (air))
dV2p = dV[:,0:-1]+dV[:,1::]
dv0 = dV2p[0,:]/dDp2p # initial time with two-point moving average
dv2p = dV2p[-1,:]/dDp2p # final time step

# create figure to plot results
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12,6))
ax1.loglog((sbb*2.0)[1:-1], dv0, '-', label='Initial')
ax1.loglog(Jacob_x, Jacob_y, '--', label='Jacobson (2005) Fig. 13.4')
ax1.loglog((sbb*2.0)[1:-1], dv2p, '-.', label='Moving-centre, two-point moving average')

# ax0.loglog((sbb*2.0)[1:-1], dv1, '+', label='Moving-centre, no averaging')
ax1.set_xlabel(r'Particle diameter $\rm{(}\mathit{D}\rm{_p, \, \mu m})$', fontsize=12)
ax1.set_ylabel(r'd$\nu$ ($\rm{\mu m^{3}\, cm^{-3}})/dlog_{10}{\mathit{D}\rm_{p}}$', fontsize=12)
ax1.set_ylim([0.1, 30000000.0])
ax1.set_xlim([0.1, 100.0])
ax1.text(x=0.068, y=2.0e7, s='b)', size=12)
ax1.yaxis.set_tick_params(size=12)
ax1.xaxis.set_tick_params(size=12)
ax1.legend()

# ----------------------------------------------------------------------------------------
# Zhang et al. (1999) plot

# open saved files
output_by_sim = '/Users/Simon_OMeara/Documents/Manchester/postdoc_stuff/box-model/PyCHAM_Gitw/PyCHAM/output/empty_chem/test_Zhang_60sb_273.15K'


# withdraw times (s)
fname = str(output_by_sim+'/t')
t_array = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)

# withdraw number-size distributions (# particles/cc (air))
fname = str(output_by_sim+'/N_wet')
N = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header)

# withdraw radii at size bin centre (um)
fname = str(output_by_sim+'/x')
x = np.loadtxt(fname, delimiter=',', skiprows=1) # skiprows=1 omits header)

# withdraw size bin bounds, represented by radii (um)
fname = str(output_by_sim+'/sbb')
sbb = np.loadtxt(fname,delimiter=',',skiprows=1) # skiprows=1 omits header)
# correct for extension of upper limit
sbb[-1] = sbb[-1]*1.0e-2
# correct for zeroing of lower limit
sbb[0] = 1.0e-40

Dpbb = sbb*2.0 # diameters at size bin boundaries (um)
dDp = (np.log10(Dpbb))[1::]-(np.log10(Dpbb))[0:-1] # difference in the log10 of bin bounds
# volume concentration of particles per size bin (um3/cc (air))
dV = ((4.0/3.0)*np.pi*x**3.0)*N

# volumes normalised by size bin width (um3/cm)
dv0 = dV[0,:]/dDp # initial time
dv1 = dV[-1,:]/dDp # final time step
# two-point moving average
# combined difference in the log10(diameter) of neighbouring bins (um)
dDp2p = dDp[0:-1]+dDp[1::]
# combined volume concentration of neighbouring bins (um3/cc (air))
dV2p = dV[:,0:-1]+dV[:,1::]
dv0 = dV2p[0,:]/dDp2p # initial time with two-point moving average
dv2p = dV2p[-1,:]/dDp2p # final time step

# three-point moving average
dDp3p = (dDp[0:-2]+dDp[1:-1]+dDp[2::])
# combined volume concentration of neighbouring bins (um3/cc (air))
dV3p = dV[:,0:-2]+dV[:,1:-1]+dV[:,2::]
dv3p = dV3p[-1,:]/dDp3p # final time step

ax0.semilogx((sbb*2.0)[1:-1], dv0, '-', label='Initial')
# plt.semilogx((x*2.0)[-1, :], dv1, '-o', label='Moving-centre, one-point moving average')
# plt.semilogx((x*2.0)[-1, 1:-1], dv3p, '-.', label='Moving-centre, three-point moving average')

# ----------------------------------------------------------------------------------------
# Zhang et al. (1999) output

lowersize = 0.0005
uppersize = 500.0
num_bins = 100

rad_bounds = 10.0**(np.linspace(np.log10(lowersize), 
						np.log10(uppersize), num=(num_bins+1)))
rwid = (rad_bounds[1::]-rad_bounds[0:-1]) # width of size bins (um)
x_output = rad_bounds[0:-1]+rwid/2.0 # particle radius (um)

# nucleation mode:
pconc = 1500.0
mean = 0.045
std = np.log(1.03)
loc = 0.0
scale = np.exp(np.log(mean))

# number fraction-size distribution - enforce high resolution to ensure size
# distribution of seed particles fully captured
hires = 10**(np.linspace(np.log10(lowersize), np.log10(uppersize), 1000))
pdf_output = stats.lognorm.pdf(hires, std, loc, scale)
pdf_out_nuc = np.interp(x_output, hires, pdf_output)

# number concentration of all size bins (# particle/cc (air))
Nperbin_nuc = (pdf_out_nuc/sum(pdf_out_nuc))*pconc

# accumulation mode:
mean = 0.060
pconc = 8000.0
std = np.log(1.7)
loc = 0.0
scale = np.exp(np.log(mean))

# number fraction-size distribution - enforce high resolution to ensure size
# distribution of seed particles fully captured
pdf_output = stats.lognorm.pdf(hires, std, loc, scale)
pdf_out_accum = np.interp(x_output, hires, pdf_output)

# number concentration of all size bins (# particle/cc (air))
Nperbin_accum = (pdf_out_accum/sum(pdf_out_accum))*pconc

# coarse mode:
mean = 1.10
pconc = 1.9
std = np.log(2.1)
loc = 0.0
scale = np.exp(np.log(mean))

# number fraction-size distribution - enforce high resolution to ensure size
# distribution of seed particles fully captured
pdf_output = stats.lognorm.pdf(hires, std, loc, scale)
pdf_out_coarse = np.interp(x_output, hires, pdf_output)

# number concentration of all size bins (# particle/cc (air))
Nperbin_coarse = (pdf_out_coarse/sum(pdf_out_coarse))*pconc

# total Nperbin:
Nperbin = Nperbin_nuc+Nperbin_accum+Nperbin_coarse       

Vperbin = (4.0/3.0)*np.pi*x_output**3.0*Nperbin

ax0.semilogx(x_output*2.0, Vperbin/(np.log10(rad_bounds[1::]*2.0)-np.log10(rad_bounds[0:-1]*2.0)), '--', label='Zhang et al. (1999) Fig. 3 exact solution')

# ----------------------------------------------------------------------------------------

ax0.semilogx((sbb*2.0)[1:-1], dv2p, '-.', label='Moving-centre, two-point moving average')
ax0.set_ylabel(r'd$\nu$ ($\rm{\mu m^{3}\, cm^{-3}})/dlog_{10}{\mathit{D}\rm_{p}}$', fontsize=12)
ax0.set_xlim([1.0e-3, 1.0e1])
ax0.text(x=6.0e-4, y=31, s='a)', size=12)
ax0.yaxis.set_tick_params(size=12)
ax0.xaxis.set_tick_params(size=12)
ax0.legend()

fig.savefig('mov_cen_test.png')

plt.show()
