import numpy as np
import CoolProp.CoolProp as CP
import time
import matplotlib.pyplot as plt
import pandas as pd

# Sub functions
def gnielinskiBlended(f):
	'''
	This function smoothly blendes a laminar expression for Nusselt number taken from Incropera - Fundamentals of Heat and
	Mass Transfer (Eq. 8.55) with the Gnielinski Nu-corr for turbulent flow in circular pipes.
	'''

	# Flag which records if the Reynolds number is outside its bounds
	ReOutsideBounds = False

	#Sigmoid properties
	k = 0.011
	ReMidPoint = 2700

	#Heat transfer correlation for circular pipes
	sigmoid = 1 / (1 + np.exp(-k * (f.Re_mean[-1] - ReMidPoint)))
	fLam = 64 / f.Re_mean[-1]
	NuLam = 3.66
	fTurb = (0.79 * np.log(f.Re_mean[-1]) - 1.64) ** -2  # petukhov
	NuTurb = ((fTurb / 8) * (f.Re_mean[-1] - 1000) * f.Pr_mean[-1]) / (1 + 12.7 * ((fTurb / 8) ** 0.5) * (f.Pr_mean[-1] ** (2 / 3) - 1))  # gnielinski
	f.f = (1 - sigmoid) * fLam + sigmoid * fTurb
	f.Nu = (1 - sigmoid) * NuLam + sigmoid * NuTurb

	return f

def Lienhard(f, L, Afr, sigma):
	# Method for integrating the boundary layer using correlations by Lienhard
	#import sympy as sp
	#x = sp.symbols('x')
	#x = np.zeros((np.append(np.shape(f.sigma), 50)))
	x = np.linspace(0.00001, L, 100)

	f.u = f.mdot/(f.rho[-2]*Afr*sigma)
	f.Rex = np.multiply.outer(f.rho[-2]/f.mu_mean[-1]*f.u, x)

	upr_uinf = 0.1  # Turbulence intensity
	Rel = 3.6*10**5*(100*upr_uinf)**(-5/4)
	f.x_l = Rel*f.mu_mean[-1]/(f.rho[-2]*f.u)

	# Laminar region
	a = 0.332 #Uniform Wall Temperature
	#a = 0.453 #Uniform Heat Flux
	Nu_lam = a*f.Rex**(1/2)*f.Pr_mean[-1]**(1/3)
	f.h_lam = Nu_lam*f.k_mean[-1]/x
	f_lam = Nu_lam*2/f.Rex*f.Pr_mean[-1]**(-1/3)

	# Transitional region
	c = 0.9922*np.log10(Rel)-3.013 # For Rel < 5*10^5
	Nu_trans = a*Rel**(1/2)*f.Pr_mean[-1]**(1/3)*(f.Rex/Rel)**c
	f_trans = Nu_trans*2/f.Rex*f.Pr_mean[-1]**(-1/3)

	# Turbulent region
	f_turb = 0.455 / (np.log(0.06*f.Rex)**2)
	Nu_turb = (f.Rex*f.Pr_mean[-1]*f_turb/2) / (1+12.7*(f.Pr_mean[-1]**(2/3)-1)*(f_turb/2)**(1/2))

	f.Nu_x = (Nu_lam**5+(Nu_trans**(-10)+Nu_turb**(-10))**(-1/2))**(1/5)
	f.h_x = f.Nu_x*f.k_mean[-1]/x

	f.f_x = (f_lam**5+(f_trans**(-10)+f_turb**(-10))**(-1/2))**(1/5)

	f.Nu = np.trapz(f.Nu_x, axis=3)

	f.f = np.trapz(f.f_x, axis=3)

	return f

def KL_curvefit(HEX):
	# f1 side
	HEX.f1_l_d_h = HEX.f1_L/(4*HEX.f1_sigma/HEX.f1_alpha)
	k1, a, b, c = [ 3.60039451e-01, -4.00821276e-01,  2.12762744e-05, -4.12945363e-01]
	HEX.f1_j = k1 * (HEX.f1_l_d_h)**a * HEX.f1_Re**c + b*HEX.f1_l_d_h
	HEX.f1_St = HEX.f1_j/(HEX.f1_Pr**(2/3))
	HEX.f1_Nu = HEX.f1_St*HEX.f1_Re*HEX.f1_Pr # = HEX.f1_j*HEX.Re*HEX.Pr**(1/3)

	k1, a, b, c = [ 5.08290733e-01, -5.51216597e-01,  1.09255635e-04, -2.34092870e-01]

	HEX.f1_f = k1 * (HEX.f1_l_d_h)**a * HEX.f1_Re**c + b*HEX.f1_l_d_h

	# f2 side
	HEX.f2_l_d_h = HEX.f2_L/(4*HEX.f2_sigma/HEX.f2_alpha)
	k1, a, b, c = [3.60039451e-01, -4.00821276e-01, 2.12762744e-05, -4.12945363e-01]
	HEX.f2_j = k1*(HEX.f2_l_d_h)**a*HEX.f2_Re**c+b*HEX.f2_l_d_h
	HEX.f2_St = HEX.f2_j/(HEX.f2_Pr**(2/3))
	HEX.f2_Nu = HEX.f2_St*HEX.f2_Re*HEX.f2_Pr

	k1, a, b, c = [5.08290733e-01, -5.51216597e-01, 1.09255635e-04, -2.34092870e-01]

	HEX.f2_f = k1*(HEX.f2_l_d_h)**a*HEX.f2_Re**c+b*HEX.f2_l_d_h

	return HEX

def KL_curvefit_light(HEX, f1, f2):
	# Reynolds number
	f1.f1_Re_mean = 4*f1.mdot/(HEX.f1_alpha*HEX.f1_Afr*f1.mu_mean)
	f2.f2_Re_mean = 4*f2.mdot/(HEX.f2_alpha*HEX.f2_Afr*f2.mu_mean)

	# f1 side
	HEX.f1_l_d_h = HEX.f1_L/(4*HEX.f1_sigma/HEX.f1_alpha)
	k1, a, b, c = [ 3.60039451e-01, -4.00821276e-01,  2.12762744e-05, -4.12945363e-01]
	f1_j = k1 * (HEX.f1_l_d_h)**a * f1.f1_Re_mean**c + b*HEX.f1_l_d_h
	f1_St = f1_j/(f1.Pr_mean**(2/3))
	HEX.f1_Nu = f1_St*f1.f1_Re_mean*f1.Pr_mean

	k1, a, b, c = [ 5.08290733e-01, -5.51216597e-01,  1.09255635e-04, -2.34092870e-01]

	HEX.f1_f = k1 * (HEX.f1_l_d_h)**a * f1.f1_Re_mean**c + b*HEX.f1_l_d_h

	# f2 side
	HEX.f2_l_d_h = HEX.f2_L/(4*HEX.f2_sigma/HEX.f2_alpha)
	k1, a, b, c = [3.60039451e-01, -4.00821276e-01, 2.12762744e-05, -4.12945363e-01]
	f2_j = k1*(HEX.f2_l_d_h)**a*f2.f2_Re_mean**c+b*HEX.f2_l_d_h
	f2_St = f2_j/(f2.Pr_mean**(2/3))
	HEX.f2_Nu = f2_St*f2.f2_Re_mean*f2.Pr_mean

	k1, a, b, c = [5.08290733e-01, -5.51216597e-01, 1.09255635e-04, -2.34092870e-01]

	HEX.f2_f = k1*(HEX.f2_l_d_h)**a*f2.f2_Re_mean**c+b*HEX.f2_l_d_h

	return HEX

def KaysAndLondonHEX(HEX, f1, f2):
	f1_Re_mean = 4*f1.mdot/(HEX.f1_alpha*HEX.f1_Afr*f1.mu_mean)
	f2_Re_mean = 4*f2.mdot/(HEX.f2_alpha*HEX.f2_Afr*f2.mu_mean)

	# f1 side
	data = pd.read_excel('KaysLondonCollection.xlsx', HEX.f1_hex)

	# Read heat transfer and friction parameters
	Re_f1_ref = np.flip(data['NR'].to_numpy())
	f_f1_ref = np.flip(data.get('f').to_numpy())
	j_f1_ref = np.flip(data.get('NstNPr2/3').to_numpy())

	HEX.f1_f = np.interp(f1_Re_mean, Re_f1_ref, f_f1_ref)
	HEX.f1_j = np.interp(f1_Re_mean, Re_f1_ref, j_f1_ref)
	f1_St = HEX.f1_j/(f1.Pr_mean[0]**(2/3))
	HEX.f1_Nu = f1_St*f1_Re_mean*f1.Pr_mean[0]

	if HEX.f2_hex == 'tube':
		k = 0.011
		ReMidPoint = 2700

		#Heat transfer correlation for circular pipes
		sigmoid = 1/(1+np.exp(-k*(f2_Re_mean-ReMidPoint)))
		fLam = 64/f2_Re_mean
		NuLam = 3.66
		fTurb = (0.79*np.log(f2_Re_mean)-1.64)**-2  # petukhov
		NuTurb = ((fTurb/8)*(f2_Re_mean-1000)*f2.Pr_mean[0])/(1+12.7*((fTurb/8)**0.5)*(f2.Pr_mean[0]**(2/3)-1))  # gnielinski
		HEX.f2_f = (1-sigmoid)*fLam+sigmoid*fTurb
		HEX.f2_Nu = (1-sigmoid)*NuLam+sigmoid*NuTurb
	else:
		# f2 side
		data = pd.read_excel('KaysLondonCollection.xlsx', HEX.f2_hex)

		# Read heat transfer and friction parameters
		Re_f2_ref = np.flip(data['NR'].to_numpy())
		f_f2_ref = np.flip(data.get('f').to_numpy())
		j_f2_ref = np.flip(data.get('NstNPr2/3').to_numpy())

		HEX.f2_f = np.interp(f2_Re_mean, Re_f2_ref, f_f2_ref)
		HEX.f2_j = np.interp(f2_Re_mean, Re_f2_ref, j_f2_ref)
		f2_St = HEX.f2_j/(f2.Pr_mean[0]**(2/3))
		HEX.f2_Nu = f2_St*f2_Re_mean*f2.Pr_mean[0]

	return HEX

def get_frontalArea_FlowDirr(HEX):
	# ----------------------------------Set frontal area and HEX.flow----------------------------------
	# f1 frontal area
	if HEX.f1_flowdir=='Lx':
		f1_flow = 1
		HEX.f1_Afr = HEX.Ly*HEX.Lz
	elif HEX.f1_flowdir=='-Lx':
		f1_flow = -1
		HEX.f1_Afr = HEX.Ly*HEX.Lz
	elif HEX.f1_flowdir=='Ly':
		f1_flow = 2
		HEX.f1_Afr = HEX.Lx*HEX.Lz
	elif HEX.f1_flowdir=='-Ly':
		f1_flow = -2
		HEX.f1_Afr = HEX.Lx*HEX.Lz
	elif HEX.f1_flowdir=='Lz':
		f1_flow = 3
		HEX.f1_Afr = HEX.Lx*HEX.Ly
	elif HEX.f1_flowdir=='-Lz':
		f1_flow = -3
		HEX.f1_Afr = HEX.Lx*HEX.Ly
	else:
		print("Wrong flow direction given for f1")

	# f2 frontal area
	if HEX.f2_flowdir=='Lx':
		f2_flow = 1
		HEX.f2_Afr = HEX.Ly*HEX.Lz
	elif HEX.f2_flowdir=='-Lx':
		f2_flow = -1
		HEX.f2_Afr = HEX.Ly*HEX.Lz
	elif HEX.f2_flowdir=='Ly':
		f2_flow = 2
		HEX.f2_Afr = HEX.Lx*HEX.Lz
	elif HEX.f2_flowdir=='-Ly':
		f2_flow = -2
		HEX.f2_Afr = HEX.Lx*HEX.Lz
	elif HEX.f2_flowdir=='Lz':
		f2_flow = 3
		HEX.f2_Afr = HEX.Lx*HEX.Ly
	elif HEX.f2_flowdir=='-Lz':
		f2_flow = -3
		HEX.f2_Afr = HEX.Lx*HEX.Ly
	else:
		print("Wrong flow direction given for f2")

	if f1_flow==f2_flow:
		HEX.flow = "Parallel"
	elif f1_flow==-f2_flow:
		HEX.flow = "Counter"
	else:
		HEX.flow = "Cross"

def get_sigma_alpha_sigmaFTT_Dh(HEX):
	HEX.f2_sigma = (1-HEX.chi)/(1+HEX.sigma_r)
	HEX.f1_sigma = HEX.f2_sigma*HEX.sigma_r

	HEX.f2_alpha = 2/HEX.wt*HEX.chi/(HEX.alpha_r+1)
	HEX.f1_alpha = HEX.f2_alpha*HEX.alpha_r

	HEX.f1_sigma_ftt = (HEX.f1_alpha-HEX.f2_alpha)/HEX.f1_alpha
	HEX.f2_sigma_ftt = (HEX.f2_alpha-HEX.f1_alpha)/HEX.f2_alpha

	HEX.f1_Dh = 4*HEX.f1_sigma/HEX.f1_alpha
	HEX.f2_Dh = 4*HEX.f2_sigma/HEX.f2_alpha

def get_surfaceEfficiencies(HEX):
	# Fin efficiency
	HEX.f1_ml = HEX.FinAR*(2*HEX.f1_h_ave/HEX.k)**0.5
	HEX.f1_eta_fin = np.tanh(HEX.f1_ml)/HEX.f1_ml

	f1_finned = (HEX.f1_sigma_ftt*np.ones_like(HEX.f1_eta_fin))>=0

	# Overall surface efficiency
	HEX.f1_eta_O = np.ones_like(HEX.f1_eta_fin)
	HEX.f1_eta_O[f1_finned] = (1-HEX.f1_sigma_ftt*(1-HEX.f1_eta_fin))[f1_finned]
	# -----------Where f2 is enhanced side-----------
	# Fin efficiency
	HEX.f2_ml = HEX.FinAR*(2*HEX.f2_h_ave/HEX.k)**0.5
	HEX.f2_eta_fin = np.tanh(HEX.f2_ml)/HEX.f2_ml

	f2_finned = (HEX.f2_sigma_ftt*np.ones_like(HEX.f2_eta_fin))>=0

	# Overall surface efficiency
	HEX.f2_eta_O = np.ones_like(HEX.f2_eta_fin)
	HEX.f2_eta_O[f2_finned] = (1-HEX.f2_sigma_ftt*(1-HEX.f2_eta_fin))[f2_finned]

def get_C_Cmin_Cr_NTUmax(HEX, f1, f2):
	HEX.f1_C = (f1.h[-1]-f1.h[-2])/(f1.T0[-1]-f1.T0[-2])*f1.mdot
	HEX.f2_C = (f2.h[-1]-f2.h[-2])/(f2.T0[-1]-f2.T0[-2])*f2.mdot

	# Capacity rate min and ratio
	HEX.Cmin = np.minimum(HEX.f1_C, HEX.f2_C)
	HEX.Cr = HEX.Cmin/np.maximum(HEX.f1_C, HEX.f2_C)

	# Maximum number of transfer units
	HEX.NTUmax = HEX.Lx*HEX.Ly*HEX.Lz*HEX.f1_alpha*HEX.U_f1/HEX.Cmin

def get_U_f1(HEX):
	HEX.U_f1 = 1/(1/(HEX.f1_eta_O*HEX.f1_h_ave)+HEX.wt/((HEX.f1_alpha+HEX.f2_alpha)/(2*HEX.f1_alpha)*HEX.k)+1/(
				HEX.f2_eta_O*HEX.f2_h_ave*HEX.f2_alpha/HEX.f1_alpha))

def get_effectiveness(HEX):
	if HEX.flow=="Counter":
		# For counter flow
		HEX.eps = (1-np.exp(-HEX.NTUmax*(1-HEX.Cr)))/(1-HEX.Cr*np.exp(-HEX.NTUmax*(1-HEX.Cr)))
	elif HEX.flow=="Parallel":
		HEX.eps = (1-np.exp(-HEX.NTUmax*(1+HEX.Cr)))/(1+HEX.Cr)
	elif HEX.flow=="Cross":
		# For cross flow, both fluid unmixed
		# HEX.eps = 1-np.exp((1/HEX.Cr)*HEX.NTUmax**.22*(np.exp(-HEX.Cr*HEX.NTUmax**.78)-1))
		# For cross flow, both fluid mixed
		HEX.eps = HEX.NTUmax/(HEX.NTUmax/(1-np.exp(-HEX.NTUmax))+HEX.Cr*HEX.NTUmax/(1-np.exp(-HEX.NTUmax*HEX.Cr))-1)
	else:
		HEX.eps = HEX.NTUmax/(HEX.NTUmax/(1-np.exp(-HEX.NTUmax))+HEX.Cr*HEX.NTUmax/(1-np.exp(-HEX.NTUmax*HEX.Cr))-1)

def InitiateConditions(f):
	if not hasattr(f, 'rho'):
		f.rho = [CP.PropsSI("D", "P", f.p01, "T", f.T01, f.fluid)]
		f.h = [CP.PropsSI("H", "P", f.p01, "T", f.T01, f.fluid)]
		f.T0 = [f.T01]
		f.p0 = [f.p01]
		f.mu_mean = [np.array([1])]
		f.Pr_mean = [np.array([1])]
		f.k_mean = [np.array([1])]
		f.Re_mean = [np.array([1])]
		f.a_mean = [np.array([1])]
		f.v_max = [np.array([1])]
		f.Mach = [np.array([1])]
	else:
		f.mu_mean.append(f.mu_mean[0])
		f.Pr_mean.append(f.Pr_mean[0])
		f.k_mean.append(f.k_mean[0])
		f.Re_mean.append(f.Re_mean[0])
		f.a_mean.append(f.a_mean[0])
		f.v_max.append(f.v_max[0])
		f.Mach.append(f.Mach[0])

	f.p0.append(f.p0[-1]*0.99)
	f.T0.append(f.T0[-1]*0.99)
	f.rho.append(f.rho[-1]*0.99)
	f.h.append(f.h[-1]*0.99)

def Get_Properties(f, dataSource='Coolprop', verbose=False):
	f.T0_mean = (f.T0[-1]+f.T0[-2])/2
	f.p0_mean = (f.p0[-1]+f.p0[-2])/2

	if (dataSource == 'LOCAL') and (f.fluid == 'REFPROP::Parahydrogen'):
		if verbose:
			print("Local parahydrogen")
		fun = np.array([1/(f.T0[-1]**2), 1/(f.p0[-1]**2), 1/f.T0[-1], 1/f.p0[-1], 1*f.T0[-1], 1*f.p0[-1], 1*f.T0[-1]**2, 1*f.p0[-1]**2, 1], dtype=object)
		f.rho[-1] = np.sum(np.array([ 1.24437565e+05,  1.00000000e+00, -2.06794367e+03,  1.30246311e+05, -1.07976849e-01,  2.17380161e-06,  1.40972492e-04, -4.93692919e-14, 1.96131765e+01])*fun)
		f.h[-1] = np.sum(np.array([-2.69721272e+09,  1.00000000e+00,  1.13820931e+08,  1.00000000e+00, 2.38296543e+04, -9.44562022e-03, -1.31559159e+01,  6.24372971e-10, -1.79914578e+06])*fun)
		fun_mean = np.array([1/(f.T0_mean**2), 1/(f.p0_mean**2), 1/f.T0_mean, 1/f.p0_mean, 1*f.T0_mean, 1*f.p0_mean, 1*f.T0_mean**2, 1*f.p0_mean**2, 1], dtype=object)
		f.k_mean[-1] = np.sum(np.array([ 7.19288552e+02, 1.00000000e+00, -2.42108893e+01, -7.03004207e+03, -1.44089332e-04, 9.16771507e-10, 4.18344407e-07, 6.20207010e-17, 2.70227407e-01])*fun_mean)
		f.mu_mean[-1] = np.sum(np.array([ 1.60914156e-02, -3.11720433e+02, -4.42397409e-04, 4.53068470e-01, 1.10598934e-08, 1.24409540e-13, 9.71203227e-12, -2.47157542e-21, 5.44021040e-06])*fun_mean)
		f.Pr_mean[-1] = np.sum(np.array([ 4.27829545e+03, 1.00000000e+00, -9.91063663e+01, 4.06171091e+02, -3.19779186e-03, 1.64491717e-08, 3.97430618e-06, -1.33871848e-15, 1.52228725e+00])*fun_mean)

	elif (dataSource == 'LOCAL') and (f.fluid == 'Air'):
		if verbose:
			print("Local air")
		fun = np.array([1/(f.T0[-1]**2), 1/(f.p0[-1]**2), 1/f.T0[-1], 1/f.p0[-1], 1*f.T0[-1], 1*f.p0[-1], 1*f.T0[-1]**2, 1*f.p0[-1]**2, 1], dtype=object)
		f.rho[-1] = np.sum(np.array([ 2.52014191e+04,  1.00000000e+00,  1.46884379e+03,  7.05211266e+01, -7.26884160e-04,  9.16816372e-06,  3.85477344e-07,  2.89719551e-16, -3.82303508e+00])*fun)
		f.h[-1] = np.sum(np.array([8.08125770e+08, 1.00000000e+00, -1.23015129e+07, 5.57592228e+04, 8.45864957e+02, -1.34209705e-03, 1.49504691e-01, 1.83983609e-11, 1.90896315e+05])*fun)
		fun_mean = np.array([1/(f.T0_mean**2), 1/(f.p0_mean**2), 1/f.T0_mean, 1/f.p0_mean, 1*f.T0_mean, 1*f.p0_mean, 1*f.T0_mean**2, 1*f.p0_mean**2, 1], dtype=object)
		f.k_mean[-1] = np.sum(np.array([ 1.43556803e+02,  2.34416762e+03, -2.19648482e+00, -1.08149068e-02, 6.61908199e-05,  2.39443465e-10, -1.02436466e-08,  1.37574036e-17, 1.31794616e-02])*fun_mean)
		f.mu_mean[-1] = np.sum(np.array([ 1.11964165e-01, -1.54440082e-01, -1.78903379e-03,  1.43044853e-06, 4.17577120e-08,  1.18161457e-13, -8.78518790e-12,  4.71869288e-21, 1.15173755e-05])*fun_mean)
		f.Pr_mean[-1] = np.sum(np.array([-2.69467615e+03, -2.22223327e+04,  4.20663962e+01,  4.84828161e-01, 7.41433297e-05,  5.21905954e-09,  7.15333520e-08, -1.69928019e-16, 5.68654081e-01])*fun)
	else:
		if verbose:
			print("Coolprop")
		f.rho[-1] = np.ndarray.reshape(CP.PropsSI("D", "P", np.ndarray.ravel(f.p0[-1]), "T", np.ndarray.ravel(f.T0[-1]), f.fluid), np.shape(f.p0[-1]))
		f.rho_mean = (f.rho[-1]+f.rho[-2])/2
		f.h[-1] = np.ndarray.reshape(CP.PropsSI("H", "D", np.ndarray.ravel(f.rho[-1]), "T", np.ndarray.ravel(f.T0[-1]), f.fluid), np.shape(f.p0[-1]))
		f.mu_mean[-1] = np.ndarray.reshape(CP.PropsSI("V", "D", np.ndarray.ravel(f.rho_mean), "T", np.ndarray.ravel(f.T0_mean), f.fluid), np.shape(f.p0[-1]))
		f.Pr_mean[-1] = np.ndarray.reshape(CP.PropsSI("PRANDTL", "D", np.ndarray.ravel(f.rho_mean), "T", np.ndarray.ravel(f.T0_mean), f.fluid), np.shape(f.p0[-1]))
		f.k_mean[-1] = np.ndarray.reshape(CP.PropsSI("CONDUCTIVITY", "D", np.ndarray.ravel(f.rho_mean), "T", np.ndarray.ravel(f.T0_mean), f.fluid), np.shape(f.p0[-1]))

def ReshapeObjects(objs):
	listOfLens = np.array([])
	listOfAttr = np.array([])
	listOfObj = np.array([])

	for i in range(0,len(objs)):
		for attr in dir(objs[i]):
			if np.size(getattr(objs[i], attr))>1:
				listOfLens = np.append(listOfLens, len(getattr(objs[i], attr)))
				listOfAttr = np.append(listOfAttr, attr)
				listOfObj = np.append(listOfObj, objs[i])

	j = 0
	for i in range(0, len(objs)):
		for attr in dir(objs[i]):
			if np.size(getattr(objs[i], attr))>1:
				for ii in range(j-1, -1, -1):
					setattr(objs[i], attr, getattr(objs[i], attr)[np.newaxis, ...])
				for ii in range(j+1, len(listOfLens)):
					setattr(objs[i], attr, getattr(objs[i], attr)[..., np.newaxis])
				j = j+1
		setattr(objs[i], "listOfSweepAttrs", listOfAttr)
		setattr(objs[i], "listOfSweepObjs", listOfObj)
		setattr(objs[i], "listOfLens", listOfLens)

# Main function
def GenHex(f1, f2, HEX, verbose=True):

	# ----------------------------------Calculations outside iteration----------------------------------
	# Gets frontal areas and whether it's a counter/cross/parallell flow from given flow directions
	get_frontalArea_FlowDirr(HEX)
	get_sigma_alpha_sigmaFTT_Dh(HEX)

	# ----------------------------------Initial conditions----------------------------------
	# Initiates necessary fluid parameters as vectors and sets the required inlet values from coolprop
	InitiateConditions(f1)
	InitiateConditions(f2)

	# ----------------------------------Aerothermal iterative loop start----------------------------------
	# (iterating density and enthalpy for mean conditions)
	t = time.time()
	# Could be increased but usually converges before 5 is reached
	numIter = 5
	for i in range(numIter):
		if verbose:
			print("Aerothermal iteration",i,"after",time.time()-t,"s")
			t_i = time.time()
		# ----------------------------------Get f1 conditions----------------------------------
		Get_Properties(f1, dataSource=f1.dataSource, verbose=verbose)
		if verbose:
			print("Got all f1 values from source:", f1.dataSource, "in", time.time()-t_i,"s")
			t_i = time.time()

		# ----------------------------------Get f2 conditions----------------------------------
		Get_Properties(f2, dataSource=f2.dataSource, verbose=verbose)
		if verbose:
			print("Got all f1 values from source:", f2.dataSource, "in", time.time()-t_i,"s")
			t_i = time.time()

		# ----------------------------------Calculations----------------------------------
		# Reynolds number
		f1.Re_mean[-1] = 4*f1.mdot/(HEX.f1_alpha*HEX.f1_Afr*f1.mu_mean[-1])
		f2.Re_mean[-1] = 4*f2.mdot/(HEX.f2_alpha*HEX.f2_Afr*f2.mu_mean[-1])

		HEX.f1_Re = f1.Re_mean[-1]
		HEX.f1_Pr = f1.Pr_mean[-1]

		HEX.f2_Re = f2.Re_mean[-1]
		HEX.f2_Pr = f2.Pr_mean[-1]

		# Baseline heat transfer and pressure loss coefficients
		if HEX.correlation == "Gnielinski":
			# Assume both sides as internal of tubes
			f1 = gnielinskiBlended(f1)
			HEX.f1_Nu = f1.Nu
			HEX.f1_f = f1.f
			f2 = gnielinskiBlended(f2)
			HEX.f2_Nu = f2.Nu
			HEX.f2_f = f2.f
		elif HEX.correlation == "Lienhard":
			# Assume both sides as flat plates
			f1 = Lienhard(f1, HEX.Lx, HEX.f1_Afr, HEX.f1_sigma)
			HEX.f1_Nu = f1.Nu
			HEX.f1_f = f1.f
			f2 = Lienhard(f2, HEX.Ly, HEX.f2_Afr, HEX.f2_sigma)
			HEX.f2_Nu = f2.Nu
			HEX.f2_f = f2.f
		elif HEX.correlation == "KaysAndLondon":
			HEX = KaysAndLondonHEX(HEX, f1, f2)
		elif HEX.correlation == "KL":
			HEX = KL_curvefit(HEX)
		else:
			# Both sides from Kays and London curve-fit correlation
			# f1 = KL_curvefit(f1)
			# f2 = KL_curvefit(f2)
			print("Wrong correlation given, using the correlations derived from Kays and London")
			HEX = KL_curvefit(HEX)

		# Nusselt to heat transfer coefficient
		HEX.f1_h_ave = HEX.f1_Nu*f1.k_mean[-1]/HEX.f1_Dh
		HEX.f2_h_ave = HEX.f2_Nu*f2.k_mean[-1]/HEX.f2_Dh

		if verbose:
			print("Calculated heat transfer coefficient in",time.time()-t_i,"s")
			t_i = time.time()

		# Assemble U_f1

		# -----------Where f1 is enhanced side-----------
		get_surfaceEfficiencies(HEX)
		# -----------overall conductance (f1 side)-----------
		get_U_f1(HEX)

		# -----------Calc temp change (effectivness based on f1 side)-----------
		# Fluid heat capacity rate
		get_C_Cmin_Cr_NTUmax(HEX, f1, f2)

		if verbose:
			print("Calculated NTU in",time.time()-t_i,"s")
			t_i = time.time()

		# Effectivness
		get_effectiveness(HEX)

		# Calculate delta T
		f1_dT0 = -HEX.eps*(HEX.Cmin/HEX.f1_C)*(f1.T0[-2]-f2.T0[-2])
		f2_dT0 = HEX.eps*(HEX.Cmin/HEX.f2_C)*(f1.T0[-2]-f2.T0[-2])

		# -----------Calculate pressure loss-----------
		f1_dp0 = - 0.5*f1.mdot**2/(f1.rho[-2])*1/(HEX.f1_Afr)**2 * ((1/HEX.f1_sigma**2+1)*(f1.rho[-2]/f1.rho[-1]-1) + HEX.f1_f*HEX.Lx*HEX.Ly*HEX.Lz/HEX.f1_Afr*HEX.f1_alpha/HEX.f1_sigma**3*2*f1.rho[-2]/(f1.rho[-2]+f1.rho[-1]))
		f2_dp0 = - 0.5*f2.mdot**2/(f2.rho[-2])*1/(HEX.f2_Afr)**2 * ((1/HEX.f2_sigma**2+1)*(f2.rho[-2]/f2.rho[-1]-1) + HEX.f2_f*HEX.Lx*HEX.Ly*HEX.Lz/HEX.f2_Afr*HEX.f2_alpha/HEX.f2_sigma**3*2*f2.rho[-2]/(f2.rho[-2]+f2.rho[-1]))

		# Check convergence
		# T0_f1
		if np.max(1-np.abs((f1.T0[-1]-f1.T0[-2])/f1_dT0))<0.001:
			# T0_f2
			if np.max(1-np.abs((f2.T0[-1]-f2.T0[-2])/f2_dT0))<0.001:
				# p0_f1
				if np.max(1-np.abs((f1.p0[-1]-f1.p0[-2])/f1_dp0))<0.001:
					# p0_f2
					if np.max(1-np.abs((f2.p0[-1]-f2.p0[-2])/f2_dp0))<0.001:
						f1.T0[-1] = f1.T0[-2] + f1_dT0
						f1.p0[-1] = f1.p0[-2] + f1_dp0
						f2.T0[-1] = f2.T0[-2] + f2_dT0
						f2.p0[-1] = f2.p0[-2] + f2_dp0
						#print(np.mean(f1.T0[-1,...]), np.mean(f2.T0[-1,...]))
				break

		if i == numIter-1:
			print("Did not converge")
			print("Max dT0_1", np.max(f1.T0[-1]-f1.T0[-2]))
			print("Max dT0_2", np.max(f2.T0[-1]-f2.T0[-2]))
			print("Max dp0_1", np.max(1-np.abs((f1.p0[-1]-f1.p0[-2])/f1.p0[-2])))
			print("Max dp0_2", np.max(1-np.abs((f2.p0[-1]-f2.p0[-2])/f2.p0[-2])))

		f1.T0[-1] = f1.T0[-2]+f1_dT0
		f1.p0[-1] = f1.p0[-2]+f1_dp0
		f2.T0[-1] = f2.T0[-2]+f2_dT0
		f2.p0[-1] = f2.p0[-2]+f2_dp0

		HEX.f1_Q = (f1.T0[-1]-f1.T0[-2])*HEX.f1_C
		HEX.f2_Q = (f2.T0[-1]-f2.T0[-2])*HEX.f2_C

		if verbose:
			print("Calculated dT0 and dp0 in",time.time()-t_i,"s")
			t_i = time.time()

		HEX.mass = HEX.Lx*HEX.Ly*HEX.Lz*HEX.chi*HEX.density  # Alu
		HEX.f1_aveMetalWallTemperature = (f1.T0[-1]+f1.T0[-2])/2-HEX.f1_Q/(HEX.Lx*HEX.Ly*HEX.Lz*HEX.f1_alpha)/HEX.f1_h_ave
		HEX.f2_aveMetalWallTemperature = (f2.T0[-1]+f2.T0[-2])/2-HEX.f2_Q/(HEX.Lx*HEX.Ly*HEX.Lz*HEX.f2_alpha)/HEX.f2_h_ave
# Initiate objects
f1 = type('f1', (), {})()
f2 = type('f2', (), {})()
HEX = type('HEX', (), {})()

# Core air
f1.T01 = np.array([444])
f1.p01 = np.array([5000000])
f1.mdot = np.array([32])
f1.fluid = 'Air'
f1.dataSource = 'Coolprop'

# Ambient air
f2.T01 = np.array([316])
f2.p01 = np.array([142000])
f2.mdot = np.array([32])
f2.fluid = 'Air'
f2.dataSource = 'Coolprop'

# Geometrical input
# Wall thickness [m]
HEX.wt = 0.0002
# Fin characteristic dimension
HEX.FinAR = 0.01/(HEX.wt**0.5)
# HEX outer volume
HEX.Lx = 0.2
HEX.Ly = 0.2
HEX.Lz = (1+HEX.Ly/2)*np.pi

# Flow directions, to set correct frontal areas and whether it's a counter/cross/parallell flow HEX
HEX.f1_flowdir = 'Ly'
HEX.f2_flowdir = 'Lx'

# Solid conductivity
HEX.k = 120
# Solid density
HEX.density = 2840 # Alu 2219

# What correlation to use for the aerothermal performance estimation, options are:
# Gnielinski (internal channels) (works ok)
# Lienhard (flat plate) (broken)
# KL (korrelations from Kays and London)
# KaysAndLondon (specific heat exchanger from Kays and London library)
# - Requires the KaysLondonCollection.xlsx
# - HEX.f1_hex, HEX.f2_hex should be actual heat exchangers from the library
# - HEX.f2_hex can be set to 'tube' if it is tube internal flow
HEX.correlation = "Lienhard"
# Undisturbed flow length (if tube internal a larger value should be used)
HEX.f1_L = 0.01
HEX.f2_L = 0.01

# The dimensionless geometrical parameters
HEX.sigma_r = np.linspace(0.9, 1.1, 10)
HEX.alpha_r = np.linspace(0.9, 1.1, 10)
HEX.chi = np.linspace(0.1, 0.2, 10)

# Reshaping all input to matrice form, should be used if more than 1 value is given as a vector
ReshapeObjects([f1, f2, HEX])
print("Number of configurations:", np.prod(HEX.listOfLens))
print("Order of swept attributes:", HEX.listOfSweepAttrs)

GenHex(f1, f2, HEX, verbose=False)

print("min/max temp diff in f1: ", np.min(f1.T0[-1]-f1.T0[-2]),"/",np.max(f1.T0[-1]-f1.T0[-2]))
print("min/max temp diff in f2: ", np.min(f2.T0[-1]-f2.T0[-2]),"/",np.max(f2.T0[-1]-f2.T0[-2]))