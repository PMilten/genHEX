import numpy as np
import CoolProp.CoolProp as CP
import time
import pandas as pd
import os
import pickle
from scipy.optimize import root_scalar

def gnielinski(f):
	'''
	This function evaluates the Gnielinski Nu-corr for turbulent flow in circular pipes.
	'''

	# Flag which records if the Reynolds number is outside its bounds
	ReOutsideBounds = False

	#Heat transfer correlation for circular pipes
	f.f_D = 0.316 / f.Re_mean[-1] ** (1 / 4) # Darcy friction
	f.Nu = ((f.f_D / 8) * (f.Re_mean[-1] - 1000) * f.Pr_mean[-1]) / (1 + 12.7 * ((f.f_D / 8) ** 0.5) * (f.Pr_mean[-1] ** (2 / 3) - 1))  # gnielinski

def gnielinskiBlended(f):
	'''
	This function smoothly blendes a laminar expression for Nusselt number taken from Incropera - Fundamentals of Heat and
	Mass Transfer (Eq. 8.55) with the Gnielinski Nu-corr for turbulent flow in circular pipes.
	'''

	# Flag which records if the Reynolds number is outside its bounds
	ReOutsideBounds = False

	#Sigmoid properties
	k = np.array([0.011])
	ReMidPoint = np.array([2700])

	#Heat transfer correlation for circular pipes
	# print("Re_mean", f.Re_mean[-1])
	# print("func", -k * (f.Re_mean[-1] - ReMidPoint))
	sigmoid = 1 / (1 + np.exp(-k * (f.Re_mean[-1] - ReMidPoint)))
	fLam_D = 64 / f.Re_mean[-1]
	NuLam = 3.66
	fTurb_D = (0.79 * np.log(f.Re_mean[-1]) - 1.64) ** -2  # Petukhov (Darcy friction)
	NuTurb = ((fTurb_D / 8) * (f.Re_mean[-1] - 1000) * f.Pr_mean[-1]) / (1 + 12.7 * ((fTurb_D / 8) ** 0.5) * (f.Pr_mean[-1] ** (2 / 3) - 1))  # gnielinski
	f.f_D = (1 - sigmoid) * fLam_D + sigmoid * fTurb_D
	f.Nu = (1 - sigmoid) * NuLam + sigmoid * NuTurb
	f.j = f.Nu/(f.Re_mean[-1]*f.Pr_mean[-1]**(1/3))

# def Lienhard(f, L, Afr, sigma):
# 	# Method for integrating the boundary layer using correlations by Lienhard
# 	#import sympy as sp
# 	#x = sp.symbols('x')
# 	#x = np.zeros((np.append(np.shape(f.sigma), 50)))
# 	x = np.linspace(0.00001, L, 100)
#
# 	f.u = f.mdot/(f.rho[-2]*Afr*sigma)
# 	f.Rex = np.multiply.outer(f.rho[-2]/f.mu_mean[-1]*f.u, x)
#
# 	upr_uinf = 0.1  # Turbulence intensity
# 	Rel = 3.6*10**5*(100*upr_uinf)**(-5/4)
# 	f.x_l = Rel*f.mu_mean[-1]/(f.rho[-2]*f.u)
#
# 	# Laminar region
# 	a = 0.332 #Uniform Wall Temperature
# 	#a = 0.453 #Uniform Heat Flux
# 	Nu_lam = a*f.Rex**(1/2)*f.Pr_mean[-1]**(1/3)
# 	f.h_lam = Nu_lam*f.k_mean[-1]/x
# 	f_lam = Nu_lam*2/f.Rex*f.Pr_mean[-1]**(-1/3)
#
# 	# Transitional region
# 	c = 0.9922*np.log10(Rel)-3.013 # For Rel < 5*10^5
# 	Nu_trans = a*Rel**(1/2)*f.Pr_mean[-1]**(1/3)*(f.Rex/Rel)**c
# 	f_trans = Nu_trans*2/f.Rex*f.Pr_mean[-1]**(-1/3)
#
# 	# Turbulent region
# 	f_turb = 0.455 / (np.log(0.06*f.Rex)**2)
# 	Nu_turb = (f.Rex*f.Pr_mean[-1]*f_turb/2) / (1+12.7*(f.Pr_mean[-1]**(2/3)-1)*(f_turb/2)**(1/2))
#
# 	f.Nu_x = (Nu_lam**5+(Nu_trans**(-10)+Nu_turb**(-10))**(-1/2))**(1/5)
# 	f.h_x = f.Nu_x*f.k_mean[-1]/x
#
# 	f.f_x = (f_lam**5+(f_trans**(-10)+f_turb**(-10))**(-1/2))**(1/5)
#
# 	f.Nu = np.trapz(f.Nu_x, axis=3)
#
# 	f.f = np.trapz(f.f_x, axis=3)

def KL_curvefit(HEX):
	# f1 side
	HEX.f1_l_d_h = HEX.f1_l/(4*HEX.f1_sigma/HEX.f1_alpha)
	k1, a, b, c = [ 3.60039451e-01, -4.00821276e-01,  2.12762744e-05, -4.12945363e-01]
	HEX.f1_j = k1 * (HEX.f1_l_d_h)**a * HEX.f1_Re**c + b*HEX.f1_l_d_h
	HEX.f1_St = HEX.f1_j/(HEX.f1_Pr**(2/3))
	HEX.f1_Nu = HEX.f1_St*HEX.f1_Re*HEX.f1_Pr # = HEX.f1_j*HEX.Re*HEX.Pr**(1/3)

	k1, a, b, c = [ 5.08290733e-01, -5.51216597e-01,  1.09255635e-04, -2.34092870e-01]

	HEX.f1_f_D = 4*(k1 * (HEX.f1_l_d_h)**a * HEX.f1_Re**c + b*HEX.f1_l_d_h)

	# f2 side
	HEX.f2_l_d_h = HEX.f2_l/(4*HEX.f2_sigma/HEX.f2_alpha)
	k1, a, b, c = [3.60039451e-01, -4.00821276e-01, 2.12762744e-05, -4.12945363e-01]
	HEX.f2_j = k1*(HEX.f2_l_d_h)**a*HEX.f2_Re**c+b*HEX.f2_l_d_h
	HEX.f2_St = HEX.f2_j/(HEX.f2_Pr**(2/3))
	HEX.f2_Nu = HEX.f2_St*HEX.f2_Re*HEX.f2_Pr

	k1, a, b, c = [5.08290733e-01, -5.51216597e-01, 1.09255635e-04, -2.34092870e-01]

	HEX.f2_f_D = 4*(k1*(HEX.f2_l_d_h)**a*HEX.f2_Re**c+b*HEX.f2_l_d_h)

def ASME26_correlations(f):
	'''
	This function uses the correlations published with the ASME GT26 publication
	'''

	# Heat transfer correlation for circular pipes
	f.f_D = 6.85 / (f.Re_mean[-1] ** (1 / 2.24)*f.Pr_mean[-1]**(1 / 3)*(f.ell/f.D_h)**(1/2.75))
	f.j = 0.26 / (f.Re_mean[-1] ** (1 / 2.46)*f.Pr_mean[-1]**(1 / 3)*(f.ell/f.D_h)**(1/4.01))

def KaysAndLondonHEX(HEX, f1, f2):
	f1_Re_mean = 4*f1.mdot/(HEX.f1_alpha*HEX.f1_Afr*f1.mu_mean)
	f2_Re_mean = 4*f2.mdot/(HEX.f2_alpha*HEX.f2_Afr*f2.mu_mean)

	# f1 side
	data = pd.read_excel('KaysLondonCollection.xlsx', HEX.f1_hex)

	# Read heat transfer and friction parameters
	Re_f1_ref = np.flip(data['NR'].to_numpy())
	f_F_f1_ref = np.flip(data.get('f').to_numpy())
	j_f1_ref = np.flip(data.get('NstNPr2/3').to_numpy())

	HEX.f1_f_D = 4*np.interp(f1_Re_mean, Re_f1_ref, f_F_f1_ref)
	HEX.f1_j = np.interp(f1_Re_mean, Re_f1_ref, j_f1_ref)
	f1_St = HEX.f1_j/(f1.Pr_mean[0]**(2/3))
	HEX.f1_Nu = f1_St*f1_Re_mean*f1.Pr_mean[0]

	if HEX.f2_hex == 'tube':
		k = 0.011
		ReMidPoint = 2700

		#Heat transfer correlation for circular pipes
		sigmoid = 1/(1+np.exp(-k*(f2_Re_mean-ReMidPoint)))
		fLam_D = 64/f2_Re_mean
		NuLam = 3.66
		fTurb_D = (0.79*np.log(f2_Re_mean)-1.64)**-2  # petukhov
		NuTurb = ((fTurb_D/8)*(f2_Re_mean-1000)*f2.Pr_mean[0])/(1+12.7*((fTurb_D/8)**0.5)*(f2.Pr_mean[0]**(2/3)-1))  # gnielinski
		HEX.f2_f = (1-sigmoid)*fLam_D+sigmoid*fTurb_D
		HEX.f2_Nu = (1-sigmoid)*NuLam+sigmoid*NuTurb
	else:
		# f2 side
		data = pd.read_excel('KaysLondonCollection.xlsx', HEX.f2_hex)

		# Read heat transfer and friction parameters
		Re_f2_ref = np.flip(data['NR'].to_numpy())
		f_F_f2_ref = np.flip(data.get('f').to_numpy())
		j_f2_ref = np.flip(data.get('NstNPr2/3').to_numpy())

		HEX.f2_f_D = 4*np.interp(f2_Re_mean, Re_f2_ref, f_F_f2_ref)
		HEX.f2_j = np.interp(f2_Re_mean, Re_f2_ref, j_f2_ref)
		f2_St = HEX.f2_j/(f2.Pr_mean[0]**(2/3))
		HEX.f2_Nu = f2_St*f2_Re_mean*f2.Pr_mean[0]

def get_frontalArea_FlowDirr(HEX):
	# ----------------------------------Set frontal area and HEX.flow----------------------------------
	# Total volume
	HEX.Vt = HEX.Lx*HEX.Ly*HEX.Lz

	# f1 flow direction and length
	if HEX.f1_flowdir=='Lx':
		f1_flow = 1
		HEX.f1_L = HEX.Lx
	elif HEX.f1_flowdir=='-Lx':
		f1_flow = -1
		HEX.f1_L = HEX.Lx
	elif HEX.f1_flowdir=='Ly':
		f1_flow = 2
		HEX.f1_L = HEX.Ly
	elif HEX.f1_flowdir=='-Ly':
		f1_flow = -2
		HEX.f1_L = HEX.Ly
	elif HEX.f1_flowdir=='Lz':
		f1_flow = 3
		HEX.f1_L = HEX.Lz
	elif HEX.f1_flowdir=='-Lz':
		f1_flow = -3
		HEX.f1_L = HEX.Lz
	else:
		print("Wrong flow direction given for f1")

	# f2 flow direction and length
	if HEX.f2_flowdir=='Lx':
		f2_flow = 1
		HEX.f2_L = HEX.Lx
	elif HEX.f2_flowdir=='-Lx':
		f2_flow = -1
		HEX.f2_L = HEX.Lx
	elif HEX.f2_flowdir=='Ly':
		f2_flow = 2
		HEX.f2_L = HEX.Ly
	elif HEX.f2_flowdir=='-Ly':
		f2_flow = -2
		HEX.f2_L = HEX.Ly
	elif HEX.f2_flowdir=='Lz':
		f2_flow = 3
		HEX.f2_L = HEX.Lz
	elif HEX.f2_flowdir=='-Lz':
		f2_flow = -3
		HEX.f2_L = HEX.Lz
	else:
		print("Wrong flow direction given for f2")

	# Calculate frontal areas for each pass
	HEX.f1_Afr = HEX.Vt / (HEX.f1_L * HEX.f1_n_passes)
	HEX.f2_Afr = HEX.Vt / (HEX.f2_L * HEX.f2_n_passes)

	if f1_flow==f2_flow:
		HEX.flow = "Parallel"
	elif f1_flow==-f2_flow:
		HEX.flow = "Counter"
	else:
		HEX.flow = "Cross"

def get_t(HEX):
	Af_Aw = max(HEX.alpha_r - 1, 1 / HEX.alpha_r - 1)
	HEX.t = (HEX.tw + HEX.tf * Af_Aw) / (1 + Af_Aw)

def get_sigma_alpha_sigmaFTT_Dh(HEX):
	HEX.f2_sigma = (1-HEX.chi)/(1+HEX.sigma_r)
	HEX.f1_sigma = HEX.f2_sigma*HEX.sigma_r

	get_t(HEX)

	HEX.f2_alpha = 2/HEX.t*HEX.chi/(HEX.alpha_r+1)
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

def get_surfaceEfficiencies_assumption(HEX):
	# Fin efficiency
	HEX.f1_eta_fin = np.array([0.8])

	f1_finned = (HEX.f1_sigma_ftt*np.ones_like(HEX.f1_eta_fin))>=0

	# Overall surface efficiency
	HEX.f1_eta_O = np.ones_like(HEX.f1_eta_fin)
	HEX.f1_eta_O[f1_finned] = (1-HEX.f1_sigma_ftt*(1-HEX.f1_eta_fin))[f1_finned]
	# -----------Where f2 is enhanced side-----------
	# Fin efficiency
	HEX.f2_eta_fin = np.array([0.8])

	f2_finned = (HEX.f2_sigma_ftt*np.ones_like(HEX.f2_eta_fin))>=0

	# Overall surface efficiency
	HEX.f2_eta_O = np.ones_like(HEX.f2_eta_fin)
	HEX.f2_eta_O[f2_finned] = (1-HEX.f2_sigma_ftt*(1-HEX.f2_eta_fin))[f2_finned]

def get_C_Cmin_Cr_NTUmax(HEX, f1, f2):
	HEX.f1_C = f1.cp_mean[-1]*f1.mdot
	HEX.f2_C = f2.cp_mean[-1]*f2.mdot

	# Capacity rate min and ratio
	HEX.C_min = np.minimum(HEX.f1_C, HEX.f2_C)
	HEX.C_r = HEX.C_min/np.maximum(HEX.f1_C, HEX.f2_C)

	# Maximum number of transfer units
	HEX.NTUmax = HEX.Vt*HEX.f1_alpha*HEX.U_f1/HEX.C_min

def get_C_Cmin_Cr(HEX, f1, f2):
	HEX.f1_C = f1.cp_mean[-1]*f1.mdot
	HEX.f2_C = f2.cp_mean[-1]*f2.mdot

	# Capacity rate min and ratio
	HEX.C_min = np.minimum(HEX.f1_C, HEX.f2_C)
	HEX.C_r = HEX.C_min/np.maximum(HEX.f1_C, HEX.f2_C)

def get_Re(f1, f2 ,HEX):
	f1.Re_mean = 4 * f1.mdot / (f1.mu_mean[-1] * HEX.f1_alpha * HEX.f1_Afr)
	f2.Re_mean = 4 * f2.mdot / (f2.mu_mean[-1] * HEX.f2_alpha * HEX.f2_Afr)

def get_U_f1(HEX):
	HEX.U_f1 = 1/(1/(HEX.f1_eta_O*HEX.f1_h_ave)+HEX.t/((HEX.f1_alpha+HEX.f2_alpha)/(2*HEX.f1_alpha)*HEX.k)+1/(
				HEX.f2_eta_O*HEX.f2_h_ave*HEX.f2_alpha/HEX.f1_alpha))
	print("f1 thermal resistance", 1/(HEX.f1_eta_O*HEX.f1_h_ave))
	print("f2 thermal resistance", 1/(HEX.f2_eta_O*HEX.f2_h_ave*HEX.f2_alpha/HEX.f1_alpha))

def get_effectiveness(HEX):
	if HEX.flow=="Counter":
		# For counter flow
		HEX.eps = (1-np.exp(-HEX.NTUmax*(1-HEX.C_r)))/(1-HEX.C_r*np.exp(-HEX.NTUmax*(1-HEX.C_r)))
	elif HEX.flow=="Parallel":
		HEX.eps = (1-np.exp(-HEX.NTUmax*(1+HEX.C_r)))/(1+HEX.C_r)
	elif HEX.flow=="Cross":
		# For cross flow, both fluid unmixed
		# HEX.eps = 1-np.exp((1/HEX.C_r)*HEX.NTUmax**.22*(np.exp(-HEX.C_r*HEX.NTUmax**.78)-1))
		# For cross flow, both fluid mixed
		HEX.eps = HEX.NTUmax/(HEX.NTUmax/(1-np.exp(-HEX.NTUmax))+HEX.C_r*HEX.NTUmax/(1-np.exp(-HEX.NTUmax*HEX.C_r))-1)
	else:
		HEX.eps = HEX.NTUmax/(HEX.NTUmax/(1-np.exp(-HEX.NTUmax))+HEX.C_r*HEX.NTUmax/(1-np.exp(-HEX.NTUmax*HEX.C_r))-1)

def get_NTU_from_effectiveness(HEX):
	"""Return NTU given ε and C_r for counterflow or crossflow."""
	if HEX.f1_n_passes > 1 or HEX.f2_n_passes > 1:
		n_passes = HEX.f1_n_passes * HEX.f2_n_passes
		f_p = lambda eps_p: HEX.eps - (((1 - eps_p * HEX.C_r) / (1 - eps_p)) ** n_passes - 1) / (
					((1 - eps_p * HEX.C_r) / (1 - eps_p)) ** n_passes - HEX.C_r)
		result = root_scalar(f_p, bracket=[0.01, 0.99], method='brentq')
		HEX.eps_p = np.array([result.root])
		print(f"effectiveness per pass = {HEX.eps_p[0]:.3f}")
	else:
		HEX.eps_p = HEX.eps
	try:
		if HEX.flow.lower() == "counter":
			if HEX.C_r == 1:
				HEX.NTU = HEX.eps_p/(1-HEX.eps_p)
			elif HEX.C_r == 0:
				HEX.NTU = -np.log(1-HEX.eps_p)
			else:
				HEX.NTU = -np.log((HEX.eps_p - 1) / (HEX.eps_p * HEX.C_r - 1)) / (1 - HEX.C_r)
		elif HEX.flow.lower() == "cross":
			# Crossflow both fluids mixed
			f = lambda NTU: HEX.eps_p-NTU/(NTU/(1-np.exp(-NTU))+HEX.C_r*NTU/(1-np.exp(-NTU*HEX.C_r))-1)
			result = root_scalar(f, bracket=[0.001, 8], method='brentq')
			HEX.NTU = np.array([result.root])
		else:
			raise ValueError("flow_type must be 'counter' or 'cross'")
	except ValueError:
		print("Solve failed: Requested effectiveness is to high for your configuration")

def get_fluid_properties(f):
	"""Return density, Cp, Pr, and viscosity from CoolProp."""
	if np.size(f.T0) == 1:
		f.T0 = np.append(f.T0, f.T0[0]+1)
	if np.size(f.p0) == 1:
		f.p0 = np.append(f.p0, f.p0[0]*0.995)

	try:
		f.rho = CP.PropsSI("D", "T", f.T0, "P", f.p0, f.fluid)
		f.h = CP.PropsSI("H", "T", f.T0, "P", f.p0, f.fluid)
		f.rho_mean = np.array([CP.PropsSI("D", "T", (f.T0[:-1] + f.T0[1:]) / 2, "P", (f.p0[:-1] + f.p0[1:]) / 2, f.fluid)], dtype=float)
		#f.cp_mean = np.array([PropsSI("C", "T", (f.T0[:-1] + f.T0[1:]) / 2, "P", (f.p0[:-1] + f.p0[1:]) / 2, f.fluid)], dtype=float)
		f.cp_mean = np.abs((f.h[:-1] - f.h[1:])/(f.T0[:-1] - f.T0[1:]))
		f.Pr_mean = np.array([CP.PropsSI("PRANDTL", "T", (f.T0[:-1] + f.T0[1:]) / 2, "P", (f.p0[:-1] + f.p0[1:]) / 2, f.fluid)], dtype=float)
		f.mu_mean = np.array([CP.PropsSI("V", "T", (f.T0[:-1] + f.T0[1:]) / 2, "P", (f.p0[:-1] + f.p0[1:]) / 2, f.fluid)], dtype=float)
		f.k_mean = np.array([CP.PropsSI("CONDUCTIVITY", "T", (f.T0[:-1] + f.T0[1:]) / 2, "P", (f.p0[:-1] + f.p0[1:]) / 2, f.fluid)], dtype=float)
	except ValueError as e:
		print(f"⚠️ Could not get properties for {f.fluid}: {e}")

def load_setup(setup_dir=".", extension=".pkl"):
	"""
	Smart loader for heat exchanger setup files.

	- If exactly one setup file exists → auto-load it.
	- If multiple setup files exist → prompt user to choose.
	- If none exist → return None.

	Returns
	-------
	filename : str or None
		The chosen or auto-loaded filename.
	setup : object or None
		The loaded setup object (dict or custom objects).
	"""
	# Find all .pkl files in the directory
	setup_files = [f for f in os.listdir(setup_dir) if f.endswith(extension)]

	if not setup_files:
		print("⚠️ No setup files found in current directory.")
		return None, None

	# Auto-load if there is only one setup file
	if len(setup_files) == 1:
		filename = setup_files[0]
		print(f"📂 Automatically loading '{filename}' (only one found).")
	else:
		# Let the user select if there are multiple
		print("\n📁 Available setup files:")
		for i, f in enumerate(setup_files, 1):
			print(f"  {i}. {f}")

		while True:
			choice = input(f"Select setup file [1–{len(setup_files)}]: ").strip()
			if choice.isdigit() and 1 <= int(choice) <= len(setup_files):
				filename = setup_files[int(choice) - 1]
				break
			print("⚠️ Invalid selection. Try again.")

	# Load the selected file
	try:
		with open(os.path.join(setup_dir, filename), "rb") as f:
			setup = pickle.load(f)
		print(f"✅ Successfully loaded '{filename}'")
		return filename, setup
	except Exception as e:
		print(f"❌ Failed to load '{filename}': {e}")
		return filename, None

# Main function
def GenHEX(f1, f2, HEX, verbose=True):

	# ----------------------------------Calculations outside iteration----------------------------------
	# Gets frontal areas and whether it's a counter/cross/parallell flow from given flow directions
	get_frontalArea_FlowDirr(HEX)
	get_sigma_alpha_sigmaFTT_Dh(HEX)

	# ----------------------------------Initial conditions----------------------------------
	# gets the required values from coolprop
	get_fluid_properties(f1)
	get_fluid_properties(f2)

	# ----------------------------------Aerothermal iterative loop start----------------------------------
	# (iterating density and enthalpy for mean conditions)
	t = time.time()
	# Could be increased but usually converges before 5 iterations if all is ok
	numIter = 10
	for i in range(numIter):

		if verbose:
			print("Aerothermal iteration",i ,"after", time.time()-t, "s")
			t_i = time.time()
		# ----------------------------------Get f1 conditions----------------------------------
		get_fluid_properties(f1)
		if verbose:
			print("Got all f1 properties in", time.time()-t_i,"s")
			t_i = time.time()

		# ----------------------------------Get f2 conditions----------------------------------
		get_fluid_properties(f2)
		if verbose:
			print("Got all f2 properties in", time.time() - t_i, "s")
			t_i = time.time()


		# ----------------------------------Calculations----------------------------------
		# Reynolds number
		get_Re(f1, f2, HEX)

		HEX.f1_Re = f1.Re_mean[-1]
		HEX.f1_Pr = f1.Pr_mean[-1]

		HEX.f2_Re = f2.Re_mean[-1]
		HEX.f2_Pr = f2.Pr_mean[-1]

		# Baseline heat transfer and pressure loss coefficients
		if HEX.f1_correlation == "Gnielinski":
			# Assume both sides as internal of tubes
			gnielinski(f1)
		elif HEX.f1_correlation == "tubes (blended Gnielinski)":
			# Assume both sides as internal of tubes
			gnielinskiBlended(f1)
		# elif HEX.f1_correlation == "Lienhard":
		# 	# Assume both sides as flat plates
		# 	f1 = Lienhard(f1, HEX.Lx, HEX.f1_Afr, HEX.f1_sigma)
		elif HEX.f1_correlation == "ASME26 correlations":
			f1.D_h = HEX.f1_Dh
			f1.ell = HEX.f1_l_D_h*HEX.f1_Dh
			ASME26_correlations(f1)
		elif HEX.f1_correlation == "KaysAndLondon":
			HEX = KaysAndLondonHEX(HEX, f1, f2)
		elif HEX.f1_correlation == "general (Kays and London correlations)":
			HEX = KL_curvefit(HEX)
		else:
			# Both sides from Kays and London curve-fit correlation
			print("Wrong correlation given, using the correlations derived from Kays and London")
			HEX = KL_curvefit(HEX)

		if HEX.f2_correlation == "Gnielinski":
			# Assume both sides as internal of tubes
			gnielinski(f2)
			HEX.f2_Nu = f2.Nu
			HEX.f2_f_D = f2.f_D
		elif HEX.f2_correlation == "tubes (blended Gnielinski)":
			# Assume both sides as internal of tubes
			gnielinskiBlended(f2)
			HEX.f2_Nu = f2.Nu
			HEX.f2_f_D = f2.f_D
		# elif HEX.f2_correlation == "Lienhard":
		# 	# Assume both sides as flat plates
		# 	Lienhard(f2, HEX.Lx, HEX.f1_Afr, HEX.f1_sigma)
		# 	HEX.f2_Nu = f2.Nu
		# 	HEX.f2_f = f2.f
		elif HEX.f2_correlation == "ASME26 correlations":
			f2.D_h = HEX.f2_Dh
			f2.ell = HEX.f2_l_D_h * HEX.f2_Dh
			ASME26_correlations(f2)
		elif HEX.f2_correlation == "KaysAndLondon":
			KaysAndLondonHEX(HEX, f1, f2)
		elif HEX.f2_correlation == "general (Kays and London correlations)":
			KL_curvefit(HEX)
		else:
			# Both sides from Kays and London curve-fit correlation
			print("Wrong correlation given, using the correlations derived from Kays and London")
			KL_curvefit(HEX)

		HEX.f1_f_D = f1.f_D
		HEX.f2_f_D = f2.f_D

		# Nusselt to heat transfer coefficient
		# HEX.f1_h_ave = HEX.f1_Nu*f1.k_mean[-1]/HEX.f1_Dh
		HEX.f1_h_ave = f1.mu_mean*f1.cp_mean/(f1.Pr_mean**(2/3))*f1.Re_mean/HEX.f1_Dh*f1.j
		# HEX.f2_h_ave = HEX.f2_Nu*f2.k_mean[-1]/HEX.f2_Dh
		HEX.f2_h_ave = f2.mu_mean * f2.cp_mean / (f2.Pr_mean ** (2 / 3)) * f2.Re_mean / HEX.f2_Dh * f2.j

		if verbose:
			print("Calculated heat transfer coefficient in",time.time()-t_i,"s")
			t_i = time.time()

		# Assemble U_f1

		# -----------Where f1 is enhanced side-----------
		if hasattr(HEX, 'FinAR'):
			get_surfaceEfficiencies(HEX)
		else:
			get_surfaceEfficiencies_assumption(HEX)
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
		f1_dT0 = -HEX.eps*(HEX.C_min/HEX.f1_C)*(f1.T0[-2]-f2.T0[-2])
		f2_dT0 = HEX.eps*(HEX.C_min/HEX.f2_C)*(f1.T0[-2]-f2.T0[-2])

		# -----------Calculate pressure loss-----------
		# Based on DARCY friction factors!!!
		f1_dp0 = - 0.5*HEX.f1_n_passes*f1.mdot**2/(f1.rho[-2])*1/(HEX.f1_Afr)**2 * ((1/HEX.f1_sigma**2+1)*(f1.rho[-2]/f1.rho[-1]-1) + HEX.f1_f_D/4*HEX.f1_L*HEX.f1_alpha/HEX.f1_sigma**3*2*f1.rho[-2]/(f1.rho[-2]+f1.rho[-1]))
		f2_dp0 = - 0.5*HEX.f2_n_passes*f2.mdot**2/(f2.rho[-2])*1/(HEX.f2_Afr)**2 * ((1/HEX.f2_sigma**2+1)*(f2.rho[-2]/f2.rho[-1]-1) + HEX.f2_f_D/4*HEX.f2_L*HEX.f2_alpha/HEX.f2_sigma**3*2*f2.rho[-2]/(f2.rho[-2]+f2.rho[-1]))

		if verbose:
			print("pressure loss f1", f1_dp0)
			print("pressure loss f2", f2_dp0)

		# Check convergence
		# T0_f1
		if np.nanmean(1-np.abs((f1.T0[-1]-f1.T0[-2])/f1_dT0))<0.001:
			# T0_f2
			if np.nanmean(1-np.abs((f2.T0[-1]-f2.T0[-2])/f2_dT0))<0.001:
				# p0_f1
				if np.nanmean(np.abs((f1.p0[-1]-f1.p0[-2])-f1_dp0)/f1.p0[-2])<0.01:
					# p0_f2
					if np.nanmean(np.abs((f2.p0[-1]-f2.p0[-2])/f2_dp0)/f2.p0[-2])<0.01:
						f1.T0[-1] = f1.T0[-2] + f1_dT0
						f1.p0[-1] = f1.p0[-2] + f1_dp0
						f2.T0[-1] = f2.T0[-2] + f2_dT0
						f2.p0[-1] = f2.p0[-2] + f2_dp0
						#print(np.mean(f1.T0[-1,...]), np.mean(f2.T0[-1,...]))
						break

		if i == numIter-1:
			print("Did not converge")
			print(np.nanmax(1-np.abs((f1.T0[-1]-f1.T0[-2])/f1_dT0)), np.nanmax(1-np.abs((f2.T0[-1]-f2.T0[-2])/f2_dT0)), np.nanmax(1-np.abs((f1.p0[-1]-f1.p0[-2])/f1_dp0)),np.nanmax(1-np.abs((f2.p0[-1]-f2.p0[-2])/f2_dp0)))
			print("Max dT0_1", np.nanmax(f1.T0[-1]-f1.T0[-2]))
			print("Max dT0_2", np.nanmax(f2.T0[-1]-f2.T0[-2]))
			print("Max dp0_1", np.nanmax(1-np.abs((f1.p0[-1]-f1.p0[-2])/f1.p0[-2])))
			print("Max dp0_2", np.nanmax(1-np.abs((f2.p0[-1]-f2.p0[-2])/f2.p0[-2])))

		f1.T0[-1] = f1.T0[-2]+f1_dT0
		f1.p0[-1] = f1.p0[-2]+f1_dp0
		f2.T0[-1] = f2.T0[-2]+f2_dT0
		f2.p0[-1] = f2.p0[-2]+f2_dp0

		if verbose:
			print("Calculated dT0 and dp0 in",time.time()-t_i,"s")
			t_i = time.time()

	HEX.f1_Q = (f1.T0[-1]-f1.T0[-2])*HEX.f1_C
	HEX.f2_Q = (f2.T0[-1]-f2.T0[-2])*HEX.f2_C

	HEX.mass = HEX.Lx*HEX.Ly*HEX.Lz*HEX.chi*HEX.rho
	HEX.f1_aveMetalWallTemperature = (f1.T0[-1]+f1.T0[-2])/2-HEX.f1_Q/(HEX.Lx*HEX.Ly*HEX.Lz*HEX.f1_alpha)/HEX.f1_h_ave
	HEX.f2_aveMetalWallTemperature = (f2.T0[-1]+f2.T0[-2])/2+HEX.f2_Q/(HEX.Lx*HEX.Ly*HEX.Lz*HEX.f2_alpha)/HEX.f2_h_ave

