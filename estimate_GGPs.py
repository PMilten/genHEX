from GenHEX_functions import *
from openpyxl import load_workbook

# -------------------------------------------------------
# Load setup
# -------------------------------------------------------
class Fluid:
    """Fluid parameters."""
    pass

class HeatExchanger:
    """Geometric, material, and general setup."""
    pass

def estimate_GGPS(setup):
    f1 = setup["f1"]
    f2 = setup["f2"]
    HEX = setup["HEX"]

    print(f"📂 Loaded setup for {f1.fluid} → {f2.fluid}")

    # -------------------------------------------------------
    # Step 1: Get fluid properties at inlets
    # -------------------------------------------------------

    # Primary fluid
    get_fluid_properties(f1)

    # Secondary fluid
    if f2.fluid == 'oil':
        if np.size(f2.T0_out) == 1:
            f2.T0_out = np.append(f2.T0, f2.T0[0] + 1)
        if np.size(f2.p0) == 1:
            f2.p0 = np.append(f2.p0,f2.p0[0] * 0.995)
        f2.cp_mean = np.array([2300])
        f2.rho_mean = np.array([824])
        f2.Pr_mean = np.array([238])
        f2.mu_mean = np.array([0.015])
    else:
        get_fluid_properties(f2)

    # -------------------------------------------------------
    # Step 2: Calculate heat capacity rates
    # -------------------------------------------------------
    get_C_Cmin_Cr(HEX, f1, f2)

    print("Heat capacity rates:")
    print(f"  C₁ = {HEX.f1_C[0]:.2f} W/K")
    print(f"  C₂ = {HEX.f2_C[0]:.2f} W/K")
    print(f"  Cr = {HEX.C_r[0]:.2f}\n")

    # -------------------------------------------------------
    # Step 3: Estimate heat transfer rate Q
    # -------------------------------------------------------
    HEX.Q = HEX.eps * HEX.C_min * np.abs(f1.T0[-2] - f2.T0[-2])
    print(f"🔥 Heat transfer rate Q = {HEX.Q[0]:.2f} W\n")

    # Outlet temperatures
    f1.T0[-1] = f1.T0[-2] - np.sign(f1.T0[-2] - f2.T0[-2]) * HEX.Q[-1] / HEX.f1_C[-1]
    f2.T0[-1] = f2.T0[-2] + np.sign(f1.T0[-2] - f2.T0[-2]) * HEX.Q[-1] / HEX.f2_C[-1]

    # Average properties
    get_fluid_properties(f1)
    if f2.fluid == 'oil':
        if np.size(f2.T0) == 1:
            f2.T0 = np.append(f2.T0, f2.T0[0] + 1)
        if np.size(f2.p0) == 1:
            f2.p0 = np.append(f2.p0, f2.p0[0] * 0.995)
        f2.cp_mean = np.array([2300])
        f2.rho_mean = np.array([824])
        f2.Pr_mean = np.array([238])
        f2.mu_mean = np.array([0.015])
    else:
        get_fluid_properties(f2)


    print(f"Primary fluid ({f1.fluid}) mean properties: ρ={f1.rho_mean[0]:.2f} kg/m³, cp={f1.cp_mean[0]:.1f} J/kg·K, Pr={f1.Pr_mean[0]:.3f}")
    print(f"Secondary fluid ({f2.fluid}) mean properties: ρ={f2.rho_mean[0]:.2f} kg/m³, cp={f2.cp_mean[0]:.1f} J/kg·K, Pr={f2.Pr_mean[0]:.3f}\n")

    # Get flow directions, frontal areas and arrangement
    get_frontalArea_FlowDirr(HEX)

    # -------------------------------------------------------
    # Step 4: Calculate NTU and UA, given eps
    # -------------------------------------------------------

    get_NTU_from_effectiveness(HEX)
    HEX.UA = HEX.NTU * HEX.C_min

    print(f"NTU ({HEX.flow}-flow) = {HEX.NTU[0]:.3f}\n")

    def solve_GGPS_numeric(HEX, f1, f2):
        # Calculate overall surface efficiency

        HEX.f1_eta_o = min(1 - (1 - HEX.alpha_r ** (-1)) * (1 - HEX.eta_f), 1)
        HEX.f2_eta_o = min(1 - (1 - HEX.alpha_r ** (1)) * (1 - HEX.eta_f), 1)

        get_sigma_alpha_sigmaFTT_Dh(HEX)

        # Calculate Reynolds numbers
        get_Re(f1, f2, HEX)

        if HEX.f1_correlation == "Gnielinski":
            # Assume both sides as internal of tubes
            gnielinski(f1)
        elif HEX.f1_correlation == "tubes (blended Gnielinski)":
            # Assume both sides as internal of tubes
            gnielinskiBlended(f1)
        elif HEX.f1_correlation == "ASME26 correlations":
            f1.D_h = HEX.f1_Dh
            f1.ell = HEX.f1_l_D_h*HEX.f1_Dh
            ASME26_correlations(f1)
        else:
            print("Unknown correlation")

        if HEX.f2_correlation == "Gnielinski":
            # Assume both sides as internal of tubes
            gnielinski(f2)
        elif HEX.f2_correlation == "tubes (blended Gnielinski)":
            # Assume both sides as internal of tubes
            gnielinskiBlended(f2)
        elif HEX.f2_correlation == "ASME26 correlations":
            f2.D_h = HEX.f2_Dh
            f2.ell = HEX.f2_l_D_h*HEX.f2_Dh
            ASME26_correlations(f2)
        else:
            print("Unknown correlation")

        # Assumption equations
        sigma_r = (
                    (f1.mdot * HEX.f1_L) / (f2.mdot * HEX.f2_L) *
                    (
                        (f2.cp_mean * f2.mdot) / ( f1.cp_mean * f1.mdot) *
                        1/HEX.R_h *
                        HEX.f1_n_passes / HEX.f2_n_passes *
                        HEX.f2_eta_o / HEX.f1_eta_o *
                        (( f2.rho_mean / (f2.Pr_mean ** (2 / 3)) / ( f1.rho_mean / (f1.Pr_mean ** (2 / 3))))) *
                        (f2.j / f2.f_D) / (f1.j / f1.f_D)
                    ) ** 0.5)
        alpha_r = (sigma_r *
                   HEX.f2_eta_o / HEX.f1_eta_o *
                   ( ((f2.cp_mean / (f2.Pr_mean ** (2 / 3))) / (f1.cp_mean / (f1.Pr_mean ** (2 / 3))))) *
                   ( f2.mdot * HEX.f2_L) / (f1.mdot * HEX.f1_L) *
                   f2.j / f1.j)
        k = HEX.UA * HEX.t * (alpha_r + 1) / (2 * (1 + sigma_r)) * ( sigma_r / alpha_r * HEX.f2_L / HEX.f1_L * ( f1.Pr_mean ** (2 / 3) / (HEX.f1_eta_o * f1.cp_mean * f1.mdot * f1.j)) + ( f2.Pr_mean ** (2 / 3) / (HEX.f2_eta_o * f2.cp_mean * f2.mdot * f2.j)))
        chi = (1 / (HEX.f2_L / k + 1))

        sigma_r_2 = (alpha_r*
                     (f1.mdot*HEX.f1_L)**2/(f2.mdot*HEX.f2_L)**2*
                     HEX.f1_L/HEX.f2_L*
                     f2.rho_mean/f1.rho_mean*
                     1/HEX.R_h*
                     HEX.f1_n_passes / HEX.f2_n_passes *
                     f1.f_D/f2.f_D)**(1/3)

        # Estimate pressure losses - DARCY friction factor!
        HEX.f1_dp = HEX.f1_n_passes*f1.mdot ** 2 * HEX.f1_L ** 3 * f1.f_D * HEX.f1_alpha / (8 * f1.rho_mean * HEX.f1_sigma ** 3 * HEX.Vt ** 2)
        HEX.f2_dp = HEX.f2_n_passes*f2.mdot ** 2 * HEX.f2_L ** 3 * f2.f_D * HEX.f2_alpha / (8 * f2.rho_mean * HEX.f2_sigma ** 3 * HEX.Vt ** 2)

        return sigma_r, alpha_r, chi, sigma_r_2

    print("Estimating GGPs:")
    for i in range(20):
        print(f"Iteration {i:.0f}: sigma_r={HEX.sigma_r[0]:.4f}, alpha_r={HEX.alpha_r[0]:.4f}, chi={HEX.chi[0]:.4f}")
        sigma_r_val, alpha_r_val, chi_val, sigma_r_2_val = solve_GGPS_numeric(HEX, f1, f2)

        if np.abs(1-sigma_r_val/HEX.sigma_r)<0.001 and np.abs(1-alpha_r_val/HEX.alpha_r)<0.001 and np.abs(1-chi_val/HEX.chi)<0.001:
            print(f"GGP estimation converged to target accuracy of 0.1% in {i:.0f} iterations")
            HEX.sigma_r = (sigma_r_val+HEX.sigma_r)/2
            HEX.alpha_r = (alpha_r_val+HEX.alpha_r)/2
            HEX.chi = (chi_val+HEX.chi)/2
            break

        HEX.sigma_r = sigma_r_val
        HEX.alpha_r = alpha_r_val
        HEX.chi = chi_val

    print("Pressure losses estimated as:")
    print(f"primary fluid Dp = {HEX.f1_dp[-1]:.3f} [Pa], {HEX.f1_dp[-1]/f1.p0[-2]*100:.3f} [%]")
    print(f"secondary fluid Dp = {HEX.f2_dp[-1]:.3f} [Pa], {HEX.f2_dp[-1]/f2.p0[-2]*100:.3f} [%]\n")

    HEX.mass = HEX.Vt*HEX.chi*HEX.rho

    print(f"Heat exchanger mass: {HEX.mass[0]:.3f} [kg]")

    # -------------------------------------------------------
    # Step 5: Save results
    # -------------------------------------------------------
    results = {
        "f1": f1,
        "f2": f2,
        "HEX": HEX,
        "Q": HEX.Q,
        "C_min": HEX.C_min,
        "C_r": HEX.C_r,
        "NTU": HEX.NTU,
        "UA": HEX.UA,
    }

    setup = {"f1": f1, "f2": f2, "HEX": HEX}
    # with open(filename, "wb") as f:
    #     pickle.dump(setup, f)

    # print("\n✅ GGPs estimated and saved to "+ filename+":")
    # print("  → f1, f2, and HEX objects are included.\n")

    HEX.f1_Aff = HEX.f1_Afr*HEX.f1_sigma
    HEX.f2_Aff = HEX.f2_Afr*HEX.f2_sigma
    HEX.f1_A = HEX.Vt*HEX.f1_alpha
    HEX.f2_A = HEX.Vt*HEX.f2_alpha
    HEX.f1_A_f = np.max([HEX.f1_A-HEX.f2_A,0])
    HEX.f2_A_f = np.max([HEX.f2_A-HEX.f1_A,0])

    print("HEX.f1_Aff =", HEX.f1_Aff)
    print("HEX.f2_Aff =", HEX.f2_Aff)
    print("HEX.f1_A   =", HEX.f1_A)
    print("HEX.f2_A   =", HEX.f2_A)
    print("HEX.f1_A_f =", HEX.f1_A_f)
    print("HEX.f2_A_f =", HEX.f2_A_f)
    print("HEX.f1_eta_o =", HEX.f1_eta_o)
    print("HEX.f2_eta_o =", HEX.f2_eta_o)
    #print("HEX.f1_fs =", HEX.tw/HEX.f1_Dh*2*290*10**6/np.abs(f1.p0[0]-f2.p0[0]))
    print("HEX.f1_fs =", HEX.tw/(4*HEX.f1_sigma*HEX.Vt/(HEX.f1_A-max([HEX.f1_A_f,0])))*2*290*10**6/np.abs(f1.p0[0]-f2.p0[0]))
    print("HEX.f2_fs =", HEX.tw/(4*HEX.f2_sigma*HEX.Vt/(HEX.f2_A-max([HEX.f2_A_f,0])))*2*290*10**6/np.abs(f1.p0[0]-f2.p0[0]))

    h_f1 = f1.cp_mean/f1.Pr_mean*f1.mdot*HEX.f1_L/(HEX.Vt*HEX.f1_sigma)*f1.j
    h_f2 = f2.cp_mean/f2.Pr_mean*f2.mdot*HEX.f2_L/(HEX.Vt*HEX.f2_sigma)*f2.j

    return setup