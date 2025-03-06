# -*- coding: utf-8 -*-
"""
Last edited: MNS, 4 March 2025.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def generate_spec_mon(params_mon, wn_mon, nEvals, n_vib):
    """
    Generate a simulated monomer spectrum.
    
    Parameters:
        params_mon: array-like, containing [E_central (nEvals elements), E_vib, sigma_mon, gamma_mon, S]
        wn_mon: numpy array of wavenumbers (cm^-1)
        nEvals: number of electronic transitions
        n_vib: number of vibrational levels
        
    Returns:
        fit_mon: simulated (fitted) spectrum (normalized)
        peak_positions: positions of the spectral peaks
        peak_str: relative intensities of each peak (normalized for plotting sticks)
        eta: a weight representing the homogeneous amplitude
    """
    
    # Unpack parameters
    E_central = np.array(params_mon[:nEvals])
    E_vib = params_mon[nEvals]
    sigma_mon = params_mon[nEvals+1]
    gamma_mon = params_mon[nEvals+2]
    S = params_mon[nEvals+3]
    
    # Calculate relative intensities for each vibrational transition
    intensities = []
    peak_positions = []
    for i in range(nEvals):
        for j in range(n_vib):
            # Poisson distribution weight: I ~ exp(-S)*S^j/j!
            intensity = np.exp(-1 * S) * (S**j) / np.math.factorial(j)
            peak = E_central[i] + j * E_vib
            intensities.append(intensity)
            peak_positions.append(peak)
            
    intensities = np.array(intensities)
    
    # For stick plot, normalize the intensities
    peak_str = intensities / np.max(intensities) if np.max(intensities) != 0 else intensities

    fit_mon = np.zeros_like(wn_mon)
    for peak, intensity in zip(peak_positions, intensities):
        
        # Estimate eta as the fraction of homogeneous broadening over total broadening.
        GaussianProf = np.exp(-(wn_mon - peak)**2 / (2 * sigma_mon**2)) / (np.sqrt(2*np.pi) * sigma_mon)
        fwhmG = 2 * np.sqrt(2 * np.log(2)) * sigma_mon
        
        LorentzianProf = (gamma_mon) / (((wn_mon - peak)**2 + gamma_mon**2) * np.pi)
        fwhmL = 2 * gamma_mon
        
        # Creates the pseudo-Voigt profile.
        fwhmTot = (fwhmG**5 + 2.69269 * fwhmG**4 * fwhmL + 2.42843 * fwhmG**3 * fwhmL**2 + 0.07842 * fwhmG * fwhmL**4 + fwhmL**5)**(1/5)
        eta = 1.36603 * (fwhmL / fwhmTot) - 0.47719 * (fwhmL / fwhmTot)**2 + 0.11116 * (fwhmL / fwhmTot)**3
        broadening = (1 - eta) * GaussianProf + eta * LorentzianProf
        fit_mon += intensity * broadening

        
    # Normalize the fitted spectrum
    fit_mon = fit_mon / np.max(fit_mon) if np.max(fit_mon) != 0 else fit_mon
    
    return fit_mon, np.array(peak_positions), peak_str, eta

def monomer_model(params, wn_mon, exp_spec_mon, nEvals, n_vib):
    """
    Calculate the error between the simulated and experimental monomer spectra.
    
    Parameters:
        params: array-like, parameters [E_central (nEvals values), E_vib, sigma_mon, gamma_mon, S]
        wn_mon: wavenumber array (cm^-1)
        exp_spec_mon: experimental spectrum (normalized)
        nEvals: number of electronic transitions
        n_vib: number of vibrational levels
        
    Returns:
        error: sum of squared differences between the simulated and experimental spectrum
        fit_mon: simulated spectrum
        peak_positions: peak positions (for sticks)
        peak_str: corresponding peak intensities
        eta: homogeneous amplitude fraction
    """
    
    # Unpack parameters
    E_central = np.array(params[:nEvals])
    E_vib = params[nEvals]
    sigma_mon = params[nEvals+1]
    gamma_mon = params[nEvals+2]
    S = params[nEvals+3]
    
    # Combine into one parameter array 
    params_mon = np.concatenate((E_central, [E_vib, sigma_mon, gamma_mon, S]))
    fit_mon, peak_positions, peak_str, eta = generate_spec_mon(params_mon, wn_mon, nEvals, n_vib)
    error = np.sum((fit_mon - exp_spec_mon)**2)
    return error, fit_mon, peak_positions, peak_str, eta

def objective_func(params, wn_mon, exp_spec_mon, nEvals, n_vib):
    
    """Objective function wrapper for optimization."""
    err, _, _, _, _ = monomer_model(params, wn_mon, exp_spec_mon, nEvals, n_vib)
    return err

def main():
    # ---------------------------
    # Data Loading and Processing
    # ---------------------------
    
    # Path on local computer where .npz file is saved
    fpath = r'C:\Users\madel\Documents\MIT\Schlau-Cohen group\Research foci\Chlorins\Modelling absorption spectra\20250304 - converting code to Python\G4AbsorptionData.npz'
    
    dataLib = np.load(fpath)
    G4monomer = dataLib['G4monomer']
    
    # Create variables to describe wavelength and absorption spectrum.
    wavelengths = G4monomer[:, 0]  # nm 
    wavelengths = wavelengths[~np.isnan(wavelengths)]
    G4abs = G4monomer[:, 1]
    G4abs = G4abs[~np.isnan(G4abs)]
    
    # Select the wavelength range that frames Soret region
    idx_low = np.argmin(np.abs(wavelengths - 500))
    idx_high = np.argmin(np.abs(wavelengths - 350))
    
    # Ensure proper ordering
    if idx_low > idx_high:
        idx_low, idx_high = idx_high, idx_low
        
    wl_abs = wavelengths[idx_low:idx_high+1]  # wavelength range in nm
    wn_abs = 1.0 / wl_abs * 1e7  # convert to wavenumbers (cm^-1)
    exp_spec_mon = G4abs[idx_low:idx_high+1]
    exp_spec_mon = exp_spec_mon / np.max(exp_spec_mon) if np.max(exp_spec_mon) != 0 else exp_spec_mon
    
    # Plot the experimental spectrum
    plt.figure()
    plt.plot(wn_abs, exp_spec_mon, label='Experimental')
    plt.xlim([22000, 28000])
    plt.ylim([0, 1.15])
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Peak-normalized absorbance')
    plt.xticks(fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('Experimental Monomer Spectrum')
    plt.legend()
    
    # ---------------------------
    # Optimization Setup
    # ---------------------------
    # Hand-picked electronic transitions (E00, E01, …)
    Evals = [23866.3, 23866.3]  # in cm^-1; degenerate transitions
    nEvals = len(Evals)
    
    # Number of vibrational levels
    n_vib = 5
    
    # Parameter vector for monomer only:
    # [E_central (2 values), E_vib, sigma_mon, gamma_mon, S]
    # Initial guess and bounds:
    lb = [23866.3, 23866.3, 0, 0, 0, 0]
    ub = [23866.3, 23866.3, 5000, 10000, 10000, 10]
    bounds = list(zip(lb, ub))
    
    # Initial guess: note that Evals are fixed by the bounds.
    guess = [23866.3, 23866.3, 0, 400, 400, 0.5]
    
    # ---------------------------
    # Global Optimization
    # ---------------------------
    # Use SciPy's differential_evolution for global search.
    result = differential_evolution(objective_func, bounds, args=(wn_abs, exp_spec_mon, nEvals, n_vib), disp=True)
    params_opt = result.x
    fval = result.fun
    
    # Evaluate the fitted spectrum with the optimized parameters.
    error, fit_mon, peak_positions, peak_str, eta = monomer_model(params_opt, wn_abs, exp_spec_mon, nEvals, n_vib)
    
    # Display fitted parameters
    print("Fitted Parameters:")
    print("E_elec (cm^-1):", params_opt[:nEvals])
    print("E_vib (cm^-1):", params_opt[nEvals])
    print("Inhomogeneous broadening (monomer, cm^-1):", params_opt[nEvals+1])
    print("Homogeneous broadening (monomer, cm^-1):", params_opt[nEvals+2])
    print("Inhomogeneous amplitude:", 1 - eta)
    print("Homogeneous amplitude:", eta)
    print("Huang-Rhys factor S:", params_opt[nEvals+3])
    
    # ---------------------------
    # Plotting the Results
    # ---------------------------
    plt.figure()
    plt.plot(wn_abs, exp_spec_mon, color='#bc3908', linewidth=1.5, label='Experimental')
    plt.plot(wn_abs, fit_mon, 'k--', linewidth=0.75, label='Fitted')
    
    # Add vertical lines ("sticks") for peak positions
    for pos, height in zip(peak_positions, peak_str):
        plt.plot([pos, pos], [0, height], 'k-', linewidth=0.75)
    plt.box(True)
    plt.xlim([22000, 28000])
    plt.ylim([0, 1.15])
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Peak-normalized absorbance')
    plt.title('G4 Monomer')
    plt.legend(frameon=False)
    plt.xticks(fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.show()

if __name__ == '__main__':
    main()
