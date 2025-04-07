import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from numpy.linalg import eig


def load_and_process_spectra(G4monomer, G4aggregate):
    # Remove any rows with NaN values
    G4monomer = G4monomer[~np.isnan(G4monomer).any(axis=1)]
    G4aggregate = G4aggregate[~np.isnan(G4aggregate).any(axis=1)]
    
    # Crops monomer spectrum to Soret region
    idx1 = np.abs(G4monomer[:, 0] - 300).argmin()
    idx2 = np.abs(G4monomer[:, 0] - 525).argmin()
    start, end = sorted([idx1, idx2])
    wl_abs = G4monomer[start:end, 0][::-1]  # reverse manually
    wn_abs = 1e7 / wl_abs
    exp_spec_mon = G4monomer[start:end, 1][::-1]
    exp_spec_mon /= exp_spec_mon.max()

    # Crops aggregate spectrum to Soret region
    idx1 = np.abs(G4aggregate[:, 0] - 300).argmin()
    idx2 = np.abs(G4aggregate[:, 0] - 525).argmin()
    start, end = sorted([idx1, idx2])
    wl_abs = G4aggregate[start:end, 0][::-1]  # reverse manually
    wn_abs = 1e7 / wl_abs
    yoff_agg = 0.025; # vertical offset in abs spectrum
    exp_spec_agg = G4aggregate[start:end, 1][::-1] - yoff_agg
    exp_spec_agg /= exp_spec_agg.max()

    return wn_abs, exp_spec_mon, exp_spec_agg


def generate_lattice(nWidth, nLength, nHeight, aDist, bDist, molHeight, posVecDisorder=True):
    # Number of monomers in the aggregate structure
    nMonomers = nWidth * nLength * nHeight
    
    # No. of molecule in each "sheet" of the lattice
    nLayer = nMonomers // nHeight
    
    # COM coordinates for each molecule within the crystal.
    # index 1: [x y z], in angstroms
    # index 2: number of monomers 
    posVec = np.zeros((3, nMonomers))

    # Sets the x-coordinates of the first "sheet" of monomers
    for ww in range(0, nLayer, nWidth):
        wRng = slice(ww, ww + nWidth)
        posVec[0, wRng] = (ww // nWidth) * aDist

    # Sets the y-coordinates of the first "sheet" of monomers
    indY = np.mod(np.arange(nLayer), nLength)
    indY = np.roll(indY, -2)
    posVec[1, :nLayer] = indY * bDist

    # Replicates the x- and y- coordinates to all other "sheets" in the
    # lattice and sets appropriate z-coordinates for all monomers
    for ss in range(nLayer, nMonomers, nLayer):
        posVec[0, ss:ss + nLayer] = posVec[0, :nLayer]
        posVec[1, ss:ss + nLayer] = posVec[1, :nLayer]
        zInd = (ss + nLayer - 1) // nLayer - 1
        posVec[2, ss:ss + nLayer] = zInd * molHeight
        
    if posVecDisorder==True:
        disorder_std = 0.3  # in angstroms
        posVec = posVec + np.random.normal(0, disorder_std, size=posVec.shape)
    else:
        pass    

    return posVec


def compute_theta_matrix(posVec, nMonomers, nEvals, thetaVecDisorder=True):
    # Creates matrix of theta TDM relative angle inputs.
    # Indices: [molecule #, molecule #, Bx/y TDM]
    thetaVals = np.zeros((nMonomers, nMonomers, nEvals))
    for mm in range(nMonomers):
        for nn in range(nMonomers):
            # Coordinates of two molecules under investigation
            x, y, z = posVec[:, nn] - posVec[:, mm]
            
            # Uses trigonometry to find relative TDM angle
            theta_Bx = np.arctan2(y, x) # relative angle of Bx TDM
            theta_By = np.pi / 2 + theta_Bx # relative angle of By TDM
            
            # Saves the angle values as matrix entries
            thetaVals[mm, nn, 0] = theta_Bx
            thetaVals[mm, nn, 1] = theta_By
            
    if thetaVecDisorder==True:
        # Add Gaussian angular noise with std deviation of 5 degrees (~0.087 rad)
        angle_disorder_deg = 10
        angle_disorder_rad = np.deg2rad(angle_disorder_deg)
        thetaVals = thetaVals + np.random.normal(0, angle_disorder_rad, size=thetaVals.shape)
    else:
        pass
        
    return np.nan_to_num(thetaVals)


def generate_spec_aggregate(params, omega_cm, nEvals, n_vib, nMonomers, posVec, thetaVals):
    E_central = params[:nEvals] # Excited state energies in cm^-1
    E_vib = params[nEvals] # Vibrational progression energy in cm^-1
    sigma_cm = params[nEvals + 1] # Inhomogeneous broadening FWHM in cm^-1
    gamma_cm = params[nEvals + 2] # Homogeneous broadening FWHM in cm^-1
    S = params[nEvals + 3] # Huang-Rhys parameter
    phi = params[nEvals + 4] # Phase factor in radians

    # Defines constants for unit conversions
    cLight = 2.99792458e8  # m/s; speed of light
    hPlanck = 1.054571817e-34  # J⋅s; Planck constant
    epsilon0 = 8.8541878188e-12  # F/m; vacuum permittivity
    
    # Dielectric constant of medium
    kappa = 78.4 # unitless, dielectric constant of water at RT
    epsilon = kappa * epsilon0 # F⋅m−1 = C⋅V−1⋅m−1

    # The transition dipole of S0-->Sn excitation
    # (To find numerical magnitude, need to calculate from DFT... but since 
    # it won't influence spectra after normalization, just assume it 
    # to be arbitrary value for now.)
    mu01 = 1  # Debye; (symmetric porphyrins typically < 1 D)
    mu01 = (1 / cLight) * 1e-21 * mu01  # Convert to C·m

    # Franck-Condon overlap integrals
    FC_factors = np.array([(-1)**n * np.exp(-S/2) * S**(n/2) / np.sqrt(factorial(n)) for n in range(n_vib + 1)])
    FC_factors = np.tile(FC_factors, nMonomers)

    # Initialize arrays for: spectrum, energy, oscillator strength, coupling
    spectrum = np.zeros_like(omega_cm)
    energies = []
    Oscillator_strength = []
    Jarray = np.zeros_like(thetaVals)

    # Iterates through the specified potential well minima
    for ee in range(nEvals):
        # Builds vibrational progression around each central energy value
        E_00 = E_central[ee]
        dim = nMonomers * (n_vib + 1)
        
        # Hamiltonian construction for dimer with vibrational levels
        H = np.zeros((dim, dim), dtype=complex)

        # Defines diagonal elements (excited state vibrational levels)
        for mm in range(nMonomers):
            for ii in range(n_vib + 1):
                idx = ii + mm * (n_vib + 1)
                H[idx, idx] = E_00 + ii * E_vib # Molecule 1

        # Defines off-diagonal elements (pair-wise electronic coupling)
        for mm in range(nMonomers): # molecule A
            for nn in range(mm + 1, nMonomers): # molecule B
                # Assigns the position vectors from inputted list
                rA = posVec[:, mm] # [x,y,z] in angstroms
                rB = posVec[:, nn] # [x,y,z] in angstroms

                # Finds the magnitude of COM displacement vector
                Rvec = rB - rA
                Rcom = np.linalg.norm(Rvec) * 1e-10  # in meters

                # Avoid division by zero
                if Rcom == 0:
                    continue

                # Assigns TDM relative angle value
                theta = thetaVals[mm, nn, ee]
                
                # Determines Coulombic coupling from relative TDM orientation angle
                J = mu01**2 * (1 - 3 * np.cos(theta)**2) / (4 * np.pi * epsilon * Rcom**3) # C⋅V = Joule
                J = J / (hPlanck * cLight) # unit of cm-1; 1 cm-1 = E / hc

                # Saves the matrix element of the coupling matrix
                Jarray[mm, nn, ee] = J
                Jarray[nn, mm, ee] = J

                # Evaluates the coupling intensity from FC overlap
                for ii in range(n_vib + 1): # for vibrational excited states of molecule A
                    for jj in range(n_vib + 1): # for vibrational excited states of molecule B
                        idx_mm = ii + mm * (n_vib + 1) 
                        idx_nn = jj + nn * (n_vib + 1) 

                        # Coulombic coupling value
                        val = J * FC_factors[ii + mm * (n_vib + 1)] * FC_factors[jj + nn * (n_vib + 1)]
                        H[idx_mm, idx_nn] = val * np.exp(1j * phi) # Upper triangular components
                        H[idx_nn, idx_mm] = val * np.exp(-1j * phi) # Lower triangular components

        # Diagonalizes the Hamiltonian to find relevant energies
        eigvals, eigvecs = eig(H)
        energies.extend(eigvals.real)

        # Calculates the oscillator strengths
        for n in range(eigvecs.shape[1]):
            f = np.abs(np.dot(FC_factors, eigvecs[:, n]))**2
            Oscillator_strength.append(f)

    # Saves energies and oscillator strengths as arrays
    energies = np.array(energies)
    Oscillator_strength = np.array(Oscillator_strength)

    # Constructs the absorption spectrum, including broadening
    # parameters from the monomer fit.
    for k in range(len(energies)):
        # Creates and normalizes the inhomogeneous broadening component
        GaussianProf = np.exp(-(omega_cm - energies[k])**2 / (2 * sigma_cm**2)) / (np.sqrt(2 * np.pi) * sigma_cm)
        
        # Creates and normalizes the homogeneous broadening component
        LorentzianProf = gamma_cm / ((omega_cm - energies[k])**2 + gamma_cm**2) / np.pi

        # Creates the pseudo-Voigt profile:
        # Approximates amplitude of homogeneous broadening; accurate to within 1% (see Ida, T. et al. J Appl Crystallogr 2000, 33 (6), 1311–1316.).
        fwhmG = 2 * np.sqrt(2 * np.log(2)) * sigma_cm
        fwhmL = 2 * gamma_cm
        fwhmTot = (fwhmG**5 + 2.69269*fwhmG**4*fwhmL + 2.42843*fwhmG**3*fwhmL**2 +
                   0.07842*fwhmG*fwhmL**4 + fwhmL**5)**(1/5)
        eta = 1.36603*(fwhmL/fwhmTot) - 0.47719*(fwhmL/fwhmTot)**2 + 0.11116*(fwhmL/fwhmTot)**3
        
        # Computes the broadened spectrum
        broadening = (1 - eta) * GaussianProf + eta * LorentzianProf
        spectrum += Oscillator_strength[k] * broadening

    # Normalizes the resulting spectrum
    spectrum /= spectrum.max()
    return spectrum, energies, Oscillator_strength, eta, Jarray


def plot_experimental_spectra(wn_abs, exp_spec_mon, exp_spec_agg):
    plt.figure()
    plt.plot(wn_abs, exp_spec_mon, label='Monomer')
    plt.plot(wn_abs, exp_spec_agg, label='Aggregate')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Normalized Absorbance')
    plt.xlim(19050, 28000)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.title("Experimental Spectra")
    plt.tight_layout()
    plt.show()


def plot_theta_map(thetaVals, nMonomers, Btransition=1):
    nMonomers = thetaVals.shape[0]
    xq, yq = np.meshgrid(np.arange(nMonomers), np.arange(nMonomers))
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(xq, yq, np.degrees(thetaVals[:, :, Btransition]), shading='auto', vmin=0, vmax=360)
    cbar=plt.colorbar(label='Relative TDM Angle (degrees)')
    cbar.set_ticks(np.arange(0,360+1,60))
    plt.xlabel('Monomer Index')
    plt.ylabel('Monomer Index')
    
    # Set custom ticks at cell centers
    tickVal = np.arange(nMonomers + 1)
    tick_positions = np.arange(nMonomers) 
    tick_labels = [str(i+1) for i in range(nMonomers)]  # label from 1 to n
    
    plt.xticks(tick_positions, tick_labels)  
    plt.yticks(tick_positions, tick_labels)
    
    plt.title('Soret Transition: B_y' if Btransition == 1 else 'Soret Transition: B_x')
    plt.tight_layout()
    plt.show()


def plot_simulation_results(wn_abs, exp_spec_mon, exp_spec_agg, SimSpectra, lattice_shape):
    plt.figure(figsize=(6, 3))
    plt.plot(wn_abs, exp_spec_mon, 'red', label='Monomer (Expt.)')
    plt.plot(wn_abs, exp_spec_agg, 'blue', label='Aggregate (Expt.)')
    plt.plot(wn_abs, SimSpectra, 'k:', label='Aggregate (Sim.)')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Norm. Absorbance')
    plt.title(f"Lattice size: {lattice_shape}")
    plt.xlim(19500, 28500)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_J_matrix(Jarray, nMonomers, Btransition=1):
    nMonomers = Jarray.shape[0]
    xq, yq = np.meshgrid(np.arange(nMonomers), np.arange(nMonomers))
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(xq, yq, Jarray[:, :, Btransition], shading='auto', cmap='coolwarm', vmin=-300, vmax=300)
    cbar=plt.colorbar(label='J (Coulombic coupling, cm$^{-1}$)')
    cbar.set_ticks(np.arange(-300,300+1,100))
    plt.xlabel('Monomer Index')
    plt.ylabel('Monomer Index')
    
    # Set custom ticks at cell centers
    tickVal = np.arange(nMonomers + 1)
    tick_positions = np.arange(nMonomers) 
    tick_labels = [str(i+1) for i in range(nMonomers)]  # label from 1 to n
    
    plt.xticks(tick_positions, tick_labels)  
    plt.yticks(tick_positions, tick_labels)
    
    plt.title('Coulombic Coupling: B_y' if Btransition == 1 else 'Coulombic Coupling: B_x')
    plt.tight_layout()
    plt.show()


def main():
    # Executes the spectrum simulation.
    # ===
    # A vibronic exciton Hamiltonian is used here (contained in
    # generate_spec_aggregate.m). Note that this model does not account for
    # Herzberg-Teller contributions of the transition dipole moment (i.e.,
    # assumes point-dipole approximation) and, thus, ignores interference
    # between the Franck-Condon and Herzberg-Teller terms. As such, the
    # simulated spectra will likely miss any contributions from the Qx/y
    # transitions, but will enable a reasonable approximation of the Bx/y
    # bands with aggregation.

    # For more information, see: 
    # Kundu, S. et al. J. Phys. Chem. B 2022, 126 (15), 2899–2911. 
    # Roy, P. P. et al. J. Phys. Chem. Lett. 2022, 13 (32), 7413–7419. 


    # === Parameters ===
    dataFiles = np.load(r'C:\Users\madel\Documents\MIT\Schlau-Cohen group\Research foci\Chlorins\Modelling absorption spectra\20250304 - converting code to Python\G4AbsorptionData.npz')
    G4monomer = dataFiles['G4monomer']
    G4aggregate = dataFiles['G4aggregate']
    
    # Hand-pick electronic energy transitions (E00, E01, ... )
    Evals = [23866.3, 23866.3] # cm-1; degenerate B x/y transitions
    
    # From the fit of the monomer absorption spectrum:
    Espace = 1248.8711 # cm-1, vibrational energy level spacing
    sigma = 300.783 # cm-1, inhomogeneous broadening
    gamma = 268.4821 # cm-1, homogeneous broadening
    S = 0.09164 # Huang-Rhys parameter
    phi = 0 # rad; phase factor in Coulombic coupling term
    params = Evals + [Espace, sigma, gamma, S, phi]
    
    # Number of vibrational levels
    n_vib = 5
    
    # No. molecules composing aggregate
    # (Assuming a cubic lattice "box" structure)
    nWidth, nLength, nHeight = 2, 20, 5 # molecules
    
    # Distance between molecular COM's from G4 in THF / H20 XRD data.
    # (Geometric values written w.r.t. monoclinic xtal structure.)
    aDist = 8.5616 # angstroms; length a measurement: 8.5616(13) 
    bDist = 6.2654 # angstroms; length b measurement: 6.2654(8)
    cDist = 26.976 # angstroms; length c measurement: 26.976(4)
    # molHeight = 11.3 # angstroms; approximated from molecular cell volume
    molHeight = 3.374 # angstroms; approximated mon-mon pi-stacking distance
    nMonomers = nWidth * nLength * nHeight # number of monomers in the aggregate

    # Loads and processes experimental spectra
    wn_abs, exp_spec_mon, exp_spec_agg = load_and_process_spectra(G4monomer, G4aggregate)
    #plot_experimental_spectra(wn_abs, exp_spec_mon, exp_spec_agg)
    
    # Initializes the spectrum simulation.
    addDisorder = True
    nSimulations = 10
    
    if addDisorder==True:
        # Binary value to turn on / off disorder in the positions
        posVecDisorder=True
        
        # Binary value to turn on / off disorder in the TDM angle
        thetaVecDisorder=True
        
        # Executes the simulations
        spectra = []
        for _ in range(nSimulations):
            # Initializes positions of all molecules.
            posVec = generate_lattice(nWidth, nLength, nHeight, aDist, bDist, molHeight, posVecDisorder)
            
            # Initializes relative angles between all molecules
            thetaVals = compute_theta_matrix(posVec, nMonomers, len(Evals), thetaVecDisorder)
            
            # Executes one simulation with the set positions / TDM angles
            spec, *_ = generate_spec_aggregate(params, wn_abs, len(Evals), n_vib, nMonomers, posVec, thetaVals)
            spectra.append(spec)
            
        # Ensemble average over all simulations    
        SimSpectra = np.mean(spectra, axis=0)
        
        # Plots the averaged simulated spectrum
        plot_simulation_results(wn_abs, exp_spec_mon, exp_spec_agg, SimSpectra, f"{nWidth} x {nLength} x {nHeight}")
        
        
    else:
        # Initializes positions of all molecules.
        posVec = generate_lattice(nWidth, nLength, nHeight, aDist, bDist, molHeight, posVecDisorder=False)
        
        # Initializes relative angles between all molecules
        thetaVals = compute_theta_matrix(posVec, nMonomers, len(Evals), thetaVecDisorder=False)
        plot_theta_map(thetaVals, nMonomers, Btransition=0)
        
        # Executes the simulation
        SimSpectra, energies, osc_strength, eta, Jarray = generate_spec_aggregate(
        params, wn_abs, len(Evals), n_vib, nMonomers, posVec, thetaVals)
        
        # Plots the simulated spectrum and coupling matrix.
        plot_simulation_results(wn_abs, exp_spec_mon, exp_spec_agg, SimSpectra, f"{nWidth} x {nLength} x {nHeight}")
        plot_J_matrix(Jarray, nMonomers, Btransition=0)

if __name__ == "__main__":
    main()
