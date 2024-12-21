import numpy as np

class CosmologicalParameters:
    def __init__(self, cosmology='planck', verbose=True):
        """
        Initialize the cosmological parameters for a given cosmology.
        
        Args:
            cosmology (str): Name of the cosmology ('planck' or 'mice').
        """
        if cosmology == 'planck':
            self.H_0 = 67.6  # Hubble constant in km/s/Mpc
            self.h = self.H_0 / 100.0
            
            self.Omega_b = 0.022 / self.h**2
            self.Omega_m = 0.31
            
            self.A_s = 2.02730058 * 10**-9
            self.n_s = 0.97
            self.sigma_8 = 0.8
            
            self.Omega_nu_massive = 0.000644 / self.h**2
            self.num_nu_massive = 1
        
        elif cosmology == 'mice':
            self.H_0 = 70.0  # Hubble constant in km/s/Mpc
            self.h = self.H_0 / 100.0
            
            self.Omega_b = 0.044
            self.Omega_m = 0.25
            
            self.A_s = 2.445 * 10**-9
            self.n_s = 0.95
            self.sigma_8 = 0.8
            
            self.Omega_nu_massive = 0.0
            self.num_nu_massive = 0
        
        else:
            raise ValueError("Cosmology not recognized. Please choose 'planck' or 'mice'.")
        
        if verbose:
            print(f"Initialized cosmology: {cosmology}")
    
    def __repr__(self):
        """String representation for debugging and inspection."""
        return f"Cosmology(H_0={self.H_0}, Omega_m={self.Omega_m}, sigma_8={self.sigma_8})"

class RedshiftDistributions:
    def __init__(self, nz_flag, file_paths=None, verbose=True):
        """
        Initialize the redshift distributions based on the chosen flag.

        Args:
            nz_flag (str): Flag to select the redshift distribution ('fid', 'clusteringz').
            file_paths (dict, optional): Paths to the n(z) files. Default uses internal paths.
            verbose (bool): Whether to print initialization messages.
        """
        # Default file paths
        default_paths = {
            'fid': 'nz/nz_DNFpdf_shift_stretch_wrtclusteringz1-4_wrtVIPERS5-6_v2.txt',
            'clusteringz': 'nz/nz_clusteringz.txt'
        }
        file_paths = file_paths or default_paths
        
        if nz_flag not in file_paths:
            raise ValueError(f"Unknown redshift distribution: {nz_flag}")
        
        try:
            self.nz_data = np.loadtxt(file_paths[nz_flag])
        except OSError as e:
            raise FileNotFoundError(f"Error loading n(z) file for {nz_flag}: {e}")
        
        self.z_edges = {
            'fid': {0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0], 4: [1.0, 1.1], 5: [1.1, 1.2]},
            'clusteringz': {0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0]}
        }[nz_flag]

        self.nbins = len(self.nz_data.T) - 1
        if verbose:
            print(f"Using {nz_flag} n(z), which has {self.nbins} redshift bins")

    def __repr__(self):
        """String representation for debugging."""
        return f"RedshiftDistributions(nbins={self.nbins}, edges={self.z_edges})"

    def nz_interp(self, z, bin_z):
        """Interpolate n(z) for a given redshift z and bin."""
        return np.interp(z, self.nz_data[:, 0], self.nz_data[:, bin_z + 1])

    def z_average(self, bin_z):
        """Calculate the average redshift for a given bin."""
        return np.trapz(self.nz_data[:, 0] * self.nz_data[:, bin_z + 1], self.nz_data[:, 0])

    def z_vector(self, bin_z, Nz=100, verbose=True):
        """Generate a vector of redshift values around the average redshift."""
        z_avg = self.z_average(bin_z)
        z_vector = np.linspace(z_avg - 0.25, z_avg + 0.25, Nz)
        z_values = self.nz_interp(z_vector, bin_z)
        
        if verbose:
            print(f"[bin_z: {bin_z}, z_avg: {z_avg:.3f}, "
                  f"integral of the n(z) (total): {np.trapz(self.nz_data[:, bin_z + 1], self.nz_data[:, 0]):.3f}, "
                  f"integral of the n(z) (over the z range used): {np.trapz(z_values, z_vector):.3f}]")
        
        return z_vector
