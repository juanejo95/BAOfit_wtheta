import numpy as np

class savedir_template:
    def __init__(self, include_wiggles, nz_flag, cosmology_template, verbose=True):
        """
        Initialize the savedir_template class.

        Args:
            include_wiggles (str): Indicates whether wiggles are included (e.g., "yes" or "no").
            nz_flag (str): The flag used for the n(z) configuration.
            cosmology_template (str): The cosmology template identifier.
            verbose (bool): Whether to print the save directory message. Default is True.
        """
        self.include_wiggles = include_wiggles
        self.nz_flag = nz_flag
        self.cosmology_template = cosmology_template
        self.verbose = verbose

    def __call__(self):
        """
        Print a message with the save directory path when the instance is called, 
        if verbose is True.
        """
        savedir_string = f'wtheta_template{self.include_wiggles}/nz_{self.nz_flag}/wtheta_{self.cosmology_template}'
        if self.verbose:
            print(f"Saving output to: {savedir_string}")
        return savedir_string


class cosmology:
    def __init__(self, cosmology='planck'):
        """
        Initialize the cosmological parameters for a given cosmology model.
        
        Args:
            cosmology (str): Name of the cosmology model ('planck' or 'mice').
        """
        if cosmology == 'planck':
            self.H_0 = 67.6  # Hubble constant in km/s/Mpc
            self.h = self.H_0 / 10**2
            
            self.Omega_b = 0.022 / self.h**2
            self.Omega_m = 0.31
            
            self.A_s = 2.02730058 * 10**-9
            self.n_s = 0.97
            
            self.sigma_8 = 0.8
            
            self.Omega_nu_massive = 0.000644 / self.h**2
            self.num_nu_massive = 1
        
        elif cosmology == 'mice':
            self.H_0 = 70  # Hubble constant in km/s/Mpc
            self.h = self.H_0 / 10**2
            
            self.Omega_b = 0.044
            self.Omega_m = 0.25
            
            self.A_s = 2.445 * 10**-9
            self.n_s = 0.95
            
            self.sigma_8 = 0.8
            
            self.Omega_nu_massive = 0
            self.num_nu_massive = 0
        
        else:
            raise ValueError("Cosmology model not recognized. Please choose 'planck' or 'mice'.")
        
        print(f"Initialized cosmology: {cosmology}")
    
    def __repr__(self):
        """
        String representation of the class for debugging and inspection.
        
        Returns:
            str: A summary of the cosmology parameters.
        """
        params = self.__dict__
        return f"Cosmology({params})"

class redshift_distributions:
    def __init__(self, nz_flag):
        # Load the appropriate n(z) data based on nz_flag
        if nz_flag == 'fid':
            self.nz_data = np.loadtxt('nz/nz_DNFpdf_shift_stretch_wrtclusteringz1-4_wrtVIPERS5-6_v2.txt')
            self.z_edges = {
                0: [0.6, 0.7],
                1: [0.7, 0.8],
                2: [0.8, 0.9],
                3: [0.9, 1.0],
                4: [1.0, 1.1],
                5: [1.1, 1.2]
            }
        elif nz_flag == 'fid_5':
            self.nz_data = np.loadtxt('nz/nz_DNFpdf_shift_stretch_wrtclusteringz1-4_wrtVIPERS5-6_v2_5.txt')
            self.z_edges = {
                0: [0.6, 0.7],
                1: [0.7, 0.8],
                2: [0.8, 0.9],
                3: [0.9, 1.0],
                4: [1.0, 1.1]
            }
        else:
            raise ValueError(f"Unknown redshift distributions: {nz_flag}")
        
        # Determine the number of redshift bins
        self.nbins = len(self.nz_data.T) - 1
        print(f'Using {nz_flag} n(z), which has {self.nbins} redshift bins')

    def nz_interp(self, z, bin_z):
        """Interpolate n(z) for a given redshift z and bin."""
        return np.interp(z, self.nz_data[:, 0], self.nz_data[:, bin_z + 1])

    def z_average(self, bin_z):
        """Calculate the average redshift for a given bin."""
        return np.trapz(self.nz_data[:, 0] * self.nz_data[:, bin_z + 1], self.nz_data[:, 0])

    def z_vector(self, bin_z, Nz=100):
        """Generate a vector of redshift values around the average redshift."""
        z_avg = self.z_average(bin_z)
        z_vector = np.linspace(z_avg - 0.25, z_avg + 0.25, Nz)
        z_values = self.nz_interp(z_vector, bin_z)

        print(f"[bin_z: {bin_z}, z_avg: {z_avg:.3f}, "
              f"integral of the n(z) (total): {np.trapz(self.nz_data[:, bin_z + 1], self.nz_data[:, 0]):.3f}, "
              f"integral of the n(z) (over the z range used): {np.trapz(z_values, z_vector):.3f}]")
        
        return z_vector
