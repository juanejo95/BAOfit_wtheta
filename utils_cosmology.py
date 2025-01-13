class CosmologicalParameters:
    def __init__(self, cosmology="planck", verbose=True):
        """
        Initialize the cosmological parameters for a given cosmology.
        
        Parameters:
        - cosmology (str): Name of the cosmology ("planck" or "mice").
        """
        if cosmology == "planck":
            self.H_0 = 67.6  # Hubble constant in km/s/Mpc
            self.h = self.H_0 / 100.0
            
            self.Omega_b = 0.022 / self.h**2
            self.Omega_m = 0.31
            
            self.A_s = 2.02730058 * 10**-9
            self.n_s = 0.97
            self.sigma_8 = 0.8
            
            self.Omega_nu_massive = 0.000644 / self.h**2
            self.num_nu_massive = 1
        
        elif cosmology == "mice":
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
            print(f"Initialized cosmology: {cosmology}.")
