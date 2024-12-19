import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

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

class savedir_baofit:
    def __init__(self, include_wiggles, dataset, weight_type, nz_flag, cov_type, cosmology_template, 
                 cosmology_covariance, delta_theta, theta_min, theta_max, n_broadband, bins_removed, verbose=True):
        """
        Initialize the savedir_baofit class to generate a save directory path based on various parameters.

        Args:
            include_wiggles (str): Indicates whether wiggles are included (e.g., "yes" or "no").
            dataset (str): The dataset identifier (e.g., "COLA" or others).
            weight_type (str): The weight type (e.g., "unweighted", "weighted").
            nz_flag (str): The flag used for the n(z) configuration.
            cov_type (str): The type of covariance.
            cosmology_template (str): The cosmology template identifier.
            cosmology_covariance (str): The cosmology covariance.
            delta_theta (float): The delta theta value.
            theta_min (float): The minimum theta value.
            theta_max (float): The maximum theta value.
            n_broadband (int): The number of broadband bins.
            bins_removed (int): The number of bins removed.
            verbose (bool): Whether to print the save directory message. Default is True.
        """
        self.include_wiggles = include_wiggles
        self.dataset = dataset
        self.weight_type = weight_type
        self.nz_flag = nz_flag
        self.cov_type = cov_type
        self.cosmology_template = cosmology_template
        self.cosmology_covariance = cosmology_covariance
        self.delta_theta = delta_theta
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.n_broadband = n_broadband
        self.bins_removed = bins_removed
        self.verbose = verbose

    def __call__(self):
        """
        Generate the save directory path and print the message when the instance is called, 
        if verbose is True.
        """
        if self.dataset == 'COLA':
            savedir_string = (
                f"fit_results{self.include_wiggles}/{self.dataset}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}"
            )
        elif self.dataset == 'DESY6':
            savedir_string = (
                f"fit_results{self.include_wiggles}/{self.dataset}_{self.weight_type}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}"
            )
        
        # Print if verbose is True
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
            str: A sudesign_matrixary of the cosmology parameters.
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

class wtheta_model:
    def __init__(self, alpha_min, alpha_max, n_broadband, theta_wtheta_data, wtheta_th_interp):
        """
        Initialize the wtheta_model class with necessary parameters and settings.

        Args:
            alpha_min (float): Minimum allowed alpha value.
            alpha_max (float): Maximum allowed alpha value.
            n_broadband (int): The number of broadband bins.
            theta_wtheta_data (dict): Dictionary with the values for theta used in each bin.
            wtheta_th_interp (dict): Interpolation functions for wtheta.
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.n_broadband = n_broadband
        self.theta_wtheta_data = theta_wtheta_data
        self.wtheta_th_interp = wtheta_th_interp
        
        # Set nbins based on the length of theta_wtheta_data
        self.nbins = len(self.theta_wtheta_data)

        # Predefined values based on the given context
        self.names_params = np.array([
            'alpha',
            'A_0', 'B_0', 'C_0', 'D_0', 'E_0', 'F_0', 'G_0',
            'A_1', 'B_1', 'C_1', 'D_1', 'E_1', 'F_1', 'G_1',
            'A_2', 'B_2', 'C_2', 'D_2', 'E_2', 'F_2', 'G_2',
            'A_3', 'B_3', 'C_3', 'D_3', 'E_3', 'F_3', 'G_3',
            'A_4', 'B_4', 'C_4', 'D_4', 'E_4', 'F_4', 'G_4',
            'A_5', 'B_5', 'C_5', 'D_5', 'E_5', 'F_5', 'G_5',
        ])
        self.n_broadband_max = int((len(self.names_params) - 1) / 6 - 1)  # This should be 6
        self.n_params_max = len(self.names_params)

        # Initialize the p0 array (initial parameter guesses)
        self.p0_list = [1]
        self.p0 = self._initialize_p0()

        # Bounds for parameters
        self.bounds = self._initialize_bounds()

        # Prepare the index array
        self.indices_params = self._generate_indices()

        # Apply the reduction based on indices_params to names_params, p0, and bounds
        self._apply_reduction()

    def _initialize_p0(self):
        """Initialize the p0 parameter list based on the redshift bins."""
        p0_list = [1]
        for _ in range(self.nbins):
            row = [1] + [0] * self.n_broadband_max
            p0_list.extend(row)
        return np.array(p0_list)

    def _initialize_bounds(self):
        """Initialize bounds for the parameters."""
        return (
            (self.alpha_min, *[0, -1, -1, -1, -1, -1, -1] * self.nbins),
            (self.alpha_max, *[15, 1, 1, 1, 1, 1, 1] * self.nbins)
        )

    def _generate_indices(self):
        """Generate index array based on the number of broadband bins."""
        indices_params_bb = np.arange(1, 1 + (self.n_broadband + 1)) # only nuisance parameters
        indices_params = np.concatenate([[0], indices_params_bb]) # including alpha
        for k in range(1, self.nbins):
            indices_params = np.concatenate([indices_params, indices_params_bb + k * (1 + self.n_broadband_max)])
        return indices_params

    def _apply_reduction(self):
        """Apply the reduction in size of names_params, p0, and bounds based on indices_params."""
        self.names_params = self.names_params[self.indices_params]
        self.p0 = self.p0[self.indices_params]
        self.bounds = (
            list(np.array(self.bounds[0])[self.indices_params]),
            list(np.array(self.bounds[1])[self.indices_params])
        )

    def wtheta_template_raw(self, x, alpha, *params):
        """Theoretical template calculation."""
        wtheta_template = np.concatenate([
            params[(1 + self.n_broadband_max) * i] * self.wtheta_th_interp[i](alpha * self.theta_wtheta_data[i]) +  # A
            params[(1 + self.n_broadband_max) * i + 1] +  # B
            params[(1 + self.n_broadband_max) * i + 2] / self.theta_wtheta_data[i] +  # C
            params[(1 + self.n_broadband_max) * i + 3] / self.theta_wtheta_data[i]**2 +  # D
            params[(1 + self.n_broadband_max) * i + 4] * self.theta_wtheta_data[i] +  # E
            params[(1 + self.n_broadband_max) * i + 5] * self.theta_wtheta_data[i]**2 +  # F
            params[(1 + self.n_broadband_max) * i + 6] * self.theta_wtheta_data[i]**3  # G
            for i in range(self.nbins)
        ])
        return wtheta_template

    def get_wtheta_template(self):
        """Return the wtheta_template function."""
        def wtheta_template(x, *args):
            pars = np.zeros(self.n_params_max)
            pars[self.indices_params] = args
            return self.wtheta_template_raw(x, *pars)

        return wtheta_template

class bao_fit:
    def __init__(self, wtheta_model_instance, wtheta_data, cov, savedir):
        """
        Initialize the BAO fit class.
        
        Args:
            wtheta_model_instance (wtheta_model): Instance of the wtheta_model class.
            wtheta_data_concatenated (array): Observed data for wtheta.
            cov (array): Covariance matrix.
            savedir (str): Directory to save results.
            bins_removed (list): Indices of bins to remove from the fit.
        """
        self.wtheta_model_instance = wtheta_model_instance
        self.wtheta_template = wtheta_model_instance.get_wtheta_template()
        self.wtheta_data = wtheta_data
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)  # Compute the inverse covariance matrix
        self.savedir = savedir

        # Extract parameters from wtheta_model_instance
        self.n_params = len(wtheta_model_instance.names_params)
        self.n_broadband = wtheta_model_instance.n_broadband
        self.nbins = wtheta_model_instance.nbins
        
        # Define wtheta_data_concatenated and bins_removed. I rather define them here than passing them as inputs
        self.wtheta_data_concatenated = np.concatenate([self.wtheta_data[bin_z] for bin_z in range(self.nbins)])
        self.bins_removed = [bin_z for bin_z in range(self.nbins) if np.all(self.wtheta_data[bin_z] == 0)]

        # Precompute positions of amplitude and broadband parameters
        self.pos_amplitude = np.array([1 + i * (self.n_broadband + 1) for i in np.arange(0, self.nbins)])
        self.pos_broadband = np.delete(np.arange(0, self.n_params), np.concatenate(([0], self.pos_amplitude)))

        # Construct the model matrix design_matrix
        self.design_matrix = np.zeros([len(self.wtheta_data_concatenated), self.nbins * self.n_broadband])
        self._construct_design_matrix()

        # Compute the pseudo-inverse of design_matrix
        self.pseudo_inverse_matrix = np.linalg.inv((self.design_matrix.T) @ self.inv_cov @ self.design_matrix) @ (self.design_matrix.T) @ self.inv_cov

    def _construct_design_matrix(self):
        """Construct the model matrix design_matrix."""
        for j in np.arange(0, self.nbins * self.n_broadband):
            fit_params = np.zeros(self.n_params)
            fit_params[0] = 1  # The value of alpha doesn't matter here since the amplitude is 0
            fit_params[self.pos_broadband[j]] = 1
            self.design_matrix[:, j] = self.wtheta_template(self.wtheta_data_concatenated, *fit_params)

    def least_squares(self, params):
        """Least squares function to minimize."""
        y_th = self.wtheta_template(self.wtheta_data_concatenated, *params)
        diff = self.wtheta_data_concatenated - y_th
        return diff @ self.inv_cov @ diff

    def broadband_params(self, amplitude_params, alpha):
        """Compute the broadband parameters."""
        fit_params = np.zeros(self.n_params)
        fit_params[0] = alpha
        fit_params[self.pos_amplitude] = amplitude_params
        return self.pseudo_inverse_matrix @ (self.wtheta_data_concatenated - self.wtheta_template(self.wtheta_data_concatenated, *fit_params))

    def regularized_least_squares(self, amplitude_params, alpha):
        """Least squares function with broadband parameters."""
        fit_params = np.zeros(self.n_params)
        fit_params[0] = alpha
        fit_params[self.pos_amplitude] = amplitude_params
        fit_params[self.pos_broadband] = self.broadband_params(amplitude_params, alpha)

        vector_ones = np.ones(self.nbins)
        vector_ones[self.bins_removed] = 0

        return self.least_squares(fit_params) + np.sum(((amplitude_params - vector_ones) / 0.4) ** 2)

    def fit(self):
        """Perform the fitting procedure to find the best-fit alpha."""
        tol_minimize = 1e-7
        n = 2 * 10**2
        alpha_vector = np.linspace(self.wtheta_model_instance.alpha_min, self.wtheta_model_instance.alpha_max, n)
        chi2_vector = np.zeros(n)

        for i in np.arange(0, n):
            amplitude_params = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                        bounds=([(0, None)] * self.nbins), tol=tol_minimize, args=(alpha_vector[i]))
            chi2_vector[i] = self.regularized_least_squares(amplitude_params.x, alpha_vector[i])

        # Find the best alpha value
        best = np.argmin(chi2_vector)
        alpha_best = alpha_vector[best]
        chi2_best = chi2_vector[best]

        # Save the likelihood data
        np.savetxt(self.savedir + '/likelihood_data.txt', np.column_stack([alpha_vector, chi2_vector]))

        # Compute the error region (1-sigma)
        self._compute_error_region(alpha_vector, chi2_vector, alpha_best, chi2_best)

        return alpha_best, chi2_best

    def _compute_error_region(self, alpha_vector, chi2_vector, alpha_best, chi2_best):
        """Compute the error region for the best alpha value."""
        n = len(alpha_vector)
        alpha_down, alpha_up = None, None

        # Find alpha_down and alpha_up corresponding to chi2_best + 1
        for i in range(len(alpha_vector)):
            if chi2_vector[i] < chi2_best + 1:
                alpha_down = alpha_vector[i]
                break

        for i in range(len(alpha_vector) - 1, -1, -1):
            if chi2_vector[i] < chi2_best + 1:
                alpha_up = alpha_vector[i]
                break

        err_alpha_down = alpha_best - alpha_down
        err_alpha_up = alpha_up - alpha_best
        err_alpha = (alpha_up - alpha_down) / 2

        # Calculate degrees of freedom (dof)
        dof = len(self.wtheta_data_concatenated) - self.n_params
        for bin_z in self.bins_removed:
            dof -= len(self.wtheta_data[bin_z])

        # Plot the chi-squared vs alpha
        fig = plt.figure()
        plt.plot(alpha_vector, chi2_vector)
        plt.plot(alpha_best, chi2_best, 'd')
        plt.plot(alpha_vector, np.ones(n) * (chi2_best + 1), '--r')
        plt.plot((alpha_best - err_alpha_down) * np.ones(n), np.linspace(chi2_vector.min(), chi2_vector.max(), n), '--k')
        plt.plot((alpha_best + err_alpha_up) * np.ones(n), np.linspace(chi2_vector.min(), chi2_vector.max(), n), '--k')
        plt.xlabel(r'$\alpha$', fontsize=14)
        plt.ylabel(r'$\chi^2$', fontsize=14)
        plt.savefig(self.savedir + '/chi2_profile.png', bbox_inches='tight')
        plt.close(fig)

        # Print the results
        print(f'Best-fit alpha = {alpha_best:.4f} ± {err_alpha:.4f}')
        print(f'chi2/dof = {chi2_best:.4f}/{dof}')
