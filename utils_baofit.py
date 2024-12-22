import numpy as np
import os
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from pathos.multiprocessing import ProcessingPool as Pool
import itertools
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
from utils_data import RedshiftDistributions
from utils_template import TemplateInitializer

class WThetaModel:
    def __init__(self, include_wiggles, nz_flag, cosmology_template, n_broadband, galaxy_bias):
        """
        Initialize the WThetaModel class with parameters and settings for modeling 
        the angular correlation function, w(θ).

        Args:
            include_wiggles (str): Specifies whether the power spectrum includes BAO wiggles.
            nz_flag (str): Configuration flag for the n(z) number density distribution.
            cosmology_template (str): Identifier for the cosmology template used.
            n_broadband (int): Number of broadband terms in the model.
            galaxy_bias (dict): Dictionary containing the linear galaxy bias for each 
                redshift bin.
        """
        self.include_wiggles = include_wiggles
        self.nz_flag = nz_flag
        self.cosmology_template = cosmology_template
        self.galaxy_bias = galaxy_bias
        self.n_broadband = n_broadband
        
        self.alpha_min = 0.8
        self.alpha_max = 1.2
        
        self.path_template = TemplateInitializer(
            include_wiggles=self.include_wiggles,
            nz_flag=self.nz_flag,
            cosmology_template=self.cosmology_template,
            verbose=False,
        ).path_template
        
        nz_instance = RedshiftDistributions(self.nz_flag)
        self.nbins = nz_instance.nbins

        # Predefined values based on the given context
        self.names_params = np.array([
            'alpha', 'A_0', 'B_0', 'C_0', 'D_0', 'E_0', 'F_0', 'G_0', 
            'A_1', 'B_1', 'C_1', 'D_1', 'E_1', 'F_1', 'G_1', 
            'A_2', 'B_2', 'C_2', 'D_2', 'E_2', 'F_2', 'G_2', 
            'A_3', 'B_3', 'C_3', 'D_3', 'E_3', 'F_3', 'G_3', 
            'A_4', 'B_4', 'C_4', 'D_4', 'E_4', 'F_4', 'G_4', 
            'A_5', 'B_5', 'C_5', 'D_5', 'E_5', 'F_5', 'G_5'
        ])
        self.n_broadband_max = int((len(self.names_params) - 1) / 6 - 1)  # This should be 6 (from B to G)
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

        # Interpolation of the theoretical wtheta
        self.wtheta_th_interp = self._load_and_interpolate_wtheta()
        
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
        indices_params_bb = np.arange(1, 1 + (self.n_broadband + 1))  # Only nuisance parameters
        indices_params = np.concatenate([[0], indices_params_bb])  # Including alpha
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

    def _load_and_interpolate_wtheta(self):
        """Load and interpolate the wtheta theoretical data."""
        wtheta_th_interp = {}
        for bin_z in range(self.nbins):
            # Load the theoretical wtheta for each bin
            wtheta_bb = np.loadtxt(f'{self.path_template}/wtheta_bb_bin{bin_z}.txt')[:, 1]
            wtheta_bf = np.loadtxt(f'{self.path_template}/wtheta_bf_bin{bin_z}.txt')[:, 1]
            wtheta_ff = np.loadtxt(f'{self.path_template}/wtheta_ff_bin{bin_z}.txt')[:, 1]

            # Combine these into the final wtheta model for the bin
            wtheta_combined = (
                self.galaxy_bias[bin_z]**2 * wtheta_bb + 
                self.galaxy_bias[bin_z] * wtheta_bf + 
                wtheta_ff
            )
            
            # Interpolate the combined wtheta
            theta_values = np.loadtxt(f'{self.path_template}/wtheta_bb_bin{bin_z}.txt')[:, 0]
            wtheta_th_interp[bin_z] = interp1d(theta_values, wtheta_combined, kind='cubic')

        return wtheta_th_interp

    def wtheta_template_raw(self, theta, alpha, *params):
        """Theoretical template calculation."""
        wtheta_template = np.concatenate([
            params[(1 + self.n_broadband_max) * i] * self.wtheta_th_interp[i](alpha * theta) +  # A
            params[(1 + self.n_broadband_max) * i + 1] +  # B
            params[(1 + self.n_broadband_max) * i + 2] / theta +  # C
            params[(1 + self.n_broadband_max) * i + 3] / theta**2 +  # D
            params[(1 + self.n_broadband_max) * i + 4] * theta +  # E
            params[(1 + self.n_broadband_max) * i + 5] * theta**2 +  # F
            params[(1 + self.n_broadband_max) * i + 6] * theta**3  # G
            for i in range(self.nbins)
        ])
        return wtheta_template

    def get_wtheta_template(self):
        """Return the wtheta_template function."""
        def wtheta_template(theta, *args):
            pars = np.zeros(self.n_params_max)
            pars[self.indices_params] = args
            return self.wtheta_template_raw(theta, *pars)

        return wtheta_template

class BAOFitInitializer:
    def __init__(self, include_wiggles, dataset, weight_type, nz_flag, cov_type, cosmology_template,
                 cosmology_covariance, delta_theta, theta_min, theta_max, n_broadband, bins_removed, verbose=True):
        """
        Initializes the BAOFitInitializer with parameters to generate the path for saving results.

        Parameters:
        - include_wiggles: Whether to include BAO wiggles.
        - dataset: Dataset identifier (e.g., "COLA", "DESY6").
        - weight_type: Weight type (e.g., "unweighted", "weighted").
        - nz_flag: Identifier for the n(z).
        - cov_type: Type of covariance.
        - cosmology_template: Identifier for the cosmology template.
        - cosmology_covariance: Type of cosmology covariance.
        - delta_theta: Delta theta value.
        - theta_min: Minimum theta value.
        - theta_max: Maximum theta value.
        - n_broadband: Number of broadband bins.
        - bins_removed: None, 012, 345, etc.
        - verbose: Whether to print messages.
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
        self.verbose = verbose

        # Initialize Redshift Distribution
        self.nz_instance = RedshiftDistributions(self.nz_flag, verbose=False)
        self.nbins = self.nz_instance.nbins

        # Map bins_removed to their corresponding combinations
        self.bins_removed = self._map_bins_removed(bins_removed)

        # Generate the path for saving results
        self.path_baofit = self._generate_path_baofit()

        # Create the directory if it does not exist
        os.makedirs(self.path_baofit, exist_ok=True)
        if verbose:
            print(f"Saving output to: {self.path_baofit}")

    def _map_bins_removed(self, bins_removed):
        """Convert bins_removed string into corresponding bin combinations."""
        def generate_bin_mappings():
            bin_mappings = {'None': []}
            for i in range(1, self.nbins):
                for combo in itertools.combinations(range(self.nbins), i):
                    key = ''.join(map(str, combo))
                    bin_mappings[key] = list(combo)
            return bin_mappings

        bin_mappings = generate_bin_mappings()
        return bin_mappings.get(bins_removed, bins_removed)

    def _generate_path_baofit(self):
        """Generate the save path for the BAO fit results."""
        if self.dataset == 'COLA':
            path = (
                f"fit_results{self.include_wiggles}/{self.dataset}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}"
            )
        elif self.dataset == 'DESY6':
            path = (
                f"fit_results{self.include_wiggles}/{self.dataset}_{self.weight_type}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}"
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        return path

    def get_path_baofit(self):
        """Return the generated path for saving BAO fit results."""
        return self.path_baofit

class BAOFit:
    def __init__(self, baofit_initializer, wtheta_model, theta_data, wtheta_data, cov, n_cpu=None):
        """
        Initialize the BAO fit class.

        Args:
            baofit_initializer: Instance of the BAOFitInitializer class.
            wtheta_model: Instance of the WThetaModel class.
            theta_data (array): Theta data for fitting.
            wtheta_data (list of arrays): Observed wtheta data for each bin.
            cov (array): Covariance matrix.
            n_cpu: Number of CPUs for parallel processing (default: 20).
        """
        self.wtheta_model = wtheta_model
        self.wtheta_template = wtheta_model.get_wtheta_template()
        self.theta_data = theta_data
        self.wtheta_data = wtheta_data
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)  # Compute the inverse covariance matrix
        self.n_broadband = wtheta_model.n_broadband
        self.nbins = wtheta_model.nbins
        # self.n_cpu = n_cpu if n_cpu is not None else multiprocessing.cpu_count()
        self.n_cpu = n_cpu if n_cpu is not None else 20 # Just in case...

        # Retrieve path and bins_removed from PathBAOFit
        self.path_baofit = baofit_initializer.get_path_baofit()
        self.bins_removed = baofit_initializer.bins_removed

        # Concatenate wtheta data
        self.wtheta_data_concatenated = np.concatenate([self.wtheta_data[bin_z] for bin_z in range(self.nbins)])

        # Number of parameters
        self.n_params = len(wtheta_model.names_params)
        self.n_params_true = len(wtheta_model.names_params) - (1 + self.n_broadband) * len(self.bins_removed)

        # Precompute positions of amplitude and broadband parameters
        self.pos_amplitude = np.array([1 + i * (self.n_broadband + 1) for i in range(self.nbins)])
        self.pos_broadband = np.delete(np.arange(self.n_params), np.concatenate(([0], self.pos_amplitude)))

        # Construct the design matrix
        self.design_matrix = np.zeros([len(self.wtheta_data_concatenated), self.nbins * self.n_broadband])
        self._construct_design_matrix()

        # Compute the pseudo-inverse of the design matrix
        self.pseudo_inverse_matrix = (
            np.linalg.inv((self.design_matrix.T @ self.inv_cov @ self.design_matrix))
            @ (self.design_matrix.T @ self.inv_cov)
        )
    
    def _construct_design_matrix(self):
        """Construct the model matrix."""
        for j in np.arange(0, self.nbins * self.n_broadband):
            fit_params = np.zeros(self.n_params)
            fit_params[0] = 1
            fit_params[self.pos_broadband[j]] = 1
            self.design_matrix[:, j] = self.wtheta_template(self.theta_data, *fit_params)

    def least_squares(self, params):
        """Least squares function to minimize."""
        wtheta_th = self.wtheta_template(self.theta_data,*params)
        diff = self.wtheta_data_concatenated - wtheta_th
        return diff @ self.inv_cov @ diff

    def broadband_params(self, amplitude_params, alpha):
        """Compute the broadband parameters."""
        fit_params = np.zeros(self.n_params)
        fit_params[0] = alpha
        fit_params[self.pos_amplitude] = amplitude_params
        return self.pseudo_inverse_matrix @ (self.wtheta_data_concatenated - self.wtheta_template(self.theta_data,*fit_params))

    def regularized_least_squares(self, amplitude_params, alpha):
        """Least squares with broadband parameters."""
        fit_params = np.zeros(self.n_params)
        fit_params[0] = alpha
        fit_params[self.pos_amplitude] = amplitude_params
        fit_params[self.pos_broadband] = self.broadband_params(amplitude_params, alpha)

        vector_ones = np.ones(self.nbins)
        vector_ones[self.bins_removed] = 0
        return self.least_squares(fit_params) + np.sum(((amplitude_params - vector_ones) / 0.4) ** 2)

    def fit(self):
        """Perform the fitting procedure to find the best-fit alpha."""
        tol_minimize = 10**-7
        n = 10**4
        alpha_vector = np.linspace(self.wtheta_model.alpha_min, self.wtheta_model.alpha_max, n)
        chi2_vector = np.zeros(n)

        def compute_chi2(alpha):
            amplitude_params = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                        bounds=[(0, None)] * self.nbins, tol=tol_minimize, args=(alpha,))
            chi2_value = self.regularized_least_squares(amplitude_params.x, alpha)
            return chi2_value

        with Pool(self.n_cpu) as pool: # using multiprocessing with pathos...
            chi2_vector = np.array(pool.map(compute_chi2, alpha_vector))
            
        best = np.argmin(chi2_vector)
        alpha_best = alpha_vector[best]
        chi2_best = chi2_vector[best]

        np.savetxt(self.path_baofit + '/likelihood_data.txt', np.column_stack([alpha_vector, chi2_vector]))

        # Check the chi2 values at the extremes of alpha range
        if chi2_vector[0] > chi2_best + 1 and chi2_vector[-1] > chi2_best + 1:
            # Search for alpha_down and alpha_up where chi2 crosses chi2_best + 1
            alpha_down = None
            alpha_up = None

            # Search for alpha_down where chi2 cuts chi2_best + 1 from the left
            for i in np.arange(0, best)[::-1]:
                if chi2_vector[i] < chi2_best + 1 and chi2_vector[i - 1] > chi2_best + 1:
                    alpha_down = alpha_vector[i]
                    break

            # Search for alpha_up where chi2 cuts chi2_best + 1 from the right
            for i in np.arange(best, n):
                if chi2_vector[i] < chi2_best + 1 and chi2_vector[i + 1] > chi2_best + 1:
                    alpha_up = alpha_vector[i]
                    break

            # Compute the error region (1-sigma)
            err_alpha = (alpha_up - alpha_down) / 2

            # Final amplitude and broadband parameters for the best-fit alpha
            amplitude_params_best = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                            bounds=[(0, None)] * self.nbins, tol=tol_minimize, args=(alpha_best)).x

            broadband_params_best = self.broadband_params(amplitude_params_best, alpha_best)

            params_best = np.zeros(self.n_params)
            params_best[0] = alpha_best
            params_best[self.pos_amplitude] = amplitude_params_best
            params_best[self.pos_broadband] = broadband_params_best

            theta_data_interp = np.linspace(self.theta_data[0], self.theta_data[-1], 10**3)

            wtheta_fit_best = self.wtheta_template(theta_data_interp, *params_best)

            # Plot the w(theta)
            fig, axs = plt.subplots(self.nbins, 1, figsize=(8, 2 * self.nbins), sharex=True)
            for bin_z in range(self.nbins):
                ax = axs[bin_z]
                ax.errorbar(
                    self.theta_data * 180 / np.pi,
                    100 * (self.theta_data * 180 / np.pi) ** 2 * self.wtheta_data[bin_z],
                    yerr=100 * (self.theta_data * 180 / np.pi) ** 2 * np.sqrt(np.diag(self.cov))[bin_z * len(self.theta_data):(bin_z + 1) * len(self.theta_data)],
                    fmt='.', capsize=3
                )
                ax.plot(
                    theta_data_interp * 180 / np.pi, 
                    100 * (theta_data_interp * 180 / np.pi) ** 2 * wtheta_fit_best[bin_z * len(theta_data_interp):(bin_z + 1) * len(theta_data_interp)]
                )
                ax.set_ylabel(r'$10^2 \times \theta^2w(\theta)$ (deg$^2$)', fontsize=13)
                ax.set_title(f'Bin {bin_z}', fontsize=14)
                if bin_z == self.nbins - 1:
                    ax.set_xlabel(r'$\theta$ (deg)', fontsize=13)
            plt.tight_layout()
            plt.savefig(self.path_baofit + '/wtheta_data_bestfit.png', bbox_inches='tight')
            plt.close(fig)

            # Save the w(theta)
            np.savetxt(
                self.path_baofit + '/wtheta_data_bestfit.txt',
                np.column_stack([
                    np.concatenate([self.theta_data] * self.nbins),
                    self.wtheta_data_concatenated,
                    self.wtheta_template(self.theta_data, *params_best),
                    np.sqrt(np.diag(self.cov))
                ])
            )

            # Plot the chi2 vs alpha
            fig = plt.figure()
            plt.plot(alpha_vector, chi2_vector)
            plt.plot(alpha_best, chi2_best, 'd')
            plt.plot(alpha_vector, np.ones(n) * (chi2_best + 1), '--r')
            plt.plot((alpha_best - (alpha_best - alpha_down)) * np.ones(n), np.linspace(chi2_vector.min(), chi2_vector.max(), n), '--k')
            plt.plot((alpha_best + (alpha_up - alpha_best)) * np.ones(n), np.linspace(chi2_vector.min(), chi2_vector.max(), n), '--k')
            plt.xlabel(r'$\alpha$', fontsize=14)
            plt.ylabel(r'$\chi^2$', fontsize=14)
            plt.savefig(self.path_baofit + '/chi2_profile.png', bbox_inches='tight')
            plt.close(fig)

        else:
            print('The fit does not have the 1-sigma region between ' + str(self.wtheta_model.alpha_min) + ' and ' + str(self.wtheta_model.alpha_max))

            err_alpha = 9999

            # Plot the chi-squared vs alpha
            fig = plt.figure()
            plt.plot(alpha_vector, chi2_vector)
            plt.plot(alpha_best, chi2_best, 'd')
            plt.plot(alpha_vector, np.ones(n) * (chi2_best + 1), '--r')
            plt.xlabel(r'$\alpha$', fontsize=14)
            plt.ylabel(r'$\chi^2$', fontsize=14)
            plt.savefig(self.path_baofit + '/chi2_profile_bad.png', bbox_inches='tight')
            plt.close(fig)

        # Calculate degrees of freedom (dof)
        dof = len(self.wtheta_data_concatenated) - self.n_params_true
        for bin_z in self.bins_removed:
            dof -= len(self.wtheta_data[bin_z])

        # Save the results
        results = np.array([[alpha_best, err_alpha, chi2_best, dof]])
        np.savetxt(self.path_baofit + '/fit_results.txt', results, fmt=['%.4f', '%.4f', '%.4f', '%d'])

        # Print them to the console as well
        print(f'Best-fit alpha = {alpha_best:.4f} ± {err_alpha:.4f}')
        print(f'chi2/dof = {chi2_best:.4f}/{dof}')

        return alpha_best, err_alpha, chi2_best, dof
