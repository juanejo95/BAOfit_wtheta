import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

class wtheta_model:
    def __init__(self, alpha_min, alpha_max, galaxy_bias, n_broadband, bins_removed, path_template):
        """
        Initialize the wtheta_model class with necessary parameters and settings.

        Args:
            alpha_min (float): Minimum allowed alpha value.
            alpha_max (float): Maximum allowed alpha value.
            galaxy_bias (dict): Linear galaxy bias for each redshift bin.
            n_broadband (int): The number of broadband bins.
            bins_removed (list): List of bins removed for the fitting.
            path_template (str): Path where theoretical wtheta files are located.
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.galaxy_bias = galaxy_bias
        self.n_broadband = n_broadband
        self.bins_removed = bins_removed
        self.path_template = path_template

        # Set nbins based on the number of bins (e.g., from the template data)
        self.nbins = len(galaxy_bias)  # You may want to adjust this based on your specific needs

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

class bao_fit:
    def __init__(self, wtheta_model_instance, theta_data, wtheta_data, cov, path_baofit):
        """
        Initialize the BAO fit class.
        """
        self.wtheta_model_instance = wtheta_model_instance
        self.wtheta_template = wtheta_model_instance.get_wtheta_template()
        self.theta_data = theta_data
        self.wtheta_data = wtheta_data
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)  # Compute the inverse covariance matrix
        self.path_baofit = path_baofit
        self.n_broadband = wtheta_model_instance.n_broadband
        self.nbins = wtheta_model_instance.nbins
        
        # Define the concatenated wtheta data
        self.wtheta_data_concatenated = np.concatenate([self.wtheta_data[bin_z] for bin_z in range(self.nbins)])
        self.bins_removed = [bin_z for bin_z in range(self.nbins) if np.all(self.wtheta_data[bin_z] == 0)]
        
        # Number of parameters
        self.n_params = len(wtheta_model_instance.names_params)
        self.n_params_true = len(wtheta_model_instance.names_params) - (1 + self.n_broadband) * len(self.bins_removed)

        # Precompute positions of amplitude and broadband parameters
        self.pos_amplitude = np.array([1 + i * (self.n_broadband + 1) for i in np.arange(0, self.nbins)])
        self.pos_broadband = np.delete(np.arange(0, self.n_params), np.concatenate(([0], self.pos_amplitude)))
        
        # Construct the design matrix
        self.design_matrix = np.zeros([len(self.wtheta_data_concatenated), self.nbins * self.n_broadband])
        self._construct_design_matrix()

        # Compute the pseudo-inverse of the design matrix
        self.pseudo_inverse_matrix = np.linalg.inv((self.design_matrix.T) @ self.inv_cov @ self.design_matrix) @ (self.design_matrix.T) @ self.inv_cov

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
        np.savetxt(self.path_baofit + '/likelihood_data.txt', np.column_stack([alpha_vector, chi2_vector]))
        
        # Check the chi2 values at the extremes of alpha range
        if chi2_vector[0] > chi2_best + 1 and chi2_vector[-1] > chi2_best + 1:
            # Search for alpha_down and alpha_up where chi2 crosses chi2_best + 1
            for i in np.arange(0, best)[::-1]:
                if chi2_vector[i] < chi2_best + 1 and chi2_vector[i - 1] > chi2_best + 1:
                    alpha_down = alpha_vector[i]
                    break

            for i in np.arange(best, n):
                if chi2_vector[i] < chi2_best + 1 and chi2_vector[i + 1] > chi2_best + 1:
                    alpha_up = alpha_vector[i]
                    break

            # Refine the region around alpha_best
            alpha_best_vector = np.linspace(alpha_best - 0.01, alpha_best + 0.01, n)
            chi2_best_vector = np.zeros(n)
            alpha_down_vector = np.linspace(alpha_down - 0.01, alpha_down + 0.01, n)
            chi2_down_vector = np.zeros(n)
            alpha_up_vector = np.linspace(alpha_up - 0.01, alpha_up + 0.01, n)
            chi2_up_vector = np.zeros(n)

            for i in np.arange(0, n):
                amplitude_params_best = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                                 bounds=([(0, None)] * self.nbins), tol=tol_minimize, args=(alpha_best_vector[i]))
                chi2_best_vector[i] = self.regularized_least_squares(amplitude_params_best.x, alpha_best_vector[i])

                amplitude_params_down = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                                 bounds=([(0, None)] * self.nbins), tol=tol_minimize, args=(alpha_down_vector[i]))
                chi2_down_vector[i] = self.regularized_least_squares(amplitude_params_down.x, alpha_down_vector[i])

                amplitude_params_up = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                               bounds=([(0, None)] * self.nbins), tol=tol_minimize, args=(alpha_up_vector[i]))
                chi2_up_vector[i] = self.regularized_least_squares(amplitude_params_up.x, alpha_up_vector[i])

            # Find the new best alpha within the refined region
            best_new = np.argmin(chi2_best_vector)
            alpha_best_new = alpha_best_vector[best_new]
            chi2_best_new = chi2_best_vector[best_new]

            # Check for the refined error region bounds
            alpha_down_new = alpha_down_vector[np.argmin(abs(chi2_down_vector - (chi2_best + 1)))]
            alpha_up_new = alpha_up_vector[np.argmin(abs(chi2_up_vector - (chi2_best + 1)))]
            
            # Update alpha_best, alpha_down, and alpha_up
            if alpha_best_vector[0] < alpha_best_new < alpha_best_vector[-1] and \
               alpha_down_vector[0] < alpha_down_new < alpha_down_vector[-1] and \
               alpha_up_vector[0] < alpha_up_new < alpha_up_vector[-1]:
                alpha_best, alpha_down, alpha_up = alpha_best_new, alpha_down_new, alpha_up_new
                chi2_best = chi2_best_new
            else:
                print('There is a problem with the fit!')
                
            # Compute the error region (1-sigma)
            err_alpha = (alpha_up - alpha_down) / 2
            
            # Final amplitude and broadband parameters for the best-fit alpha
            amplitude_params_best = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method='SLSQP',
                                        bounds=([(0, None)] * self.nbins), tol=tol_minimize, args=(alpha_best)).x
            
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
            print('The fit does not have the 1-sigma region between ' + str(self.wtheta_model_instance.alpha_min) + ' and ' + str(self.wtheta_model_instance.alpha_max))
            
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