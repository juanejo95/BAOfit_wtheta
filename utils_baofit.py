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
from utils_template import TemplateInitializer

class WThetaModelGalaxyBias:
    def __init__(self, include_wiggles, dataset, nz_flag, cosmology_template, base_path=None):
        """
        Initialize the WThetaModelGalaxyBias class.

        Parameters:
        - include_wiggles (str): Whether to include BAO wiggles.
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - nz_flag (str): Identifier for the n(z).
        - cosmology_template (str): Cosmology for the template.
        - base_path (str): Path to save the results. Needed to load the template.
        """
        self.include_wiggles = include_wiggles
        self.dataset = dataset
        self.nz_flag = nz_flag
        self.cosmology_template = cosmology_template

        if base_path is None:
            base_path = f"{os.environ['HOME']}/BAOfit_wtheta"
        self.base_path = base_path

        # Initialize template data
        self.template_initializer = TemplateInitializer(
            include_wiggles=self.include_wiggles,
            dataset=self.dataset,
            nz_flag=self.nz_flag,
            cosmology_template=self.cosmology_template,
            verbose=False,
            base_path=self.base_path,
        )
        self.nbins = self.template_initializer.nbins
        self.z_edges = self.template_initializer.z_edges

        # Load and interpolate the theoretical wtheta
        self.wtheta_components_interp = self._load_and_interpolate_wtheta_components()

    def _load_and_interpolate_wtheta_components(self):
        """Load and interpolate the theoretical components of w(theta) (bb, bf, ff)."""
        components_interp = {}
        for bin_z in range(self.nbins):
            # Load the theoretical wtheta for each bin
            wtheta_dict = self.template_initializer.load_wtheta(bin_z)
            theta = wtheta_dict["bb"][:, 0]
            components_interp[bin_z] = {
                "theta": theta,
                "bb": interp1d(theta, wtheta_dict["bb"][:, 1], kind="cubic"),
                "bf": interp1d(theta, wtheta_dict["bf"][:, 1], kind="cubic"),
                "ff": interp1d(theta, wtheta_dict["ff"][:, 1], kind="cubic"),
            }
        return components_interp

    def get_wtheta_function(self):
        """Return a function that computes the theoretical w(theta) for each bin and concatenates them."""
        def wtheta(theta, *galaxy_bias):
            wtheta_concatenated = []
            for bin_z in range(self.nbins):
                interp = self.wtheta_components_interp[bin_z]
                b = galaxy_bias[bin_z]
                wtheta_bin = (
                    b**2 * interp["bb"](theta) +
                    b * interp["bf"](theta) +
                    interp["ff"](theta)
                )
                wtheta_concatenated.append(wtheta_bin)

            return np.concatenate(wtheta_concatenated)

        return wtheta

class WThetaModel:
    def __init__(self, include_wiggles, dataset, nz_flag, cosmology_template, n_broadband, galaxy_bias, base_path=None):
        """
        Initialize the WThetaModel class.

        Parameters:
        - include_wiggles (str): Whether to include BAO wiggles.
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - nz_flag (str): Identifier for the n(z).
        - cosmology_template (str): Cosmology for the template.
        - n_broadband (int): Number of broadband parameters.
        - galaxy_bias (dict): Dictionary containing the linear galaxy bias for each redshift bin.
        - base_path (str): Path to save the results. Needed to load the template.
        """
        self.include_wiggles = include_wiggles
        self.dataset = dataset
        self.nz_flag = nz_flag
        self.cosmology_template = cosmology_template
        self.n_broadband = n_broadband
        self.galaxy_bias = galaxy_bias
        
        if base_path is None:
            base_path = f"{os.environ['HOME']}/BAOfit_wtheta"
        self.base_path = base_path
        
        self.template_initializer = TemplateInitializer(
            include_wiggles=self.include_wiggles,
            dataset=self.dataset,
            nz_flag=self.nz_flag,
            cosmology_template=self.cosmology_template,
            verbose=False,
            base_path=self.base_path,
        )
        self.nbins = self.template_initializer.nbins
        self.z_edges = self.template_initializer.z_edges

        # Predefined values based on the given context
        self.names_params = np.array([
            "alpha", "A_0", "B_0", "C_0", "D_0", "E_0", "F_0", "G_0", 
            "A_1", "B_1", "C_1", "D_1", "E_1", "F_1", "G_1", 
            "A_2", "B_2", "C_2", "D_2", "E_2", "F_2", "G_2", 
            "A_3", "B_3", "C_3", "D_3", "E_3", "F_3", "G_3", 
            "A_4", "B_4", "C_4", "D_4", "E_4", "F_4", "G_4", 
            "A_5", "B_5", "C_5", "D_5", "E_5", "F_5", "G_5"
        ])
        self.n_broadband_max = int((len(self.names_params) - 1) / 6 - 1)  # This should be 6 (from B to G)
        self.n_params_max = len(self.names_params)

        # Prepare the index array
        self.indices_params = self._generate_indices()

        self.names_params = self.names_params[self.indices_params]

        # Interpolation of the theoretical wtheta
        self.wtheta_th_interp = self._load_and_interpolate_wtheta()

    def _generate_indices(self):
        """Generate index array based on the number of broadband bins."""
        indices_params_bb = np.arange(1, 1 + (self.n_broadband + 1))  # Only nuisance parameters
        indices_params = np.concatenate([[0], indices_params_bb])  # Including alpha
        for bin_z in range(1, self.nbins):
            indices_params = np.concatenate([indices_params, indices_params_bb + bin_z * (1 + self.n_broadband_max)])
        return indices_params

    def _load_and_interpolate_wtheta(self):
        """Load and interpolate the theoretical w(theta)."""
        wtheta_th_interp = {}
        for bin_z in range(self.nbins):
            # Load the theoretical wtheta for each bin
            wtheta_dict = self.template_initializer.load_wtheta(bin_z)
            theta = wtheta_dict["bb"][:, 0]
            wtheta_bb = wtheta_dict["bb"][:, 1]
            wtheta_bf = wtheta_dict["bf"][:, 1]
            wtheta_ff = wtheta_dict["ff"][:, 1]

            # Combine these into the final wtheta model for the bin
            wtheta_combined = (
                self.galaxy_bias[bin_z]**2 * wtheta_bb + 
                self.galaxy_bias[bin_z] * wtheta_bf + 
                wtheta_ff
            )
            
            # Interpolate the combined wtheta
            wtheta_th_interp[bin_z] = interp1d(theta, wtheta_combined, kind="cubic")

        return wtheta_th_interp

    def wtheta_template_raw(self, theta, alpha, *params):
        """Theoretical template calculation."""
        wtheta_template = np.concatenate([
            params[(1 + self.n_broadband_max) * bin_z] * self.wtheta_th_interp[bin_z](alpha * theta) +  # A
            params[(1 + self.n_broadband_max) * bin_z + 1] +  # B
            params[(1 + self.n_broadband_max) * bin_z + 2] / theta +  # C
            params[(1 + self.n_broadband_max) * bin_z + 3] / theta**2 +  # D
            params[(1 + self.n_broadband_max) * bin_z + 4] * theta +  # E
            params[(1 + self.n_broadband_max) * bin_z + 5] * theta**2 +  # F
            params[(1 + self.n_broadband_max) * bin_z + 6] * theta**3  # G
            for bin_z in range(self.nbins)
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
    def __init__(self, include_wiggles, dataset, weight_type, mock_id, nz_flag, cov_type, cosmology_template,
                 cosmology_covariance, delta_theta, theta_min, theta_max, n_broadband, bins_removed, 
                 alpha_min=0.8, alpha_max=1.2, verbose=True, base_path=None):
        """
        Initializes the BAOFitInitializer class.

        Parameters:
        - include_wiggles (str): Whether to include BAO wiggles.
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - weight_type (int): Weight type (for dataset "DESY6" it should be either 1 or 0).
        - mock_id (int): Mock id (for dataset "DESY6_COLA" it should go from 0 to 1951).
        - nz_flag (str): Identifier for the n(z).
        - cov_type (str): Type of covariance.
        - cosmology_template (str): Cosmology for the template.
        - cosmology_covariance (str): Cosmology for the covariance.
        - delta_theta (float): Delta theta value.
        - theta_min (float): Minimum theta value.
        - theta_max (float): Maximum theta value.
        - n_broadband (int): Number of broadband parameters.
        - bins_removed (list): Redshift bins removed when running the BAO fit.
        - verbose (bool): Whether to print messages.
        - base_path (str): Path to save the results.
        """
        self.include_wiggles = include_wiggles
        self.dataset = dataset
        self.weight_type = weight_type
        self.mock_id = mock_id
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
        
        if base_path is None:
            base_path = f"{os.environ['HOME']}/BAOfit_wtheta"
        self.base_path = base_path
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.Nalpha = 10**4
        
        # Path to save the BAO-fit results
        self.path_baofit = self._generate_path_baofit()

        # Create the directory if it does not exist
        os.makedirs(self.path_baofit, exist_ok=True)
        if verbose:
            print(f"Saving output to: {self.path_baofit}")

    def _generate_path_baofit(self):
        """Generate the save path for the BAO fit results."""
        if self.dataset in ["DESY6", "DESY6_noDESI_-23.5", "DESY6_DESI_-23.5"]:
            path = (
                f"{self.base_path}/fit_results{self.include_wiggles}/{self.dataset}/weight_{self.weight_type}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}_"
                f"alphamin{self.alpha_min}_alphamax{self.alpha_max}"
            )
        elif self.dataset == "DESY6_COLA":
            path = (
                f"{self.base_path}/fit_results{self.include_wiggles}/{self.dataset}/mock_{self.mock_id}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}_"
                f"alphamin{self.alpha_min}_alphamax{self.alpha_max}"
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        return path

    def get_path_baofit(self):
        """Return the generated path for saving BAO fit results."""
        return self.path_baofit

class BAOFit:
    def __init__(self, baofit_initializer, wtheta_model, theta_data, wtheta_data, cov, close_fig=True, use_multiprocessing=False, n_cpu=None):
        """
        Initialize the BAOFit class.

        Parameters:
        - baofit_initializer: Instance of the BAOFitInitializer class.
        - wtheta_model: Instance of the WThetaModel class.
        - theta_data (array): Theta data for fitting.
        - wtheta_data (dict): Data w(theta) for each bin.
        - cov (array): Covariance matrix.
        - close_fig (bool): Whether to close the resulting figures or nnot.
        - use_multiprocessing (bool): Whether to run the BAO fits using multiprocessing.
        - n_cpu (int): Number of CPUs for parallel processing (default: 20).
        """
        self.wtheta_model = wtheta_model
        self.wtheta_template = wtheta_model.get_wtheta_template()
        self.z_edges = wtheta_model.z_edges
        self.theta_data = theta_data
        self.wtheta_data = wtheta_data
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)  # Compute the inverse covariance matrix
        self.n_broadband = wtheta_model.n_broadband
        self.nbins = wtheta_model.nbins
        self.close_fig = close_fig
        self.use_multiprocessing = use_multiprocessing
        self.n_cpu = n_cpu if n_cpu is not None else 20
        
        if self.use_multiprocessing:
            print(f"WARNING: The BAO fit will be run in parallel using {self.n_cpu} CPUs!")
        
        self.dataset = baofit_initializer.dataset
        self.path_baofit = baofit_initializer.get_path_baofit()
        self.bins_removed = baofit_initializer.bins_removed
        self.alpha_min = baofit_initializer.alpha_min
        self.alpha_max = baofit_initializer.alpha_max
        self.Nalpha = baofit_initializer.Nalpha

        # Concatenate wtheta data
        self.wtheta_data_concatenated = np.concatenate([self.wtheta_data[bin_z] for bin_z in range(self.nbins)])

        # Number of parameters
        self.n_params = len(self.wtheta_model.names_params)
        self.n_params_true = len(self.wtheta_model.names_params) - (1 + self.n_broadband) * len(self.bins_removed)

        # Precompute positions of amplitude and broadband parameters
        self.pos_amplitude = np.array([1 + bin_z * (self.n_broadband + 1) for bin_z in range(self.nbins)])
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
        """Construct the design matrix."""
        for i in np.arange(0, self.nbins * self.n_broadband):
            fit_params = np.zeros(self.n_params)
            fit_params[0] = 1
            fit_params[self.pos_broadband[i]] = 1
            self.design_matrix[:, i] = self.wtheta_template(self.theta_data, *fit_params)

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
        alpha_vector = np.linspace(self.alpha_min, self.alpha_max, self.Nalpha)
        chi2_vector = np.zeros_like(alpha_vector)

        def compute_chi2(alpha):
            amplitude_params = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method="SLSQP",
                                        bounds=[(0, None)] * self.nbins, tol=tol_minimize, args=(alpha,))
            chi2_value = self.regularized_least_squares(amplitude_params.x, alpha)
            return chi2_value
        
        if self.use_multiprocessing:
            with Pool(self.n_cpu) as pool: # using multiprocessing with pathos...
                chi2_vector = np.array(pool.map(compute_chi2, alpha_vector))
        else:
            chi2_vector = np.array([compute_chi2(alpha) for alpha in alpha_vector])
            
        best = np.argmin(chi2_vector)
        alpha_best = alpha_vector[best]
        chi2_best = chi2_vector[best]

        np.savetxt(self.path_baofit + "/likelihood_data.txt", np.column_stack([alpha_vector, chi2_vector]))

        if chi2_vector[0] > chi2_best + 1 and chi2_vector[-1] > chi2_best + 1:
            alpha_down = None
            alpha_up = None

            for i in np.arange(0, best)[::-1]:
                if chi2_vector[i] < chi2_best + 1 and chi2_vector[i - 1] > chi2_best + 1:
                    alpha_down = alpha_vector[i]
                    break

            for i in np.arange(best, self.Nalpha):
                if chi2_vector[i] < chi2_best + 1 and chi2_vector[i + 1] > chi2_best + 1:
                    alpha_up = alpha_vector[i]
                    break

            err_alpha = (alpha_up - alpha_down) / 2

            amplitude_params_best = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method="SLSQP",
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
                    fmt=".", capsize=3, 
                    label=self.dataset + " data"
                )
                ax.plot(
                    theta_data_interp * 180 / np.pi, 
                    100 * (theta_data_interp * 180 / np.pi) ** 2 * self.wtheta_model.wtheta_th_interp[bin_z](theta_data_interp),
                    label="template"
                )
                ax.plot(
                    theta_data_interp * 180 / np.pi, 
                    100 * (theta_data_interp * 180 / np.pi) ** 2 * wtheta_fit_best[bin_z * len(theta_data_interp):(bin_z + 1) * len(theta_data_interp)],
                    label="best fit"
                )
                ax.set_ylabel(r"$10^2 \times \theta^2w(\theta)$", fontsize=13)
                z_edge = self.z_edges[bin_z]
                ax.text(0.13, 0.1, f"{z_edge[0]} $< z <$ {z_edge[1]}", ha="center", va="center", transform=ax.transAxes, fontsize=18)
                if bin_z == 0:
                    ax.legend(loc="upper left", fontsize=13)
                if bin_z == self.nbins - 1:
                    ax.set_xlabel(r"$\theta$ (deg)", fontsize=13)
            plt.tight_layout()
            plt.savefig(self.path_baofit + "/wtheta_data_bestfit.png", bbox_inches="tight")
            if self.close_fig:
                plt.close(fig)
                
            # Save the w(theta)
            np.savetxt(
                self.path_baofit + "/wtheta_data_bestfit.txt",
                np.column_stack([
                    np.concatenate([self.theta_data] * self.nbins),
                    self.wtheta_data_concatenated,
                    self.wtheta_template(self.theta_data, *params_best),
                    np.sqrt(np.diag(self.cov))
                ])
            )

            # Plot the chi2 vs alpha
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(alpha_vector, chi2_vector, color="blue", lw=2, label=r"$\chi^2$ profile")
            ax.axhline(chi2_best + 1, color="black", linestyle="--", linewidth=1, label=r"$\chi^2_{\mathrm{min}} + 1$")
            ax.plot(alpha_best, chi2_best, "d", color="orange", markersize=8, label=fr"$\alpha = {alpha_best:.4f}$")
            ax.axvspan(alpha_down, alpha_up, color="k", alpha=0.1, label=fr"$\sigma_\alpha = {err_alpha:.4f}$")
            ax.set_xlim(self.alpha_min, self.alpha_max)
            ax.set_xlabel(r"$\alpha$", fontsize=16)
            ax.set_ylabel(r"$\chi^2$", fontsize=16)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.legend(loc="lower right", fontsize=12)
            plt.savefig(self.path_baofit + "/chi2_profile.png", bbox_inches="tight", dpi=300)
            if self.close_fig:
                plt.close(fig)

        else:
            print("The fit does not have the 1-sigma region between " + str(self.alpha_min) + " and " + str(self.alpha_max) +".")

            err_alpha = 9999
            
            # Plot the chi2 vs alpha
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(alpha_vector, chi2_vector, color="blue", lw=2, label=r"$\chi^2$ profile")
            ax.axhline(chi2_best + 1, color="black", linestyle="--", linewidth=1, label=r"$\chi^2_{\mathrm{min}} + 1$")
            ax.plot(alpha_best, chi2_best, "d", color="orange", markersize=8, label=fr"$\alpha = {alpha_best:.4f}$")
            ax.set_xlim(self.alpha_min, self.alpha_max)
            ax.set_xlabel(r"$\alpha$", fontsize=16)
            ax.set_ylabel(r"$\chi^2$", fontsize=16)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.legend(loc="lower right", fontsize=12)
            plt.savefig(self.path_baofit + "/chi2_profile_bad.png", bbox_inches="tight", dpi=300)
            if self.close_fig:
                plt.close(fig)

        # Calculate degrees of freedom (dof)
        dof = len(self.wtheta_data_concatenated) - self.n_params_true
        for bin_z in self.bins_removed:
            dof -= len(self.wtheta_data[bin_z])

        # Save the results
        results = np.array([[alpha_best, err_alpha, chi2_best, dof]])
        np.savetxt(self.path_baofit + "/fit_results.txt", results, fmt=["%.4f", "%.4f", "%.4f", "%d"])

        # Print them to the console as well
        print(f"Best-fit alpha = {alpha_best:.4f} ± {err_alpha:.4f}")
        print(f"chi2/dof = {chi2_best:.4f}/{dof}")

        return alpha_best, err_alpha, chi2_best, dof
