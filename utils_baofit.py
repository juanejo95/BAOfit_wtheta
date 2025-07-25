import numpy as np
import os
import hashlib
import json
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
            base_path = f"{os.environ['PSCRATCH']}/BAOfit_wtheta"
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
                    b**2 * interp["bb"](theta[bin_z]) +
                    b * interp["bf"](theta[bin_z]) +
                    interp["ff"](theta[bin_z])
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
            base_path = f"{os.environ['PSCRATCH']}/BAOfit_wtheta"
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

        letters = list("ABCDEFGH")  # Letters from A to H (depending on the number of broadband-term parameters)
        params = ["alpha"]
        for bin_z in range(self.nbins):
            params.extend([f"{letter}_{bin_z}" for letter in letters])
        self.names_params =  np.array(params)

        self.n_broadband_max = int((len(self.names_params) - 1) / self.nbins - 1)  # This should be 6 (from B to G, since A is the amplitude)
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
            params[(1 + self.n_broadband_max) * bin_z] * self.wtheta_th_interp[bin_z](alpha * theta[bin_z]) +  # A
            params[(1 + self.n_broadband_max) * bin_z + 1] +  # B
            params[(1 + self.n_broadband_max) * bin_z + 2] / theta[bin_z] +  # C
            params[(1 + self.n_broadband_max) * bin_z + 3] / theta[bin_z]**2 +  # D
            params[(1 + self.n_broadband_max) * bin_z + 4] * theta[bin_z] +  # E
            params[(1 + self.n_broadband_max) * bin_z + 5] * theta[bin_z]**2 +  # F
            params[(1 + self.n_broadband_max) * bin_z + 6] * theta[bin_z]**3 + # G
            params[(1 + self.n_broadband_max) * bin_z + 7] * theta[bin_z]**4  # H
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
        - theta_min (dict): Minimum theta value for each redshift bin.
        - theta_max (dict): Maximum theta value for each redshift bin.
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
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.Nalpha = 10**3

        if base_path is None:
            base_path = f"{os.environ['PSCRATCH']}/BAOfit_wtheta"
        self.base_path = base_path

        # Compute hash and config
        self.config_dict = self._build_config_dict()
        self.hash_path = self._compute_hash_path(self.config_dict)
        self.path_baofit = self._generate_path_baofit()

        # Create output directory
        os.makedirs(self.path_baofit, exist_ok=True)

        # Save or verify configuration
        self._check_or_save_config()

        if verbose:
            print(f"Saving output to: {self.path_baofit}")

    def _build_config_dict(self):
        """Build a dictionary of relevant configuration parameters for hashing and saving."""
        return {
            "nz_flag": self.nz_flag,
            "cov_type": self.cov_type,
            "cosmology_template": self.cosmology_template,
            "cosmology_covariance": self.cosmology_covariance,
            "delta_theta": self.delta_theta,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "n_broadband": self.n_broadband,
            "bins_removed": self.bins_removed,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
        }

    def _compute_hash_path(self, config):
        """Compute a unique hash from the configuration dictionary."""
        json_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:12]

    def _check_or_save_config(self):
        """Save config.json or compare with existing one to ensure it matches."""
        def normalize(obj):
            if isinstance(obj, dict):
                return {str(k): normalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [normalize(x) for x in obj]
            else:
                return obj
    
        config_path = os.path.join(self.path_baofit, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                existing_config = json.load(f)
            norm_existing = normalize(existing_config)
            norm_current = normalize(self.config_dict)
            if norm_existing != norm_current:
                raise ValueError(
                    f"Config mismatch in existing path: {config_path}\n"
                    f"Expected:\n{json.dumps(norm_existing, indent=2)}\n"
                    f"Found:\n{json.dumps(norm_current, indent=2)}"
                )
        else:
            with open(config_path, "w") as f:
                json.dump(self.config_dict, f, indent=2)

    def _generate_path_baofit(self):
        """Generate the save path for the BAO fit results."""
        if self.dataset in ["DESY6", "DESY6_dec_below-23.5", "DESY6_dec_above-23.5", "DESY6_DR1tiles_noDESI", "DESY6_DR1tiles_DESIonly"]:
            path = f"{self.base_path}/results/{self.dataset}/fit_results{self.include_wiggles}/weight_{self.weight_type}/{self.hash_path}"
        elif any(substr in self.dataset for substr in ["COLA", "EZ", "Abacus"]):
            path = f"{self.base_path}/results/{self.dataset}/fit_results{self.include_wiggles}/mock_{self.mock_id}/{self.hash_path}"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        return path

    def get_path_baofit(self):
        """Return the generated path for saving BAO fit results."""
        return self.path_baofit

class BAOFit:
    def __init__(self, baofit_initializer, wtheta_model, theta_data, wtheta_data, cov, close_fig=True, use_multiprocessing=False, n_cpu=None, overwrite=False):
        """
        Initialize the BAOFit class.

        Parameters:
        - baofit_initializer: Instance of the BAOFitInitializer class.
        - wtheta_model: Instance of the WThetaModel class.
        - theta_data (array): Theta data for fitting.
        - wtheta_data (dict): Data w(theta) for each bin.
        - cov (array): Covariance matrix.
        - close_fig (bool): Whether to close the resulting figures or not.
        - use_multiprocessing (bool): Whether to run the BAO fits using multiprocessing.
        - n_cpu (int): Number of CPUs for parallel processing (default: 20).
        - overwrite (bool): Whether to overwrite existing results or not.
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
        self.overwrite = overwrite

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
        wtheta_th = self.wtheta_template(self.theta_data, *params)
        diff = self.wtheta_data_concatenated - wtheta_th
        return diff @ self.inv_cov @ diff

    def broadband_params(self, amplitude_params, alpha):
        """Compute the broadband parameters."""
        fit_params = np.zeros(self.n_params)
        fit_params[0] = alpha
        fit_params[self.pos_amplitude] = amplitude_params
        return self.pseudo_inverse_matrix @ (self.wtheta_data_concatenated - self.wtheta_template(self.theta_data, *fit_params))

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

        if os.path.exists(os.path.join(self.path_baofit, "fit_results.txt")) and not self.overwrite:
            print("The output already exists! Loading the results and skipping BAO fit...")
            alpha_best, err_alpha, chi2_best, dof = np.loadtxt(os.path.join(self.path_baofit, "fit_results.txt")).T

        else:
            tol_minimize = 10**-7
            alpha_vector = np.linspace(self.alpha_min, self.alpha_max, self.Nalpha)
            chi2_vector = np.zeros_like(alpha_vector)
    
            def compute_chi2(alpha):
                amplitude_params = minimize(self.regularized_least_squares, x0=np.ones(self.nbins), method="SLSQP",
                                            bounds=[(0, None)] * self.nbins, tol=tol_minimize, args=(alpha,))
                chi2_value = self.regularized_least_squares(amplitude_params.x, alpha)
                return chi2_value
            
            if self.use_multiprocessing:
                with Pool(self.n_cpu) as pool:
                    chi2_vector = np.array(pool.map(compute_chi2, alpha_vector, chunksize=len(alpha_vector) // self.n_cpu))
            else:
                chi2_vector = np.array([compute_chi2(alpha) for alpha in alpha_vector])
                
            best = np.argmin(chi2_vector)
            alpha_best = alpha_vector[best]
            chi2_best = chi2_vector[best]
    
            np.savetxt(os.path.join(self.path_baofit, "likelihood_data.txt"), np.column_stack([alpha_vector, chi2_vector]))
    
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
    
                theta_data_interp = {}
                for bin_z in range(self.nbins):
                    theta_data_interp[bin_z] = np.linspace(self.theta_data[bin_z][0], self.theta_data[bin_z][-1], 10**3)
    
                wtheta_fit_best = self.wtheta_template(theta_data_interp, *params_best)
    
                # Plot the w(theta)
                nbins_eff = self.nbins - len(self.bins_removed)
                fig, axs = plt.subplots(nbins_eff, 1, figsize=(8, 2 * (nbins_eff)), sharex=True)
                axs = np.atleast_1d(axs)
                i = 0
                for bin_z in range(self.nbins):
                    if bin_z not in self.bins_removed:
                        ax = axs[i]
                        ax.errorbar(
                            self.theta_data[bin_z] * 180 / np.pi,
                            100 * (self.theta_data[bin_z] * 180 / np.pi) ** 2 * self.wtheta_data[bin_z],
                            yerr=100 * (self.theta_data[bin_z] * 180 / np.pi) ** 2 * np.sqrt(np.diag(self.cov))[sum(len(self.theta_data[bin_z2]) for bin_z2 in range(bin_z)):sum(len(self.theta_data[bin_z2]) for bin_z2 in range(bin_z + 1))],
                            capsize=4, capthick=1.5,
                            marker="D", markersize=6, markerfacecolor="lightblue", markeredgewidth=1.2,
                            markeredgecolor="dodgerblue", ecolor="dodgerblue", linestyle="none",
                            label=fr"\texttt{{{self.dataset}}}",
                            zorder=-1000
                        )
                        ax.plot(
                            theta_data_interp[bin_z] * 180 / np.pi, 
                            100 * (theta_data_interp[bin_z] * 180 / np.pi) ** 2 * self.wtheta_model.wtheta_th_interp[bin_z](theta_data_interp[bin_z]),
                            color="red", linestyle="--",
                            label="template"
                        )
                        ax.plot(
                            theta_data_interp[bin_z] * 180 / np.pi, 
                            100 * (theta_data_interp[bin_z] * 180 / np.pi) ** 2 * wtheta_fit_best[sum(len(theta_data_interp[bin_z2]) for bin_z2 in range(bin_z)):sum(len(theta_data_interp[bin_z2]) for bin_z2 in range(bin_z + 1))],
                            color="black",
                            label="best fit"
                        )
                        ax.set_ylabel(r"$10^2 \times \theta^2w(\theta)$", fontsize=22)
                        ax.tick_params(axis="x", labelsize=18)
                        ax.tick_params(axis="y", labelsize=18)
                        z_edge = self.z_edges[bin_z]
                        if self.dataset in ["DESIY1_LRG_EZ_ffa_deltaz0.028", "DESIY1_LRG_Abacus_altmtl_deltaz0.028", "DESIY1_LRG_EZ_complete_deltaz0.028", "DESIY1_LRG_Abacus_complete_deltaz0.028"]:
                            ax.text(0.13, 0.1, f"{z_edge[0]:.2f} $< z <$ {z_edge[1]:.2f}", ha="center", va="center", transform=ax.transAxes, fontsize=18)
                        else:
                            ax.text(0.13, 0.1, f"{z_edge[0]} $< z <$ {z_edge[1]}", ha="center", va="center", transform=ax.transAxes, fontsize=18)
                            
                        if i == 0:
                            ax.legend(loc="upper left", fontsize=18)
                        if i == nbins_eff - 1:
                            ax.set_xlabel(r"$\theta$ (deg)", fontsize=22)
                        i += 1
                if nbins_eff != 1:
                    fig.tight_layout()
                plt.savefig(os.path.join(self.path_baofit, "wtheta_data_bestfit.png"), bbox_inches="tight")
                if self.close_fig:
                    plt.close(fig)
                    
                # Save the w(theta)
                np.savetxt(
                    os.path.join(self.path_baofit, "wtheta_data_bestfit.txt"),
                    np.column_stack([
                        np.concatenate([self.theta_data[bin_z] for bin_z in range(self.nbins)]),
                        self.wtheta_data_concatenated,
                        self.wtheta_template(self.theta_data, *params_best),
                        np.sqrt(np.diag(self.cov))
                    ])
                )
    
                # Plot the chi2 vs alpha
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(alpha_vector, chi2_vector, color="dodgerblue", lw=2, label=r"$\chi^2$ profile")
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
                plt.savefig(os.path.join(self.path_baofit, "chi2_profile.png"), bbox_inches="tight", dpi=300)
                if self.close_fig:
                    plt.close(fig)
    
            else:
                print(f"The fit does not have the 1-sigma region between {self.alpha_min} and {self.alpha_max}!")
    
                err_alpha = 9999
                
                # Plot the chi2 vs alpha
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(alpha_vector, chi2_vector, color="dodgerblue", lw=2, label=r"$\chi^2$ profile")
                ax.axhline(chi2_best + 1, color="black", linestyle="--", linewidth=1, label=r"$\chi^2_{\mathrm{min}} + 1$")
                ax.plot(alpha_best, chi2_best, "d", color="orange", markersize=8, label=fr"$\alpha = {alpha_best:.4f}$")
                ax.set_xlim(self.alpha_min, self.alpha_max)
                ax.set_xlabel(r"$\alpha$", fontsize=16)
                ax.set_ylabel(r"$\chi^2$", fontsize=16)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                ax.tick_params(axis="both", which="major", labelsize=12)
                ax.legend(loc="lower right", fontsize=12)
                plt.savefig(os.path.join(self.path_baofit, "chi2_profile_bad.png"), bbox_inches="tight", dpi=300)
                if self.close_fig:
                    plt.close(fig)
    
            # Calculate degrees of freedom (dof)
            dof = len(self.wtheta_data_concatenated) - self.n_params_true
            for bin_z in self.bins_removed:
                dof -= len(self.wtheta_data[bin_z])
    
            # Save the results
            results = np.array([[alpha_best, err_alpha, chi2_best, dof]])
            np.savetxt(os.path.join(self.path_baofit, "fit_results.txt"), results, fmt=["%.4f", "%.4f", "%.4f", "%d"])

        # Print them to the console as well
        print(f"Best-fit alpha = {alpha_best:.4f} ± {err_alpha:.4f}")
        print(f"chi2/dof = {chi2_best:.1f}/{int(dof)}\n")

        return alpha_best, err_alpha, chi2_best, dof
