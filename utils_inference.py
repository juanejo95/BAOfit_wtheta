import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from multiprocessing import Pool
import emcee
from getdist import MCSamples, plots
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

from utils_data import RedshiftDistributions
from utils_cosmology import CosmologicalParameters
from utils_baofit import BAOFitInitializer

class BAOFitChecker:
    def __init__(self, dataset, weight_type, mock_id, nz_flag, cov_type, cosmology_template,
             cosmology_covariance, delta_theta, theta_min, theta_max, pow_broadband, bins_removed, 
             alpha_min=0.8, alpha_max=1.2, alpha_type="alpha_wiggigg_only", verbose=True, code_path=None, save_path=None):
        """
        Initializes the BAOFitChecker class.
        Parameters:
        - dataset (str): Dataset identifier.
        - weight_type (int): Weight type (for dataset "DESY6" it should be either 1 or 0).
        - mock_id (int or str): Identifier for the mock. Use an integer for a specific mock, or the string "mean" to use the average.
        - nz_flag (str): Identifier for the n(z).
        - cov_type (str): Type of covariance.
        - cosmology_template (str): Cosmology for the template.
        - cosmology_covariance (str): Cosmology for the covariance.
        - delta_theta (float): Delta theta value.
        - theta_min (dict): Minimum theta value for each redshift bin.
        - theta_max (dict): Maximum theta value for each redshift bin.
        - pow_broadband (list): Powers of theta for the broadband parameters.
        - bins_removed (list): Redshift bins removed when running the BAO fit.
        - alpha_min (float): Minimum alpha allowed for the BAO fit.
        - alpha_max (float): Maximum alpha allowed for the BAO fit.
        - alpha_type (str): Either "alpha_wigg_only" (default: alpha only affects the wiggle part of the template) or "alpha_wigg_nowigg" (old choice: alpha enters in both the wiggle and the no-wiggle parts of the template).
        - verbose (bool): Whether to print messages.
        - code_path (str): Path to the code.
        - save_path (str): Path where outputs are saved.
        """
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
        self.pow_broadband = sorted(pow_broadband)
        self.bins_removed = sorted(bins_removed)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_type = alpha_type
        self.verbose = verbose

        if code_path is None:
            code_path = f"{os.environ['PSCRATCH']}/BAOfit_wtheta"
        self.code_path = code_path
        
        if save_path is None:
            save_path = f"{os.environ['PSCRATCH']}/BAOfit_wtheta"
        self.save_path = save_path

        # Store base configuration. This will be used to call BAOFitInitializer
        self.base_config = dict(
            dataset=self.dataset,
            weight_type=self.weight_type,
            mock_id=self.mock_id,
            nz_flag=self.nz_flag,
            cov_type=self.cov_type,
            cosmology_template=self.cosmology_template,
            cosmology_covariance=self.cosmology_covariance,
            delta_theta=self.delta_theta,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            pow_broadband=self.pow_broadband,
            bins_removed=self.bins_removed,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_type=self.alpha_type,
            save_path=self.save_path,
        )

        # Internal storage
        self.significance = None
        self.delta_chi2 = None
        self.is_detection = False
        self.fit_results = None
        self.all_params_bestfit = None
        self.path_baofit_wigg = None

        # Run detection
        self._run_detection()

    def _run_detection(self):
        """
        Compare wiggle and no-wiggle BAO fit results to determine whether the dataset contains a significant BAO detection.
        """
        config_wigg = {**self.base_config, "include_wiggles": ""}
        config_nowigg = {**self.base_config, "include_wiggles": "_nowiggles"}

        baofit_wigg = BAOFitInitializer(verbose=False, **config_wigg) # only needed to get the path to the results
        baofit_nowigg = BAOFitInitializer(verbose=False, **config_nowigg) # only needed to get the path to the results

        path_baofit_wigg = baofit_wigg.path_baofit
        path_baofit_nowigg = baofit_nowigg.path_baofit

        try:
            results_wigg = np.loadtxt(os.path.join(path_baofit_wigg, "fit_results.txt"))
            results_nowigg = np.loadtxt(os.path.join(path_baofit_nowigg, "fit_results.txt"))
        except OSError:
            if self.verbose:
                print(f"Missing BAO-fit results for dataset {self.dataset}. Please, run the BAO fit first!")
            return

        chi2_wigg = results_wigg[2]
        chi2_nowigg = results_nowigg[2]

        # Compute detection significance
        self.delta_chi2 = chi2_nowigg - chi2_wigg

        if np.isfinite(self.delta_chi2) and self.delta_chi2 > 0:
            self.significance = np.sqrt(self.delta_chi2)
        else:
            self.significance = 0.0

        # Check if this is a detection
        bestfit_file = os.path.join(path_baofit_wigg, "wtheta_data_bestfit.txt")

        if not os.path.exists(bestfit_file):
            if self.verbose:
                print(f"Dataset {self.dataset} has a non-detection.")
            return

        # Final detection condition
        if self.delta_chi2 > 0 and np.isfinite(self.significance):

            self.is_detection = True
            self.fit_results = results_wigg
            self.all_params_bestfit = np.load(
                os.path.join(path_baofit_wigg, "all_params_bestfit.npy"),
                allow_pickle=True,
            ).item()

            self.path_baofit_wigg = path_baofit_wigg # we will use it in BAOinference

            if self.verbose:
                print(f"Dataset {self.dataset} has a detection with Δχ² = {self.delta_chi2:.3f} (significance = {self.significance:.2f}σ)")

        else:

            if self.verbose:
                print(f"Dataset {self.dataset} has a detection but a Δχ² = {self.delta_chi2:.3f}")

    def get_detection_info(self):
        """
        Return a dictionary summarizing the BAO detection status and associated fit information.
        """
        return dict(
            dataset=self.dataset,
            is_detection=self.is_detection,
            significance=self.significance,
            delta_chi2=self.delta_chi2,
            path_baofit_wigg=self.path_baofit_wigg,
            fit_results=self.fit_results,
            all_params_bestfit=self.all_params_bestfit,
        )

class BAOInference:
    def __init__(self, baofit_checker, bounds=None, nwalkers=32, nsteps=5000, burnin=1000, overwrite=False, verbose=True):
        """
        Initializes the BAOInference class.
        Parameters:
        - baofit_checker: Instance of the BAOFitChecker class.
        - bounds (list of tuple): Parameter bounds for the minimizer and MCMC in the form [(hrd_min, hrd_max), (Omega_m_min, Omega_m_max)].
        - nwalkers (int): Number of walkers used in the MCMC sampler.
        - nsteps (str): Number of MCMC steps performed by each walker.
        - burnin (str): Number of initial MCMC steps to discard as burn-in before computing posterior samples.
        - overwrite (bool): Whether to overwrite existing results.
        - verbose (bool): Whether to print messages.
        """
        self.valid = baofit_checker.is_detection
        if not self.valid:
            if verbose:
                print(f"No BAO detection for dataset {baofit_checker.dataset}. Skipping inference...")
            return
        
        self.baofit_checker = baofit_checker
        self.cosmology_template = baofit_checker.cosmology_template
        self.path_baofit_wigg = baofit_checker.path_baofit_wigg
        self.bounds = bounds or [(60.0, 130.0), (0.1, 0.9)]
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = burnin
        self.overwrite = overwrite
        self.verbose = verbose

        if self.verbose:
            print(f"Saving output to: {self.path_baofit_wigg}")

        # Use the wiggle chi2 file
        self.chi2_files = [os.path.join(self.path_baofit_wigg, "likelihood_data.txt")]

        # Redshift distribution. Needed for the effective redshift
        self.redshift_distributions = RedshiftDistributions(self.baofit_checker.dataset, self.baofit_checker.nz_flag, verbose=False, code_path=self.baofit_checker.code_path)

        z_avg = []
        for bin_z in range(self.redshift_distributions.nbins):
            if bin_z not in self.baofit_checker.bins_removed:
                if self.redshift_distributions.nz_type == "widebin":
                    z_avg.append(self.redshift_distributions.z_average(bin_z))
                elif self.redshift_distributions.nz_type == "thinbin":
                    z_avg.append(self.redshift_distributions.nz_data[bin_z, 0])
        self.z_eff = np.mean(z_avg)
        print(f"Assuming an effective redshift of {self.z_eff}")

        # Setup fiducial cosmology
        self._setup_fiducial_cosmology(self.cosmology_template)

        # Load chi2 interpolators
        self._load_chi2_interpolators()

    def _setup_fiducial_cosmology(self, cosmology_template):
        """
        Compute key cosmological quantities for the fiducial cosmology.
        """
        cosmology_params = CosmologicalParameters(cosmology_template, verbose=False)
        cosmo = cosmology_params.get_cosmology()

        self.h_fid = cosmo.h
        self.rd_fid = cosmo.rs_drag / self.h_fid
        self.Omega_m_fid = cosmo.Omega_m(0).item()
        self.hrd_fid = self.h_fid * self.rd_fid
        self.DMrd_fid = self._DMrd_theory(self.hrd_fid, self.Omega_m_fid, self.z_eff)

        if self.verbose:
            print(f"Fiducial cosmology: {self.cosmology_template}")
            print(f"  h*r_d = {self.hrd_fid:.4f}, Omega_m = {self.Omega_m_fid:.4f}")

    def _E(self, Omega_m, z):
        """
        Return the dimensionless Hubble parameter E(z) for a flat ΛCDM cosmology.
        """
        return np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

    def _integral_I(self, Omega_m, z):
        """
        Compute the line-of-sight comoving distance integral ∫dz/E(z) needed for distance calculations.
        """
        return quad(lambda zp: 1.0 / self._E(Omega_m, zp), 0, z, epsrel=1e-8)[0]

    def _DMrd_theory(self, hrd, Omega_m, z):
        """
        Compute the theoretical value of D_M(z)/r_d for the given cosmological parameters.
        """
        I = self._integral_I(Omega_m, z)
        return (299792.458 / 100.0) * I / hrd

    def _load_chi2_interpolators(self):
        """
        Load χ² grids from file and construct interpolators used to evaluate the likelihood.
        """
        self.chi2_interps = []

        for file in self.chi2_files:
            data = np.loadtxt(file)
            alpha_grid = data[:, 0]
            chi2_grid = data[:, 1]
            chi2_grid -= chi2_grid.min()

            interp = interp1d(
                alpha_grid,
                chi2_grid,
                kind="linear",
                bounds_error=False,
                fill_value=np.inf,
            )
            self.chi2_interps.append(interp)

    def log_prior(self, theta):
        """
        Evaluate the prior probability for the parameter vector θ = (h*r_d, Ω_m).
        """
        hrd, Omega_m = theta
        if self.bounds[0][0] < hrd < self.bounds[0][1] and self.bounds[1][0] < Omega_m < self.bounds[1][1]:
            return 0.0
        return -np.inf

    def log_likelihood(self, theta):
        """
        Compute the log-likelihood by comparing the predicted BAO scaling α with the interpolated χ² grid.
        """
        hrd, Omega_m = theta
        if Omega_m <= 0 or Omega_m >= 1 or hrd <= 0:
            return -np.inf
    
        DMrd_th = self._DMrd_theory(hrd, Omega_m, self.z_eff)
        alpha_th = DMrd_th / self.DMrd_fid
        chi2_total = self.chi2_interps[0](alpha_th)
    
        if not np.isfinite(chi2_total):
            return -np.inf
        return -0.5 * chi2_total

    def log_probability(self, theta):
        """
        Return the total log-posterior probability as the sum of the log-prior and log-likelihood.
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_mcmc(self, initial):
        """
        Run the MCMC sampler (or load an existing chain) to generate posterior samples of the cosmological parameters.
        """
        filepath = os.path.join(self.path_baofit_wigg, "mcmc_chain.txt")
    
        # Load existing chain if present
        if os.path.exists(filepath) and not self.overwrite:
            data = np.loadtxt(filepath)
            samples = data[:, :2]
            log_probs = data[:, 2]
            if self.verbose:
                print(f"Loaded existing MCMC file...")
            return samples, log_probs
    
        # Otherwise run the MCMC
        ndim = 2
        pos = np.array(initial) + 1e-2 * np.random.randn(self.nwalkers, ndim)
    
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.log_probability)
        sampler.run_mcmc(pos, self.nsteps, progress=self.verbose)
    
        samples = sampler.get_chain(discard=self.burnin, flat=True)
        log_probs = sampler.get_log_prob(discard=self.burnin, flat=True)
    
        data = np.column_stack([samples, log_probs])
    
        np.savetxt(
            filepath,
            data,
            header="hrd Omega_m log_prob"
        )
    
        if self.verbose:
            print(f"Saved MCMC file!")
    
        return samples, log_probs

    def run_minimizer(self, samples, log_probs):
        """
        Refine the best-fit parameters by minimizing χ² starting from the highest-probability MCMC samples.
        """
        def chi2(theta):
            lp = self.log_probability(theta)
            if not np.isfinite(lp):
                return 1e30
            return -2.0 * lp
    
        top_indices = np.argsort(log_probs)[-1000:]
        top_samples = samples[top_indices]
    
        best_result = None
        best_chi2 = np.inf
    
        for theta0 in top_samples:
            res = minimize(chi2, theta0, method="L-BFGS-B", bounds=self.bounds)
            if res.success and res.fun < best_chi2:
                best_chi2 = res.fun
                best_result = res
    
        if self.verbose and best_result is not None:
            print(
                f"Minimizer result: "
                f"h*r_d = {best_result.x[0]:.4f}, "
                f"Omega_m = {best_result.x[1]:.4f}, "
                f"chi2_min = {best_result.fun:.4f}"
            )
    
        return best_result

    def summarize_chain(self, samples):
        """
        Compute median values and 68% credible intervals for the parameters from the MCMC samples.
        """
        hrd = np.percentile(samples[:, 0], [16, 50, 84])
        Omega_m = np.percentile(samples[:, 1], [16, 50, 84])

        results = {
            "hrd_median": hrd[1],
            "hrd_err_low": hrd[1] - hrd[0],
            "hrd_err_high": hrd[2] - hrd[1],
            "Omega_m_median": Omega_m[1],
            "Omega_m_err_low": Omega_m[1] - Omega_m[0],
            "Omega_m_err_high": Omega_m[2] - Omega_m[1],
        }

        if self.verbose:
            print(
                f"MCMC results: "
                f"h*r_d = {results['hrd_median']:.2f} +{results['hrd_err_high']:.2f}/-{results['hrd_err_low']:.2f}, "
                f"Omega_m = {results['Omega_m_median']:.4f} +{results['Omega_m_err_high']:.4f}/-{results['Omega_m_err_low']:.4f}"
            )

        return results

    def make_triangle_plot(self, samples, best_fit=None, close_fig=False):
        """
        Generate and save a corner (triangle) plot of the posterior parameter distributions.
        """
        names = ["hrd", "Omega_m"]
        labels = [r"h\,r_{\rm d}\ [{\rm Mpc}/h]", r"\Omega_{\rm m}"]

        ranges = {
            "hrd": list(self.bounds[0]),
            "Omega_m": list(self.bounds[1]),
        }

        gdsamples = MCSamples(
            samples=samples,
            names=names,
            labels=labels,
            ranges=ranges,
        )

        gdsamples.updateSettings(
            {
                "smooth_scale_2D": 0.3,
                "smooth_scale_1D": 0.3,
            }
        )

        g = plots.get_subplot_plotter()
        g.settings.axis_marker_ls = "--"
        g.settings.axis_marker_color = "black"

        fig = g.triangle_plot(
            gdsamples,
            filled=True,
            contour_colors=["darkblue"],
            markers={
                "hrd": self.hrd_fid,
                "Omega_m": self.Omega_m_fid,
            },
        )

        if best_fit is not None:
            ax = g.subplots[1, 0]
            ax.plot(best_fit.x[0], best_fit.x[1], marker='D', color='orange', markersize=4)

        plt.savefig(os.path.join(self.path_baofit_wigg, "mcmc_triangle.png"), bbox_inches="tight")
        if close_fig:
            plt.close(fig)
        