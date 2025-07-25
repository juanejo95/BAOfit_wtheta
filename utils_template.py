import numpy as np
import os
import scipy
import multiprocessing
from functools import partial
from cosmoprimo import PowerSpectrumBAOFilter
from pathos.multiprocessing import ProcessingPool as Pool
from utils_cosmology import CosmologicalParameters
from utils_data import RedshiftDistributions

class TemplateInitializer:
    def __init__(self, include_wiggles, dataset, nz_flag, cosmology_template, Nk=2*10**5, Nmu=5*10**4, Nr=5*10**4, Nz=10**3, Ntheta=10**3, 
                 use_multiprocessing=False, n_cpu=None, verbose=True, base_path=None):
        """
        Initializes the template calculator based on input parameters.

        Parameters:
        - include_wiggles (str): Whether to include BAO wiggles.
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - nz_flag (str): Identifier for the n(z).
        - cosmology_template (str): Cosmology for the template.
        - Nk (int): Number of k bins for the P(k, mu).
        - Nmu (int): Number of mu bins for the P(k, mu).
        - Nr (int): Number of r bins when computing xi(r)
        - Nz (int): Number of z bins when projecting in redshift.
        - Ntheta (int): Number of theta values when computing the w(theta).
        - use_multiprocessing (bool): Whether to run the BAO fits using multiprocessing or not.
        - n_cpu (int): Number of CPUs for parallel processing (default: 20).
        - verbose (bool): Whether to print messages.
        - base_path (str): Path to save the results.
        """
        self.include_wiggles = include_wiggles
        self.dataset = dataset
        self.nz_flag = nz_flag
        self.cosmology_template = cosmology_template
        self.Nk = Nk
        self.Nmu = Nmu
        self.Nr = Nr
        self.Nz = Nz
        self.Ntheta = Ntheta
        self.use_multiprocessing = use_multiprocessing
        self.n_cpu = n_cpu if n_cpu is not None else 20
        self.verbose = verbose
        
        if base_path is None:
            base_path = f"{os.environ['PSCRATCH']}/BAOfit_wtheta"
        self.base_path = base_path
        
        self.mu_vector = np.linspace(-1, 1, self.Nmu)
        self.r_12_vector = 10**np.linspace(np.log10(10**-2), np.log10(10**5), self.Nr)
        self.theta = 10**np.linspace(np.log10(0.001), np.log10(179.5), self.Ntheta) * np.pi / 180
        
        # Legendre polynomials
        self.ells = [0, 2, 4]
        self.legendre = {ell: scipy.special.eval_legendre(ell, self.mu_vector) for ell in self.ells}

        # Bias components
        self.components = ["bb", "bf", "ff"]
        
        # Path to save the templates
        self.path_template = f"{self.base_path}/templates/{self.dataset}/wtheta_template{self.include_wiggles}/nz_{self.nz_flag}/wtheta_{self.cosmology_template}"
        
        # Make sure the directory exists
        os.makedirs(self.path_template, exist_ok=True)

        if self.verbose:
            print(f"Saving output to: {self.path_template}")
            
        # Redshift distribution
        self.redshift_distributions = RedshiftDistributions(self.dataset, self.nz_flag, verbose=False)
        self.nbins = self.redshift_distributions.nbins
        self.z_edges = self.redshift_distributions.z_edges
        
        # Initialize cosmology
        self._initialize_cosmology()

        # Create the k and kh vectors
        if "old" in self.cosmology_template:
            if self.cosmology_template == "mice_old":
                pk_old = np.loadtxt("pk_old/PkMICE_linear_nowiggle_GaussianSmoothing0p3_z0.dat")
            elif self.cosmology_template == "planck_old":
                pk_old = np.loadtxt("pk_old/Pk_Challenage_Lin_Smooth.dat")
            
            self.k_old = pk_old[:, 0] * self.cosmo.h
            self.pk_wigg_old = pk_old[:, 1] / self.cosmo.h**3 * (2 * np.pi)**3
            self.pk_nowigg_old = pk_old[:, 2] / self.cosmo.h**3 * (2 * np.pi)**3
            
            if self.cosmology_template == 'mice_old':
                kh = 10**np.linspace(np.log10(self.k_old.min() / self.cosmo.h), np.log10(10**2), self.Nk)
            elif self.cosmology_template == 'planck_old':
                kh = 10**np.linspace(np.log10(self.k_old.min() / self.cosmo.h), np.log10(self.k_old.max() / self.cosmo.h), self.Nk)
            self.kh = kh
            self.k = self.kh * self.cosmo.h

        else:
            self.kh = 10**np.linspace(np.log10(10**-4 / self.cosmo.h), np.log10(10**2), self.Nk)
            self.k = self.kh * self.cosmo.h

    def get_path_template(self):
        """Return the generated path_template."""
        return self.path_template
    
    def _initialize_cosmology(self):
        """Initialize cosmology based on the template."""
        cosmology_params = CosmologicalParameters(self.cosmology_template, verbose=self.verbose)
        self.cosmo = cosmology_params.get_cosmology()
            
    def load_pk_ell(self, bin_z):
        """
        Load precomputed Pk_ell for a given redshift bin.

        Parameters:
        - bin_z (int): Redshift bin number.
        """
        if self.verbose:
            print(f"{bin_z} - Attempting to load precomputed Pk_ell...")

        pk_ell_dict = {}
        try:
            pk_ell_dict = {
                f"{ell}_{component}": np.loadtxt(f"{self.path_template}/Pk_ell_{component}_bin{bin_z}.txt")[:, idx + 1]
                for component in self.components
                for idx, ell in enumerate(self.ells)
            }
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Precomputed Pk_ell files do not exist. Please, compute them first.")
            raise

        return pk_ell_dict
            
    def load_xi_ell(self, bin_z):
        """
        Load precomputed xi_ell for a given redshift bin.

        Parameters:
        - bin_z (int): Redshift bin number.
        """
        if self.verbose:
            print(f"{bin_z} - Attempting to load precomputed xi_ell...")

        xi_ell_dict = {}
        try:
            xi_ell_dict = {
                f"{ell}_{component}": np.loadtxt(f"{self.path_template}/xi_ell_{component}_bin{bin_z}.txt")[:, idx + 1]
                for component in self.components
                for idx, ell in enumerate(self.ells)
            }
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Precomputed xi_ell files do not exist. Please, compute them first.")
            raise

        return xi_ell_dict
    
    def load_wtheta(self, bin_z):
        """
        Load precomputed w(theta) for a given redshift bin.

        Parameters:
        - bin_z (int): Redshift bin number.
        """
        if self.verbose:
            print(f"{bin_z} - Attempting to load precomputed w(theta)...")

        wtheta_dict = {}
        try:
            wtheta_dict = {
                component: np.loadtxt(f"{self.path_template}/wtheta_{component}_bin{bin_z}.txt")
                for component in self.components
            }
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Precomputed w(theta) files do not exist. Please, compute them first.")
            raise

        return wtheta_dict

    def load_C_ell(self, bin_z):
        """
        Load precomputed C_ell for a given redshift bin.

        Parameters:
        - bin_z (int): Redshift bin number.
        """
        if self.verbose:
            print(f"{bin_z} - Attempting to load precomputed C_ell...")

        C_ell_dict = {}
        try:
            C_ell_dict = {
                component: np.loadtxt(f"{self.path_template}/C_ell_{component}_bin{bin_z}.txt")
                for component in self.components
            }
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Precomputed C_ell files do not exist. Please, compute them first.")
            raise

        return C_ell_dict

class PowerSpectrumMultipoles:
    def __init__(self, template_initializer):
        """
        Initialize the PowerSpectrumMultipoles class.

        Parameters:
        - template_initializer: Instance of the TemplateInitializer class.
        """
        self.template_initializer = template_initializer
        self.include_wiggles = self.template_initializer.include_wiggles
        self.use_multiprocessing = self.template_initializer.use_multiprocessing
        self.n_cpu = self.template_initializer.n_cpu
        self.verbose = self.template_initializer.verbose
        self.cosmo = self.template_initializer.cosmo
        self.redshift_distributions = self.template_initializer.redshift_distributions
        self.nbins = self.template_initializer.nbins
        self.path_template = self.template_initializer.get_path_template()
        self.k = self.template_initializer.k
        self.kh = self.template_initializer.kh
        self.Nk = self.template_initializer.Nk
        self.mu_vector = self.template_initializer.mu_vector
        self.ells = self.template_initializer.ells
        self.components = self.template_initializer.components
        self.legendre = self.template_initializer.legendre
        self.cosmology_template = self.template_initializer.cosmology_template
        
        # Initialize power spectrum
        self._initialize_power_spectrum()

    def _initialize_power_spectrum(self):
        """Initialize k and P(k) values."""

        if "old" in self.cosmology_template:
            # Power spectrum with wiggles
            self.Pk_wigg = np.interp(self.k, self.template_initializer.k_old, self.template_initializer.pk_wigg_old)

            # Power spectrum without wiggles
            self.Pk_nowigg = np.interp(self.k, self.template_initializer.k_old, self.template_initializer.pk_nowigg_old)
            
        else:
            # Power spectrum with wiggles
            pkz = self.cosmo.get_fourier(engine="class").pk_interpolator()
            pk = pkz.to_1d(z=0)
            self.Pk_wigg = pk(self.kh) / self.cosmo.h**3
    
            # Power spectrum without wiggles
            pknow = PowerSpectrumBAOFilter(pk, engine="peakaverage", cosmo_fid=self.cosmo).smooth_pk_interpolator()
            self.Pk_nowigg = pknow(self.kh) / self.cosmo.h**3

        np.savetxt(f"{self.path_template}/Pk_full.txt", np.column_stack([self.k, self.Pk_wigg, self.Pk_nowigg]))

        self.Sigma_0, self.delta_Sigma_0 = self.compute_sigma_parameters()

    def compute_sigma_parameters(self):
        """Compute Sigma_0 and delta_Sigma_0."""
        k_s = 0.2 * self.cosmo.h
        ell_BAO = 110 / self.cosmo.h

        q = 10**np.linspace(np.log10(self.k.min()), np.log10(k_s), self.Nk)
        Pq_nowigg = np.interp(q, self.k, self.Pk_nowigg)

        x = q * ell_BAO
        j_0 = np.sin(x) / x
        j_2 = (3 / x**2 - 1) * j_0 - 3 * np.cos(x) / x**2

        Sigma_0 = np.sqrt(np.trapz(Pq_nowigg * (1 - j_0 + 2 * j_2), q) / (6 * np.pi**2))
        delta_Sigma_0 = np.sqrt(np.trapz(Pq_nowigg * j_2, q) / (2 * np.pi**2))

        return Sigma_0, delta_Sigma_0

    def compute_sigma_tot_vector(self, z, f):
        """Compute the total sigma vector."""
        D = self.cosmo.growth_factor(z)
        Sigma = D * self.Sigma_0
        delta_Sigma = D * self.delta_Sigma_0

        Sigma_paral = (1 + f) * Sigma
        Sigma_perp = Sigma
        return np.sqrt(
            self.mu_vector**2 * Sigma_paral**2
            + (1 - self.mu_vector**2) * Sigma_perp**2
            + f**2 * self.mu_vector**2 * (self.mu_vector**2 - 1) * delta_Sigma**2
        )

    def compute_pk_ell_singlek(self, bin_z, f, Sigma_tot_vector, i):
        """Compute the power spectrum multipoles for a single value of k."""
        if self.include_wiggles == "":
            pk_term = (self.Pk_wigg[i] - self.Pk_nowigg[i]) * np.exp(-self.k[i]**2 * Sigma_tot_vector**2) + self.Pk_nowigg[i]
        elif self.include_wiggles == "_nowiggles":
            pk_term = self.Pk_nowigg[i]

        pk_dict = {}
        for ell in self.ells:
            pk_dict[f"Pk_{ell}_bb"] = (2 * ell + 1) / 2 * np.trapz(pk_term * self.legendre[ell], self.mu_vector)
            pk_dict[f"Pk_{ell}_bf"] = (2 * ell + 1) / 2 * np.trapz(2 * self.mu_vector**2 * f * pk_term * self.legendre[ell], self.mu_vector)
            pk_dict[f"Pk_{ell}_ff"] = (2 * ell + 1) / 2 * np.trapz(self.mu_vector**4 * f**2 * pk_term * self.legendre[ell], self.mu_vector)

        return pk_dict

    def compute_pk_ell(self, bin_z):
        """Compute the power spectrum multipoles for a given redshift bin."""
        if self.redshift_distributions.nz_type == "widebin":
            z_avg = self.redshift_distributions.z_average(bin_z)
        elif self.redshift_distributions.nz_type == "thinbin":
            z_avg = self.redshift_distributions.nz_data[bin_z, 0]
        f = self.cosmo.growth_rate(z_avg)
        Sigma_tot_vector = self.compute_sigma_tot_vector(z_avg, f)
        
        pk_ell_dict = {f"Pk_{ell}_{component}": np.zeros(len(self.k)) for ell in self.ells for component in self.components}
        
        print(f"{bin_z} - Computing Pk_ell...")
        if self.use_multiprocessing:
            print(f"WARNING: P_ell(k) will be computed for all k values in parallel using {self.n_cpu} CPUs!")
            with Pool(self.n_cpu) as pool:
                pk_dict = pool.map(partial(self.compute_pk_ell_singlek, bin_z, f, Sigma_tot_vector), range(len(self.k)), chunksize=len(self.k) // self.n_cpu)
        else:
            raise NotImplementedError("Sequential computation of P(k) multipoles without multiprocessing is not implemented.")

        for i, result in enumerate(pk_dict):
            for key in result:
                pk_ell_dict[key][i] = result[key]

        for component in self.components:
            data = np.column_stack([self.k] + [pk_ell_dict[f"Pk_{ell}_{component}"] for ell in self.ells])
            np.savetxt(f"{self.path_template}/Pk_ell_{component}_bin{bin_z}.txt", data)

        print(f"{bin_z} - Pk_ell computed!")
        return pk_ell_dict
    
class CorrelationFunctionMultipoles:
    def __init__(self, template_initializer):
        """
        Initialize the CorrelationFunctionMultipoles class.

        Parameters:
        - template_initializer: Instance of the TemplateInitializer class.
        """
        self.template_initializer = template_initializer
        self.path_template = self.template_initializer.path_template
        self.k = self.template_initializer.k
        self.redshift_distributions = self.template_initializer.redshift_distributions
        self.cosmo = self.template_initializer.cosmo
        self.mu_vector = self.template_initializer.mu_vector
        self.legendre = self.template_initializer.legendre
        self.ells = self.template_initializer.ells
        self.components = self.template_initializer.components
        self.use_multiprocessing = self.template_initializer.use_multiprocessing
        self.n_cpu = self.template_initializer.n_cpu
        self.Nr = self.template_initializer.Nr
        self.r_12_vector = self.template_initializer.r_12_vector
        
        self.verbose = self.template_initializer.verbose
        
    def compute_xi_ell_singler(self, pk_ell_dict, r):
        """
        Compute the correlation function multipoles for a single value of r.

        Parameters:
        - pk_ell_dict (dict): Power spectrum multipoles.
        - r (float): Radial distance.
        """
        x = self.k * r
        x_square_inv = 1 / x**2
        sinc_x = np.sin(x) / x
        cos_x_redef = np.cos(x) * x_square_inv

        j_0_x = sinc_x
        j_2_x = (3 * x_square_inv - 1) * sinc_x - 3 * cos_x_redef
        j_4_x = 5 * (2 - 21 * x_square_inv) * cos_x_redef + (1 - 45 * x_square_inv + 105 * x_square_inv**2) * sinc_x

        xi_dict = {}
        for ell, j_ell_x in zip(self.ells, [j_0_x, j_2_x, j_4_x]):
            for component in self.components:
                xi_dict[f"{ell}_{component}"] = np.trapz(
                    (1j**ell).real * j_ell_x * self.k**2 / (2 * np.pi**2) * pk_ell_dict[f"{ell}_{component}"], self.k
                )
        return xi_dict
    
    def compute_xi_ell(self, bin_z):
        """
        Compute the correlation function multipoles for a given redshift bin.

        Parameters:
        - bin_z (int): Redshift bin number.
        """
        pk_ell_dict = self.template_initializer.load_pk_ell(bin_z)
        
        xi_ell_dict = {f"{ell}_{component}": np.zeros(self.Nr) for component in self.components for ell in self.ells}
        
        print(f"{bin_z} - Computing xi_ell...")
        if self.use_multiprocessing:
            print(f"WARNING: xi_ell(r) will be computed for all r values in parallel using {self.n_cpu} CPUs!")
            with Pool(self.n_cpu) as pool:
                xi_dict = pool.map(partial(self.compute_xi_ell_singler, pk_ell_dict), self.r_12_vector, chunksize=len(self.r_12_vector) // self.n_cpu)
        else:
            raise NotImplementedError("Sequential computation of xi(r) multipoles without multiprocessing is not implemented.")

        for i, result in enumerate(xi_dict):
            for key, value in result.items():
                xi_ell_dict[key][i] = value

        for component in self.components:
            np.savetxt(
                f"{self.path_template}/xi_ell_{component}_bin{bin_z}.txt",
                np.column_stack([self.r_12_vector] + [xi_ell_dict[f"{ell}_{component}"] for ell in self.ells])
            )

        print(f"{bin_z} - xi_ell computed!")
        return xi_ell_dict
    
class WThetaCalculator:
    def __init__(self, template_initializer):
        """
        Initialize the WThetaCalculator class.

        Parameters:
        - template_initializer: Instance of the TemplateInitializer class.
        """
        self.template_initializer = template_initializer
        self.redshift_distributions = self.template_initializer.redshift_distributions
        self.cosmo = self.template_initializer.cosmo
        self.path_template = self.template_initializer.path_template
        self.r_12_vector = self.template_initializer.r_12_vector
        self.mu_vector = self.template_initializer.mu_vector
        self.legendre = self.template_initializer.legendre
        self.ells = self.template_initializer.ells
        self.components = self.template_initializer.components
        self.use_multiprocessing = self.template_initializer.use_multiprocessing
        self.n_cpu = self.template_initializer.n_cpu
        self.theta = self.template_initializer.theta
        self.Nz = self.template_initializer.Nz
        
        self.verbose = self.template_initializer.verbose

    def wtheta_calculator(self, bin_z, xi_ell_dict, theta):
        """
        Compute the wtheta for a given redshift bin and a single value of theta.

        Parameters:
        - bin_z (int): Redshift bin number.
        - xi_ell_dict (dict): Correlation function multipoles.
        - theta (float): A single value of theta (angular separation) to calculate wtheta for.
        """
        if self.redshift_distributions.nz_type == "widebin":
            z_values = self.redshift_distributions.z_values(bin_z, Nz=self.Nz, verbose=False)
        elif self.redshift_distributions.nz_type == "thinbin":
            z_values = np.linspace(self.redshift_distributions.z_edges[bin_z][0], self.redshift_distributions.z_edges[bin_z][1], self.Nz)
        phi_values = self.redshift_distributions.nz_interp(z_values, bin_z) * self.cosmo.growth_factor(z_values)
        r_values = self.cosmo.comoving_radial_distance(z_values) / self.cosmo.h
        
        integrand = {component: np.zeros((len(z_values), len(z_values))) for component in self.components}

        r_12_values = np.sqrt(r_values[:, None]**2 + r_values[None, :]**2 - 2 * r_values[:, None] * r_values[None, :] * np.cos(theta))
        mu_values = (r_values[None, :] - r_values[:, None]) / r_12_values

        xi_ell_dict_interp = {}
        legendre_interp = {}

        for component in self.components:
            xi_ell_dict_interp[component] = {}
            legendre_interp[component] = {}
            for ell in self.ells:
                xi_ell_dict_interp[component][ell] = np.interp(r_12_values, self.r_12_vector, xi_ell_dict[f"{ell}_{component}"])
                legendre_interp[component][ell] = np.interp(mu_values, self.mu_vector, self.legendre[ell])

        for component in self.components:
            integrand_sum = np.zeros_like(r_12_values)
            for ell in self.ells:
                integrand_sum += xi_ell_dict_interp[component][ell] * legendre_interp[component][ell]

            integrand[component] = phi_values[:, None] * phi_values[None, :] * integrand_sum

        for key in integrand.keys():
            integrand[key] = np.triu(integrand[key]) + np.triu(integrand[key], 1).T
                
        wtheta_dict = {
            component: np.trapz(np.trapz(integrand[component], z_values), z_values)
            for component in self.components
        }
        return wtheta_dict

    def compute_wtheta(self, bin_z):
        """
        Compute and save the w(theta) for a given redshift bin.

        Parameters:
        - bin_z (int): Redshift bin number.
        """
        xi_ell_dict = self.template_initializer.load_xi_ell(bin_z)
        
        print(f"{bin_z} - Computing w(theta)...")
        if self.use_multiprocessing:
            print(f"WARNING: w(theta) will be computed for all theta values in parallel using {self.n_cpu} CPUs!")
            with Pool(self.n_cpu) as pool:
                w_dict = pool.map(partial(self.wtheta_calculator, bin_z, xi_ell_dict), self.theta, chunksize=len(self.theta) // self.n_cpu)
        else:
            raise NotImplementedError("Sequential computation of w(theta) without multiprocessing is not implemented.")

        wtheta_dict = {
            component: np.array([w_dict[i][component] for i in range(len(self.theta))])
            for component in self.components
        }

        for component in self.components:
            np.savetxt(
                f"{self.path_template}/wtheta_{component}_bin{bin_z}.txt",
                np.column_stack([
                    self.theta,
                    wtheta_dict[component]
                ])
            )
        
        print(f"{bin_z} - w(theta) computed!")
        return wtheta_dict

class CellCalculator:
    def __init__(self, template_initializer):
        """
        Initialize the C_ellCalculator class.

        Parameters:
        - template_initializer: Instance of the TemplateInitializer class.
        """
        self.template_initializer = template_initializer
        self.nbins = self.template_initializer.nbins
        self.n_cpu = self.template_initializer.n_cpu
        self.use_multiprocessing = self.template_initializer.use_multiprocessing
        self.path_template = self.template_initializer.path_template
        self.components = self.template_initializer.components
        
        self.max_ell = 10**4 # why not
        self.ell = range(self.max_ell)

        self.wtheta_interp = {component: {} for component in self.components}
        self.theta = None
        self.sin_theta = None
        self.cos_theta = None

        self._load_and_interpolate_wtheta()

    def _load_and_interpolate_wtheta(self):
        """
        Load and interpolate wtheta for all redshift bins.
        """
        theta = self.template_initializer.load_wtheta(0)[self.components[0]][:, 0]
        self.theta = 10**np.linspace(np.log10(theta.min() * 1.0001), np.log10(theta.max() * 0.999), 10**5)
        self.sin_theta = np.sin(self.theta)
        self.cos_theta = np.cos(self.theta)

        for bin_z in range(self.nbins):
            wtheta_dict = self.template_initializer.load_wtheta(bin_z)
            for component in self.components:
                wtheta_interp_fun = scipy.interpolate.interp1d(theta, wtheta_dict[component][:, 1], kind='linear')
                self.wtheta_interp[component][bin_z] = wtheta_interp_fun(self.theta)

    def C_ell_calculator(self, bin_z, ell):
        """
        Calculate C_ell for a given redshift bin and single value of ell.
        """
        legendre = scipy.special.eval_legendre(ell, self.cos_theta)
        C_ell_dict = {
            component: 2 * np.pi * np.trapz(self.sin_theta * self.wtheta_interp[component][bin_z] * legendre, self.theta)
            for component in self.components
        }
        return C_ell_dict

    def compute_C_ell(self, bin_z):
        """
        Compute and save the C_ell for a given redshift bin.
        """
        print(f"{bin_z} - Computing C_ell...")
        if self.use_multiprocessing:
            print(f"WARNING: C_ell will be computed for all ell values in parallel using {self.n_cpu} CPUs!")
            with Pool(self.n_cpu) as pool:
                C_dict = pool.map(partial(self.C_ell_calculator, bin_z), self.ell, chunksize=len(self.ell) // self.n_cpu)
        else:
            raise NotImplementedError("Sequential computation of C_ell without multiprocessing is not implemented.")

        C_ell_dict = {
            component: np.array([C_dict[i][component] for i in range(len(self.ell))])
            for component in self.components
        }
        
        for component in self.components:
            np.savetxt(
                f"{self.path_template}/C_ell_{component}_bin{bin_z}.txt", 
                np.column_stack([
                    self.ell, 
                    C_ell_dict[component]
                ])
            )
        
        print(f"{bin_z} - C_ell computed!")
        return C_ell_dict
