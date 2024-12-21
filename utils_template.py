import numpy as np
import scipy.special
from cosmoprimo import PowerSpectrumBAOFilter
import multiprocessing
from functools import partial

class PowerSpectrumMultipoles:
    def __init__(self, cosmo, include_wiggles, nz_instance, Nk, Nmu, path_template, n_cpu=None):
        self.cosmo = cosmo
        self.Nmu = Nmu
        self.path_template = path_template
        self.include_wiggles = include_wiggles
        self.nz_instance = nz_instance
        self.n_cpu = n_cpu if n_cpu is not None else multiprocessing.cpu_count()

        self.Nk = Nk
        self.kh = 10 ** np.linspace(np.log10(1e-4 / cosmo.h), np.log10(1e2), self.Nk)
        pkz = cosmo.get_fourier(engine='class').pk_interpolator()
        pk = pkz.to_1d(z=0)
        self.Pk_wigg = pk(self.kh) / cosmo.h**3

        pknow = PowerSpectrumBAOFilter(pk, engine='wallish2018').smooth_pk_interpolator()
        self.Pk_nowigg = pknow(self.kh) / cosmo.h**3

        self.k = self.kh * cosmo.h
        np.savetxt(f'{path_template}/Pk_full.txt', np.column_stack([self.k, self.Pk_wigg, self.Pk_nowigg]))
        
        self.Sigma_0, self.delta_Sigma_0 = self.compute_sigma_parameters()

        self.ells = [0, 2, 4]
        self.mu_vector = np.linspace(-1, 1, Nmu)
        self.legendre = {ell: scipy.special.eval_legendre(ell, self.mu_vector) for ell in self.ells}
        
        # Different bias components (squared, linear and independent)
        self.components = ['bb', 'bf', 'ff']
        
    def compute_sigma_parameters(self):
        """Compute Sigma_0 and delta_Sigma_0 using the given cosmology."""
        k_s = 0.2 * self.cosmo.h
        ell_BAO = 110 / self.cosmo.h

        q = 10 ** np.linspace(np.log10(self.k.min()), np.log10(k_s), self.Nk)
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
            + f * self.mu_vector**2 * (self.mu_vector**2 - 1) * delta_Sigma**2
        )

    def compute_pk_multipoles(self, bin_z, f, Sigma_tot_vector, i):
        """Compute the Pk multipoles."""
        if self.include_wiggles == '':
            pk_term = (self.Pk_wigg[i] - self.Pk_nowigg[i]) * np.exp(-self.k[i]**2 * Sigma_tot_vector**2) + self.Pk_nowigg[i]
        elif self.include_wiggles == '_nowiggles':
            pk_term = self.Pk_nowigg[i]
        
        pk_dict = {}
        for ell in self.ells:
            pk_dict[f'Pk_{ell}_bb'] = (2 * ell + 1) / 2 * np.trapz(pk_term * self.legendre[ell], self.mu_vector)
            pk_dict[f'Pk_{ell}_bf'] = (2 * ell + 1) / 2 * np.trapz(2 * self.mu_vector**2 * f * pk_term * self.legendre[ell], self.mu_vector)
            pk_dict[f'Pk_{ell}_ff'] = (2 * ell + 1) / 2 * np.trapz(self.mu_vector**4 * f**2 * pk_term * self.legendre[ell], self.mu_vector)

        return pk_dict

    def compute_pk_ell(self, bin_z):
        """Main method to compute the power spectrum multipoles for a given bin."""
        z = self.nz_instance.z_average(bin_z)
        f = self.cosmo.growth_rate(z)
        Sigma_tot_vector = self.compute_sigma_tot_vector(z, f)
        
        print("WARNING: P_ell(k) will be computed for all k values in parallel!")
        print(f"{bin_z} - Computing Pk_ell...")

        pk_ell_dict = {f'Pk_{ell}_{component}': np.zeros(len(self.k)) for ell in self.ells for component in self.components}

        with multiprocessing.Pool(self.n_cpu) as pool:
            pk_dict = pool.map(partial(self.compute_pk_multipoles, bin_z, f, Sigma_tot_vector), range(len(self.k)))
            
        for i, result in enumerate(pk_dict):
            for key in result:
                pk_ell_dict[key][i] = result[key]

        for component in self.components:
            data = np.column_stack([self.k] + [pk_ell_dict[f'Pk_{ell}_{component}'] for ell in self.ells])
            np.savetxt(f"{self.path_template}/Pk_ell_{component}_bin{bin_z}.txt", data)

        print(f"{bin_z} - Pk_ell computed!")
        return pk_ell_dict

class CorrelationFunctionMultipoles:
    def __init__(self, power_spectrum_multipoles, Nr):
        """
        Initialize the CorrelationFunctionMultipoles class.

        Parameters:
        - power_spectrum_multipoles: Instance of PowerSpectrumMultipoles class.
        - Nr: The number of radial bins.
        """
        self.path_template = power_spectrum_multipoles.path_template
        self.Nr = Nr
        self.k = power_spectrum_multipoles.k

        self.r_12_vector = 10**np.linspace(np.log10(10**-2), np.log10(10**5), Nr)
        
        self.nz_instance = power_spectrum_multipoles.nz_instance
        self.cosmo = power_spectrum_multipoles.cosmo
        self.mu_vector = power_spectrum_multipoles.mu_vector
        self.legendre = power_spectrum_multipoles.legendre
        self.ells = power_spectrum_multipoles.ells
        self.components = power_spectrum_multipoles.components
        self.n_cpu = power_spectrum_multipoles.n_cpu

    def load_pk_ell_data(self, bin_z):
        """
        Load precomputed Pk_ell data from files for a given redshift bin.

        Parameters:
        - bin_z: Redshift bin number to load the data for.
        
        Returns:
        - pk_ell_dict: Dictionary containing precomputed Pk_ell data for different multipoles.
        """
        print(f"{bin_z} - Loading precomputed Pk_ell data...")

        pk_ell_dict = {
            f'Pk_{ell}_{component}': np.loadtxt(f"{self.path_template}/Pk_ell_{component}_bin{bin_z}.txt")[:, idx + 1]
            for component in self.components
            for idx, ell in enumerate(self.ells)
        }
        return pk_ell_dict

    def compute_xi_multipoles(self, pk_ell_dict, r):
        """
        Compute the correlation function multipoles (xi_ell) for a given radial distance.

        Parameters:
        - r: Radial distance.
        - pk_ell_dict: Dictionary containing Pk_ell data.
        
        Returns:
        - xi_dict: Dictionary containing xi_ell values for different types.
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
                xi_dict[f'xi_{ell}_{component}'] = np.trapz(
                    j_ell_x * self.k**2 / (2 * np.pi**2) * pk_ell_dict[f'Pk_{ell}_{component}'], self.k
                )
        return xi_dict
    
    def compute_xi_ell(self, bin_z):
        """
        Main method to compute the correlation function multipoles (xi_ell) for a given redshift bin.

        Parameters:
        - bin_z: Redshift bin number.

        Returns:
        - xi_ell_dict: Dictionary containing the xi_ell values for different types.
        """
        print("WARNING: xi_ell(r) will be computed for all r values in parallel!")
        print(f"{bin_z} - Computing xi_ell...")

        pk_ell_dict = self.load_pk_ell_data(bin_z)

        xi_ell_dict = {f'xi_{ell}_{component}': np.zeros(self.Nr) for component in self.components for ell in self.ells}

        with multiprocessing.Pool(self.n_cpu) as pool:
            xi_dict = pool.map(partial(self.compute_xi_multipoles, pk_ell_dict), self.r_12_vector)

        for i, result in enumerate(xi_dict):
            for key, value in result.items():
                xi_ell_dict[key][i] = value

        for component in self.components:
            np.savetxt(
                f"{self.path_template}/xi_ell_{component}_bin{bin_z}.txt",
                np.column_stack([self.r_12_vector] + [xi_ell_dict[f'xi_{ell}_{component}'] for ell in self.ells])
            )

        print(f"{bin_z} - xi_ell computed!")
        return xi_ell_dict

class WThetaCalculator:
    def __init__(self, correlation_function_multipoles, Nz, Ntheta):
        """
        Initialize the WThetaCalculator class.

        Parameters:
        - correlation_function_multipoles: Instance of the class containing correlation function multipoles (xi_ell).
        - Nz: Number of redshift bins for the double redshift integral.
        - Ntheta: Number of theta bins.
        - n_cpu: Number of CPU cores to use for multiprocessing. Default: all available cores.
        """
        self.nz_instance = correlation_function_multipoles.nz_instance
        self.cosmo = correlation_function_multipoles.cosmo
        self.path_template = correlation_function_multipoles.path_template
        self.r_12_vector = correlation_function_multipoles.r_12_vector
        self.mu_vector = correlation_function_multipoles.mu_vector
        self.legendre = correlation_function_multipoles.legendre
        self.ells = correlation_function_multipoles.ells
        self.components = correlation_function_multipoles.components
        self.n_cpu = correlation_function_multipoles.n_cpu
        self.Nz = Nz
        self.theta = 10**np.linspace(np.log10(0.001), np.log10(179.5), Ntheta) * np.pi / 180

    def load_xi_ell_data(self, bin_z):
        """
        Load precomputed xi_ell data (e.g., xi_0, xi_2, xi_4) for a given redshift bin.

        Parameters:
        - bin_z: Redshift bin number.

        Returns:
        - xi_ell_dict: Dictionary containing the xi_ell data for different multipoles.
        """
        print(f"{bin_z} - Loading precomputed xi_ell data...")
        
        xi_ell_dict = {
            f'xi_{ell}_{component}': np.loadtxt(f"{self.path_template}/xi_ell_{component}_bin{bin_z}.txt")[:, ell_idx + 1]
            for component in self.components
            for ell_idx, ell in enumerate(self.ells)
        }
        return xi_ell_dict

    def wtheta_calculator(self, bin_z, xi_ell_dict, theta):
        """
        Compute the wtheta for a given redshift bin and a single value of theta.

        Parameters:
        - bin_z: Redshift bin number.
        - theta: A single value of theta (angular separation) to calculate wtheta for.

        Returns:
        - wtheta_dict: Dictionary containing the wtheta values for different correlation types (bb, bf, ff).
        """
        z_values = self.nz_instance.z_vector(bin_z, Nz=self.Nz, verbose=False)
        D_values = self.cosmo.growth_factor(z_values)
        phi_values = self.nz_instance.nz_interp(z_values, bin_z) * D_values
        r_values = self.cosmo.comoving_radial_distance(z_values) / self.cosmo.h
        
        integrand = {f'integrand_{component}': np.zeros((len(z_values), len(z_values))) for component in self.components}
        
#         for i in range(len(z_values)):
#             for j in range(len(z_values)):
#                 if j < i:
#                     for key in integrand.keys():
#                         integrand[key][i, j] = integrand[key][j, i]
#                 else:
#                     r_12 = np.sqrt(r_values[i]**2 + r_values[j]**2 - 2 * r_values[i] * r_values[j] * np.cos(theta))
#                     mu = (r_values[j] - r_values[i]) / r_12

#                     try:
#                         for component in self.components:
#                             integrand[f'integrand_{component}'][i, j] = phi_values[i] * phi_values[j] * sum(
#                                 np.interp(r_12, self.r_12_vector, xi_ell_dict[f'xi_{ell}_{component}']) * 
#                                 np.interp(mu, self.mu_vector, self.legendre[ell])
#                                 for ell in self.ells
#                             )
#                     except ValueError:
#                         print(f"Error for r_12={r_12}, mu={mu}")

        r_12_values = np.sqrt(r_values[:, None]**2 + r_values[None, :]**2 - 2 * r_values[:, None] * r_values[None, :] * np.cos(theta))
        mu_values = (r_values[None, :] - r_values[:, None]) / r_12_values

        xi_ell_dict_interp = {}
        legendre_interp = {}

        for component in self.components:
            xi_ell_dict_interp[component] = {}
            legendre_interp[component] = {}
            for ell in self.ells:
                xi_ell_dict_interp[component][ell] = np.interp(r_12_values, self.r_12_vector, xi_ell_dict[f'xi_{ell}_{component}'])
                legendre_interp[component][ell] = np.interp(mu_values, self.mu_vector, self.legendre[ell])

        for component in self.components:
            integrand_sum = np.zeros_like(r_12_values)
            for ell in self.ells:
                integrand_sum += xi_ell_dict_interp[component][ell] * legendre_interp[component][ell]

            integrand_key = f'integrand_{component}'
            integrand[f'{integrand_key}'] = phi_values[:, None] * phi_values[None, :] * integrand_sum

        for key in integrand.keys():
            integrand[key] = np.triu(integrand[key]) + np.triu(integrand[key], 1).T
                
        wtheta_dict = {
            f'wtheta_{component}': np.trapz(np.trapz(integrand[f'integrand_{component}'], z_values), z_values)
            for component in self.components
        }
        return wtheta_dict

    def compute_wtheta(self, bin_z):
        """
        Compute and save wtheta for a given bin_z using multiprocessing.

        Parameters:
        - bin_z: Redshift bin number.
        """
        print("WARNING: w(theta) will be computed for all theta values in parallel!")
        print(f"{bin_z} - Computing w(theta)...")
        
        xi_ell_dict = self.load_xi_ell_data(bin_z)

        with multiprocessing.Pool(self.n_cpu) as pool:
            w_dict = pool.map(partial(self.wtheta_calculator, bin_z, xi_ell_dict), self.theta)

        wtheta_dict = {
            f'wtheta_{component}': np.array([w_dict[i][f'wtheta_{component}'] for i in range(len(self.theta))])
            for component in self.components
        }

        for component in self.components:
            np.savetxt(
                f"{self.path_template}/wtheta_{component}_bin{bin_z}.txt",
                np.column_stack([
                    self.theta,
                    wtheta_dict[f'wtheta_{component}']
                ])
            )
        
        print(f"{bin_z} - w(theta) computed!")
        return wtheta_dict