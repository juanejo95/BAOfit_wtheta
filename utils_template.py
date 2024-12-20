import numpy as np
import scipy.special
from cosmoprimo import PowerSpectrumBAOFilter
import multiprocessing
from functools import partial

class PowerSpectrumMultipoles:
    def __init__(self, cosmo, include_wiggles, nz_instance, Nk, Nmu, path):
        self.cosmo = cosmo
        self.Nmu = Nmu
        self.path = path
        self.include_wiggles = include_wiggles
        self.nz_instance = nz_instance

        # Precompute k, Pk_wigg, and Pk_nowigg
        self.Nk = Nk
        self.kh = 10 ** np.linspace(np.log10(1e-4 / cosmo.h), np.log10(1e2), self.Nk)
        pkz = cosmo.get_fourier(engine='class').pk_interpolator()
        pk = pkz.to_1d(z=0)
        self.Pk_wigg = pk(self.kh) / cosmo.h**3

        pknow = PowerSpectrumBAOFilter(pk, engine='wallish2018').smooth_pk_interpolator()
        self.Pk_nowigg = pknow(self.kh) / cosmo.h**3

        self.k = self.kh * cosmo.h  # Precompute k in units of h
        np.savetxt(f'{path}/Pk_full.txt', np.column_stack([self.k, self.Pk_wigg, self.Pk_nowigg]))
        
        self.Sigma_0, self.delta_Sigma_0 = self.compute_sigma_parameters()

        # Precompute mu_vector and legendre polynomials for ell = 0, 2, 4
        self.ells = [0, 2, 4]
        self.mu_vector = np.linspace(-1, 1, Nmu)
        self.legendre = {ell: scipy.special.eval_legendre(ell, self.mu_vector) for ell in self.ells}
        
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

    def compute_pk_multipoles(self, i, bin_z, Sigma_tot_vector):
        """Compute the Pk multipoles."""
        z = self.nz_instance.z_average(bin_z)
        f = self.cosmo.growth_rate(z)
        if self.include_wiggles == '':
            pk_term = (self.Pk_wigg[i] - self.Pk_nowigg[i]) * np.exp(-self.k[i]**2 * Sigma_tot_vector**2) + self.Pk_nowigg[i]
        elif self.include_wiggles == '_nowiggles':
            pk_term = self.Pk_nowigg[i]
        
        pk_dict = {}
        for ell in self.ells:
            leg = self.legendre[ell]
            coeff = (2 * ell + 1) / 2

            pk_dict[f'Pk_{ell}_bb'] = coeff * np.trapz(pk_term * leg, self.mu_vector)
            pk_dict[f'Pk_{ell}_bf'] = coeff * np.trapz(2 * self.mu_vector**2 * f * pk_term * leg, self.mu_vector)
            pk_dict[f'Pk_{ell}_ff'] = coeff * np.trapz(self.mu_vector**4 * f**2 * pk_term * leg, self.mu_vector)

        return pk_dict

    def compute_pk_ell(self, bin_z):
        """Main method to compute the power spectrum multipoles for a given bin."""
        z = self.nz_instance.z_average(bin_z)
        f = self.cosmo.growth_rate(z)
        Sigma_tot_vector = self.compute_sigma_tot_vector(z, f)

        print(f"{bin_z} - Computing Pk_ell...")

        pk_ell_dict = {key: np.zeros(len(self.k)) for key in [
            'Pk_0_bb', 'Pk_2_bb', 'Pk_4_bb',
            'Pk_0_bf', 'Pk_2_bf', 'Pk_4_bf',
            'Pk_0_ff', 'Pk_2_ff', 'Pk_4_ff'
        ]}

        for i in range(len(self.k)):
            pk_dict = self.compute_pk_multipoles(i, bin_z, Sigma_tot_vector)
            for key, value in pk_dict.items():
                pk_ell_dict[key][i] = value

        # Save pk_ell_dict
        components = ['bb', 'bf', 'ff']
        for comp in components:
            np.savetxt(
                f"{self.path}/Pk_ell_{comp}_bin{bin_z}.txt",
                np.column_stack([
                    self.k,
                    pk_ell_dict[f'Pk_0_{comp}'],
                    pk_ell_dict[f'Pk_2_{comp}'],
                    pk_ell_dict[f'Pk_4_{comp}']
                ])
            )
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
        self.path = power_spectrum_multipoles.path
        self.Nr = Nr
        self.k = power_spectrum_multipoles.k  # Get the k array from PowerSpectrumMultipoles instance

        # Define r_12_vector directly within the class
        self.r_12_vector = 10**np.linspace(np.log10(10**-2), np.log10(10**5), Nr)
        
        self.nz_instance = power_spectrum_multipoles.nz_instance
        self.cosmo = power_spectrum_multipoles.cosmo
        self.mu_vector = power_spectrum_multipoles.mu_vector
        self.legendre = power_spectrum_multipoles.legendre
        self.ells = power_spectrum_multipoles.ells

    def load_pk_ell_data(self, bin_z):
        """
        Load precomputed Pk_ell data from files for a given redshift bin.

        Parameters:
        - bin_z: Redshift bin number to load the data for.
        
        Returns:
        - pk_ell_dict: Dictionary containing precomputed Pk_ell data for different multipoles.
        """
        print(f"{bin_z} - Loading precomputed Pk_ell data...")
        components = ['bb', 'bf', 'ff']

        pk_ell_dict = {
            f'Pk_{ell}_{comp}': np.loadtxt(f"{self.path}/Pk_ell_{comp}_bin{bin_z}.txt")[:, ell_idx + 1]
            for comp in components
            for ell_idx, ell in enumerate(self.ells)
        }
        return pk_ell_dict

    def compute_xi_multipoles(self, r, pk_ell_dict):
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
            xi_dict[f'xi_{ell}_bb'] = np.trapz(j_ell_x * self.k**2 / (2 * np.pi**2) * pk_ell_dict[f'Pk_{ell}_bb'], self.k)
            xi_dict[f'xi_{ell}_bf'] = np.trapz(j_ell_x * self.k**2 / (2 * np.pi**2) * pk_ell_dict[f'Pk_{ell}_bf'], self.k)
            xi_dict[f'xi_{ell}_ff'] = np.trapz(j_ell_x * self.k**2 / (2 * np.pi**2) * pk_ell_dict[f'Pk_{ell}_ff'], self.k)

        return xi_dict

    def compute_xi_ell(self, bin_z):
        """
        Main method to compute the correlation function multipoles (xi_ell) for a given redshift bin.
        
        Parameters:
        - bin_z: Redshift bin number.
        
        Returns:
        - xi_ell_dict: Dictionary containing the xi_ell values for different types.
        """
        print(f"{bin_z} - Computing xi_ell...")

        # Load the precomputed Pk_ell data
        pk_ell_dict = self.load_pk_ell_data(bin_z)

        xi_ell_dict = {key: np.zeros(self.Nr) for key in [
            'xi_0_bb', 'xi_2_bb', 'xi_4_bb',
            'xi_0_bf', 'xi_2_bf', 'xi_4_bf',
            'xi_0_ff', 'xi_2_ff', 'xi_4_ff'
        ]}

        for i, r_12 in enumerate(self.r_12_vector):
            xi_dict = self.compute_xi_multipoles(r_12, pk_ell_dict)
            for key, value in xi_dict.items():
                xi_ell_dict[key][i] = value

            if (i + 1) % (self.Nr // 10) == 0:
                print(f"{bin_z} - {int((i + 1) / self.Nr * 100)}%")

        # Save xi_ell_dict
        components = ['bb', 'bf', 'ff']

        for comp in components:
            np.savetxt(
                f"{self.path}/xi_ell_{comp}_bin{bin_z}.txt",
                np.column_stack([
                    self.r_12_vector,
                    xi_ell_dict[f'xi_0_{comp}'],
                    xi_ell_dict[f'xi_2_{comp}'],
                    xi_ell_dict[f'xi_4_{comp}']
                ])
            )

        print(f"{bin_z} - xi_ell computed!")
        return xi_ell_dict

class WThetaCalculator:
    def __init__(self, correlation_function_multipoles, Nz, theta=None, n_cpu=None):
        """
        Initialize the WThetaCalculator class.

        Parameters:
        - correlation_function_multipoles: Instance of the class containing correlation function multipoles (xi_ell).
        - Nz: Number of redshift bins.
        - theta: Array of theta values (angular separations in radians). Default: log-spaced between 0.001° and 179.5°.
        - n_cpu: Number of CPU cores to use for multiprocessing. Default: all available cores.
        """
        self.nz_instance = correlation_function_multipoles.nz_instance
        self.cosmo = correlation_function_multipoles.cosmo
        self.path = correlation_function_multipoles.path
        self.r_12_vector = correlation_function_multipoles.r_12_vector
        self.mu_vector = correlation_function_multipoles.mu_vector
        self.legendre = correlation_function_multipoles.legendre
        self.ells = correlation_function_multipoles.ells
        self.Nz = Nz

        # Define default theta if not provided
        self.theta = theta if theta is not None else 10**np.linspace(np.log10(0.001), np.log10(179.5), 10**2) * np.pi / 180

        # Define default n_cpu if not provided
        self.n_cpu = n_cpu if n_cpu is not None else multiprocessing.cpu_count()

    def load_xi_ell_data(self, bin_z):
        """
        Load precomputed xi_ell data (e.g., xi_0, xi_2, xi_4) for a given redshift bin.

        Parameters:
        - bin_z: Redshift bin number.

        Returns:
        - xi_ell_dict: Dictionary containing the xi_ell data for different multipoles.
        """
        components = ['bb', 'bf', 'ff']

        xi_ell_dict = {
            f'xi_{ell}_{comp}': np.loadtxt(f"{self.path}/xi_ell_{comp}_bin{bin_z}.txt")[:, ell_idx + 1]
            for comp in components
            for ell_idx, ell in enumerate(self.ells)
        }
        return xi_ell_dict

    def wtheta_calculator(self, bin_z, theta):
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
        
        xi_ell_dict = self.load_xi_ell_data(bin_z)

        integrand = {key: np.zeros((len(z_values), len(z_values))) for key in [
            'integrand_bb', 'integrand_bf', 'integrand_ff'
        ]}
        
        for i in range(len(z_values)):
            for j in range(len(z_values)):
                if j < i:
                    for key in integrand.keys():
                        integrand[key][i, j] = integrand[key][j, i]
                else:
                    r_12 = np.sqrt(r_values[i]**2 + r_values[j]**2 - 2 * r_values[i] * r_values[j] * np.cos(theta))
                    mu = (r_values[j] - r_values[i]) / r_12

                    try:
                        integrand['integrand_bb'][i, j] = phi_values[i] * phi_values[j] * sum([
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_0_bb']) * np.interp(mu, self.mu_vector, self.legendre[0]),
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_2_bb']) * np.interp(mu, self.mu_vector, self.legendre[2]),
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_4_bb']) * np.interp(mu, self.mu_vector, self.legendre[4])
                        ])
                        integrand['integrand_bf'][i, j] = phi_values[i] * phi_values[j] * sum([
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_0_bf']) * np.interp(mu, self.mu_vector, self.legendre[0]),
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_2_bf']) * np.interp(mu, self.mu_vector, self.legendre[2]),
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_4_bf']) * np.interp(mu, self.mu_vector, self.legendre[4])
                        ])
                        integrand['integrand_ff'][i, j] = phi_values[i] * phi_values[j] * sum([
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_0_ff']) * np.interp(mu, self.mu_vector, self.legendre[0]),
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_2_ff']) * np.interp(mu, self.mu_vector, self.legendre[2]),
                            np.interp(r_12, self.r_12_vector, xi_ell_dict['xi_4_ff']) * np.interp(mu, self.mu_vector, self.legendre[4])
                        ])
                    except ValueError:
                        print(f"Error for r_12={r_12}, mu={mu}")
                
        wtheta_dict = {
            'wtheta_bb': np.trapz(np.trapz(integrand['integrand_bb'], z_values), z_values),
            'wtheta_bf': np.trapz(np.trapz(integrand['integrand_bf'], z_values), z_values),
            'wtheta_ff': np.trapz(np.trapz(integrand['integrand_ff'], z_values), z_values),
        }
        return wtheta_dict

    def compute_and_save_wtheta(self, bin_z):
        """
        Compute and save wtheta for a given bin_z using multiprocessing.

        Parameters:
        - bin_z: Redshift bin number.
        """
        # Display warning
        print("WARNING: w(theta) will be computed for all theta in parallel!")
        print(f"{bin_z} - Computing w(theta)...")

        with multiprocessing.Pool(self.n_cpu) as pool:
            w_dict = pool.map(partial(self.wtheta_calculator, bin_z), self.theta)

        # Organize the results into dictionaries
        wtheta_dict = {
            'wtheta_bb': np.array([w_dict[i]['wtheta_bb'] for i in range(len(self.theta))]),
            'wtheta_bf': np.array([w_dict[i]['wtheta_bf'] for i in range(len(self.theta))]),
            'wtheta_ff': np.array([w_dict[i]['wtheta_ff'] for i in range(len(self.theta))]),
        }

        # Save wtheta results for each component
        components = ['bb', 'bf', 'ff']
        for comp in components:
            np.savetxt(
                f"{self.path}/wtheta_{comp}_bin{bin_z}.txt",
                np.column_stack([
                    self.theta,
                    wtheta_dict[f'wtheta_{comp}']
                ])
            )
        
        print(f"{bin_z} - w(theta) computed!")
        
        return wtheta_dict