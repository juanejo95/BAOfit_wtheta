import numpy as np
import sys

class RedshiftDistributions:
    def __init__(self, nz_flag, file_paths=None, verbose=True):
        """
        Initialize the redshift distributions based on the chosen flag.

        Args:
            nz_flag (str): Flag to select the redshift distribution ('fid', 'clusteringz').
            file_paths (dict, optional): Paths to the n(z) files. Default uses internal paths.
            verbose (bool): Whether to print initialization messages.
        """
        # Default file paths
        default_paths = {
            'fid': 'nz/nz_DNFpdf_shift_stretch_wrtclusteringz1-4_wrtVIPERS5-6_v2.txt',
            'clusteringz': 'nz/nz_clusteringz.txt'
        }
        file_paths = file_paths or default_paths
        
        if nz_flag not in file_paths:
            raise ValueError(f"Unknown redshift distribution: {nz_flag}")
        
        try:
            self.nz_data = np.loadtxt(file_paths[nz_flag])
        except OSError as e:
            raise FileNotFoundError(f"Error loading n(z) file for {nz_flag}: {e}")
        
        self.z_edges = {
            'fid': {0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0], 4: [1.0, 1.1], 5: [1.1, 1.2]},
            'clusteringz': {0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0]}
        }[nz_flag]

        self.nbins = len(self.nz_data.T) - 1
        if verbose:
            print(f"Using {nz_flag} n(z), which has {self.nbins} redshift bins")

    def __repr__(self):
        """String representation for debugging."""
        return f"RedshiftDistributions(nbins={self.nbins}, edges={self.z_edges})"

    def nz_interp(self, z, bin_z):
        """Interpolate n(z) for a given redshift z and bin."""
        return np.interp(z, self.nz_data[:, 0], self.nz_data[:, bin_z + 1])

    def z_average(self, bin_z):
        """Calculate the average redshift for a given bin."""
        return np.trapz(self.nz_data[:, 0] * self.nz_data[:, bin_z + 1], self.nz_data[:, 0])

    def z_vector(self, bin_z, Nz=100, verbose=True):
        """Generate a vector of redshift values around the average redshift."""
        z_avg = self.z_average(bin_z)
        z_vector = np.linspace(z_avg - 0.25, z_avg + 0.25, Nz)
        z_values = self.nz_interp(z_vector, bin_z)
        
        if verbose:
            print(f"[bin_z: {bin_z}, z_avg: {z_avg:.3f}, "
                  f"integral of the n(z) (total): {np.trapz(self.nz_data[:, bin_z + 1], self.nz_data[:, 0]):.3f}, "
                  f"integral of the n(z) (over the z range used): {np.trapz(z_values, z_vector):.3f}]")
        
        return z_vector

class WThetaDataCovariance:
    def __init__(self, dataset, weight_type, cov_type, cosmology_covariance, delta_theta, theta_min, theta_max, 
                 bins_removed, diag_only, remove_crosscov, nbins):
        self.dataset = dataset
        self.weight_type = weight_type
        self.cov_type = cov_type
        self.cosmology_covariance = cosmology_covariance
        self.delta_theta = delta_theta
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.bins_removed = bins_removed
        self.diag_only = diag_only
        self.remove_crosscov = remove_crosscov
        self.nbins = nbins

    def load_wtheta_data(self):
        indices_theta_allbins = {}
        theta_wtheta_data = {}
        wtheta_data = {}

        for bin_z in range(self.nbins):
            if self.dataset == 'DESY6':
                filename_wtheta = (f'wtheta_data_Yband/wtheta_data_bin{bin_z}_DeltaTheta{self.delta_theta}_'
                                   f'weights{self.weight_type}_fstar.txt')
            
            theta, wtheta = np.loadtxt(filename_wtheta).T

            indices_theta_individualbin = np.where((theta > self.theta_min * np.pi / 180) &
                                                   (theta < self.theta_max * np.pi / 180))[0]

            theta_wtheta_data[bin_z] = theta[indices_theta_individualbin]
            
            if bin_z in self.bins_removed:
                wtheta_data[bin_z] = np.zeros(len(indices_theta_individualbin))
            else:
                wtheta_data[bin_z] = wtheta[indices_theta_individualbin]

            indices_theta_allbins[bin_z] = indices_theta_individualbin + bin_z * len(theta)

        indices_theta_allbins_concatenated = np.concatenate([indices_theta_allbins[i] for i in range(self.nbins)])

        theta_wtheta_data_concatenated = np.concatenate([theta_wtheta_data[i] for i in range(self.nbins)])
        wtheta_data_concatenated = np.concatenate([wtheta_data[bin_z] for bin_z in range(self.nbins)])

        if all(np.array_equal(v, list(theta_wtheta_data.values())[0]) for v in theta_wtheta_data.values()):
            theta_data = list(theta_wtheta_data.values())[0]
            print("All theta values are the same. Using the first one as an array.")
        else:
            print("Theta values are different. Something seems to be wrong!")
            sys.exit()

        return theta_data, wtheta_data, indices_theta_allbins_concatenated, theta_wtheta_data_concatenated

    def load_covariance_matrix(self, indices_theta_allbins_concatenated, theta_wtheta_data_concatenated):
        if self.cov_type == 'cosmolike_data':
            if self.cosmology_covariance == 'mice':
                if self.delta_theta != 0.2:
                    print(f"No mice cosmolike covariance matrix for delta_theta={self.delta_theta}")
                    sys.exit()
                cov = np.loadtxt(
                    f"cov_cosmolike/cov_Y6bao_data_DeltaTheta{str(self.delta_theta).replace('.', 'p')}_mask_g_mice.txt"
                )
            elif self.cosmology_covariance == 'planck':
                cov = np.loadtxt(
                    f"cov_cosmolike/cov_Y6bao_data_DeltaTheta{str(self.delta_theta).replace('.', 'p')}_mask_g_planck.txt"
                )
            theta_cov = np.loadtxt(f"cov_cosmolike/delta_theta_{self.delta_theta}_binning.txt")[:, 2] * np.pi / 180

        theta_cov_concatenated = np.concatenate([theta_cov] * self.nbins)

        if abs(theta_wtheta_data_concatenated - theta_cov_concatenated[indices_theta_allbins_concatenated]).max() > 10**-5:
            print('The covariance matrix and the wtheta of the mocks do not have the same theta')
            sys.exit()

        cov_adjusted = np.zeros_like(cov)
        for bin_z1 in range(self.nbins):
            for bin_z2 in range(self.nbins):
                slice_1 = slice(bin_z1 * len(theta_cov), (bin_z1 + 1) * len(theta_cov))
                slice_2 = slice(bin_z2 * len(theta_cov), (bin_z2 + 1) * len(theta_cov))
                if bin_z1 == bin_z2 or (bin_z1 not in self.bins_removed and bin_z2 not in self.bins_removed):
                    cov_adjusted[slice_1, slice_2] = cov[slice_1, slice_2]
        cov = cov_adjusted

        if self.remove_crosscov == 'y':
            cov_adjusted = np.zeros_like(cov)
            for bin_z in range(self.nbins):
                slice_ = slice(bin_z * len(theta_cov), (bin_z + 1) * len(theta_cov))
                cov_adjusted[slice_, slice_] = cov[slice_, slice_]
            cov = cov_adjusted

        if self.diag_only == 'y':
            cov = np.diag(np.diag(cov))

        cov_orig = np.copy(cov)
        cov = cov_orig[indices_theta_allbins_concatenated[:, None], indices_theta_allbins_concatenated]

        return cov

    def process(self):
        theta_data, wtheta_data, indices_theta_allbins_concatenated, theta_wtheta_data_concatenated = self.load_wtheta_data()
        cov = self.load_covariance_matrix(indices_theta_allbins_concatenated, theta_wtheta_data_concatenated)
        return theta_data, wtheta_data, cov
    