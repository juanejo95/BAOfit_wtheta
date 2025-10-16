import numpy as np
import sys
import zipfile
import re
import warnings

class RedshiftDistributions:
    def __init__(self, dataset, nz_flag, verbose=True):
        """
        Initialize the RedshiftDistributions class.

        Parameters:
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - nz_flag (str): Identifier for the n(z).
        - verbose (bool): Whether to print messages.
        """
        self.dataset = dataset
        self.nz_flag = nz_flag
        self.verbose = verbose
        
        # File paths based on dataset and nz_flag
        if self.dataset in ["DESY6", "DESY6_dec_below-23.5", "DESY6_dec_above-23.5", "DESY6_DR1tiles_noDESI", "DESY6_DR1tiles_DESIonly"]: # they have the same n(z), but in different paths
            self.nz_type = "widebin"
            if self.nz_flag == "fid":
                file_path = f"datasets/{self.dataset}/nz/nz_DNFpdf_shift_stretch_wrtclusteringz1-4_wrtVIPERS5-6_v2.txt"
                self.z_edges = {
                    0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0], 4: [1.0, 1.1], 5: [1.1, 1.2]
                }
            elif nz_flag == "clusteringz":
                file_path = f"datasets/{self.dataset}/nz/nz_clusteringz.txt"
                self.z_edges = {
                    0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0]
                }
            else:
                raise ValueError(f"Unknown nz_flag: {self.nz_flag} for dataset: datasets/{self.dataset}")
        elif self.dataset in ["DESY6_COLA", "DESY6_COLA_dec_below-23.5", "DESY6_COLA_dec_above-23.5", "DESY6_COLA_DR1tiles_noDESI", "DESY6_COLA_DR1tiles_DESIonly"]:
            self.nz_type = "widebin"
            if self.nz_flag == "mocks":
                file_path = f"datasets/{self.dataset}/nz/nz_Y6COLA.txt"
                self.z_edges = {
                    0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0], 4: [1.0, 1.1], 5: [1.1, 1.2]
                }
            else:
                raise ValueError(f"Unknown nz_flag: {self.nz_flag} for dataset: datasets/{self.dataset}")
        elif "DESIY1_LRG" in self.dataset:
            self.nz_type = "thinbin"
            if self.nz_flag == "mocks":
                file_path = f"datasets/{self.dataset}/nz/mean_nzs.txt"
        else:
            raise ValueError(f"Unknown dataset: datasets/{self.dataset}")

        # Load the redshift distributions
        try:
            self.nz_data = np.loadtxt(file_path)
        except OSError as e:
            raise FileNotFoundError(f"Error loading n(z) file for datasets/{self.dataset} with nz_flag {self.nz_flag}: {e}")

        if self.nz_type == "widebin":
            self.nbins = len(self.nz_data.T) - 1
        elif self.nz_type == "thinbin":
            self.nz_data[:, 1] /= np.trapz(self.nz_data[:, 1], self.nz_data[:, 0]) # normalize to 1. We don't actually use it but just in case
            self.nbins = len(self.nz_data)
            z_centers = self.nz_data[:, 0]
            z_bins = (z_centers[:-1] + z_centers[1:]) / 2
            z_bins = np.concatenate([[2 * z_centers[0] - z_bins[0]], z_bins, [2 * z_centers[-1] - z_bins[-1]]])
            self.z_edges = {bin_z: [z_bins[bin_z], z_bins[bin_z + 1]] for bin_z in range(self.nbins)}
        if self.verbose:
            print(f"Using datasets/{self.dataset} {self.nz_flag} n(z), which has {self.nbins} redshift bins")

    def __repr__(self):
        """String representation for debugging."""
        return f"RedshiftDistributions(dataset=datasets/{self.dataset}, nz_flag={self.nz_flag}, nbins={self.nbins}, edges={self.z_edges})"

    def nz_interp(self, z, bin_z):
        """Interpolate n(z) for a given redshift z and bin."""
        if self.nz_type == "widebin":
            return np.interp(z, self.nz_data[:, 0], self.nz_data[:, bin_z + 1])
        elif self.nz_type == "thinbin":
            z_min, z_max = self.z_edges[bin_z]
            mask = (z_min <= z) & (z < z_max)
            result = np.zeros_like(z, dtype=float)
            result[mask] = 1 / (z_max - z_min) # it is just a top-hat function
            return result if z.ndim > 0 else result.item()

    def z_average(self, bin_z):
        """Calculate the average redshift for a given bin."""
        if self.nz_type == "widebin":
            return np.trapz(self.nz_data[:, 0] * self.nz_data[:, bin_z + 1], self.nz_data[:, 0])
        elif self.nz_type == "thinbin":
            raise NotImplementedError(f"No need to implement for dataset datasets/{self.dataset}.")

    def z_values(self, bin_z, Nz=10**3, target_area=0.99, verbose=True):
        """
        Generate a vector of redshift values such that the integral of n(z) over the range 
        is at least target_area.
        """
        if self.nz_type == "widebin":
            zmin_full = self.nz_data[:, 0].min()
            zmax_full = self.nz_data[:, 0].max()
    
            delta_z = 0.01
            
            zmin = zmin_full
            zmax = zmin + delta_z
            
            if zmax > zmax_full:
                raise ValueError(f"Initial zmax ({zmax:.3f}) exceeds the maximum redshift in the data ({zmax_full:.3f}).")
        
            while True:
                z_values = np.linspace(zmin, zmax, Nz)
                nz_values = self.nz_interp(z_values, bin_z)
                
                integral = np.trapz(nz_values, z_values)
                
                if integral >= target_area:
                    break
                
                zmax += delta_z
                if zmax > zmax_full:
                    raise ValueError(f"Cannot achieve target_area ({target_area}) within the redshift range of the data.")
            
            if verbose:
                print(f"[bin_z: {bin_z}, zmin: {zmin:.3f}, zmax: {zmax:.3f}, "
                      f"integral of n(z): {integral:.5f}, target: {target_area}]")
        
            return z_values
        elif self.nz_type == "thinbin":
            raise NotImplementedError(f"No need to implement for dataset datasets/{self.dataset}.")

class GetThetaLimits:
    def __init__(self, dataset, nz_flag, dynamical_theta_limits=False):
        """
        Initialize the GetThetaLimits class.

        Parameters:
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - nz_flag (str): Identifier for the n(z).
        - dynamical_theta_limits (bool): Whether to give dynamical theta ranges or not.
        """
        self.dataset = dataset
        self.nz_flag = nz_flag
        self.dynamical_theta_limits = dynamical_theta_limits
        self.theta_width = 4

        # Redshift distribution
        self.redshift_distributions = RedshiftDistributions(self.dataset, self.nz_flag, verbose=False)
        self.nbins = self.redshift_distributions.nbins

    def _angular_bao_scale_deg(self, z, cosmo):
        rd_mpc = cosmo.rs_drag / cosmo.h
        DA = cosmo.comoving_angular_distance(z)
        theta_rad = rd_mpc / ((1 + z) * DA)
        return np.degrees(theta_rad)

    def _get_dynamical_limits(self):
        from cosmoprimo.fiducial import DESI
        cosmo = DESI()
        theta_min, theta_max = {}, {}
        
        for bin_z in range(self.nbins):
            z_low, z_high = self.redshift_distributions.z_edges[bin_z]
            zeff = 0.5 * (z_low + z_high)
            theta_bao = self._angular_bao_scale_deg(zeff, cosmo) + (zeff - 0.4)
            theta_min[bin_z] = theta_bao - self.theta_width / 2
            theta_max[bin_z] = theta_bao + self.theta_width / 2
        
        return theta_min, theta_max

    def _get_constant_limits(self):
        if "DESY6" in self.dataset:
            theta_max_val = 5
        elif "DESIY1" in self.dataset:
            theta_max_val = 8
        else:
            raise ValueError(f"Static theta limits not defined for dataset '{self.dataset}'")

        theta_min = {bin_z: 0.5 for bin_z in range(self.nbins)}
        theta_max = {bin_z: theta_max_val for bin_z in range(self.nbins)}
        return theta_min, theta_max

    def get_theta_limits(self):
        if self.dynamical_theta_limits:
            return self._get_dynamical_limits()
        else:
            return self._get_constant_limits()

class WThetaDataCovariance:
    def __init__(self, dataset, weight_type, mock_id, nz_flag, cov_type, cosmology_covariance, delta_theta, 
                 theta_min, theta_max, bins_removed, diag_only, remove_crosscov):
        """
        Initialize the WThetaDataCovariance class.

        Parameters:
        - dataset (str): Dataset identifier (e.g., "DESY6").
        - weight_type (int): Weight type (for dataset "DESY6" it should be either 1 or 0).
        - mock_id (int): Mock id (for dataset "DESY6_COLA" it should go from 0 to 1951).
        - nz_flag (str): Identifier for the n(z).
        - cov_type (str): Type of covariance.
        - cosmology_covariance (str): Cosmology for the covariance.
        - delta_theta (float): Delta theta value.
        - theta_min (dict): Minimum theta value for each redshift bin.
        - theta_max (dict): Maximum theta value for each redshift bins.
        - bins_removed (list): Redshift bins removed when running the BAO fit.
        - diag_only (str): Whether to keep only the diagonal of the covariance.
        - remove_crosscov (str): Whether to remove the cross-covariances between redshift bins.
        """
        self.dataset = dataset
        self.weight_type = weight_type
        self.mock_id = mock_id
        self.nz_flag = nz_flag
        self.cov_type = cov_type
        self.cosmology_covariance = cosmology_covariance
        self.delta_theta = delta_theta
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.bins_removed = sorted(bins_removed)
        self.diag_only = diag_only
        self.remove_crosscov = remove_crosscov
        
        # Redshift distribution
        self.redshift_distributions = RedshiftDistributions(self.dataset, self.nz_flag, verbose=False)
        self.nbins = self.redshift_distributions.nbins

    def load_wtheta_data(self):
        indices_theta_allbins = {}
        theta_wtheta_data = {}
        wtheta_data = {}
        
        zip_file = f"datasets/{self.dataset}/wtheta/wtheta.zip"
        for bin_z in range(self.nbins):
            if self.dataset == "DESY6":
                file_in_zip = (f"wtheta_data_bin{bin_z}_DeltaTheta{self.delta_theta}_weights{self.weight_type}_fstar.txt")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    with zf.open(file_in_zip) as filename_wtheta:
                        theta, wtheta = np.loadtxt(filename_wtheta).T
                        
            elif self.dataset in ["DESY6_dec_below-23.5", "DESY6_dec_above-23.5", "DESY6_DR1tiles_noDESI", "DESY6_DR1tiles_DESIonly"]:
                file_in_zip = (f"wtheta_data_bin{bin_z}_DeltaTheta{self.delta_theta}_weights{self.weight_type}.txt")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    with zf.open(file_in_zip) as filename_wtheta:
                        theta, wtheta = np.loadtxt(filename_wtheta).T[:2]

            elif self.dataset in ["DESY6_COLA", "DESY6_COLA_dec_below-23.5", "DESY6_COLA_dec_above-23.5", "DESY6_COLA_DR1tiles_noDESI", "DESY6_COLA_DR1tiles_DESIonly"]:
                if self.mock_id == "mean":
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        # Find all mock files for the given redshift bin
                        pattern = re.compile(f"wtheta_mock[0-9]+_bin{bin_z}_DeltaTheta{self.delta_theta}.txt")
                        mock_files = [name for name in zf.namelist() if pattern.match(name)]
                        
                        if bin_z == 0:
                            self.n_mocks = len(mock_files)
                            print(f"Averaging the w(theta) over {self.n_mocks} mocks!")
                        
                        all_wtheta = []
                        theta = None

                        for mock_file in mock_files:
                            with zf.open(mock_file) as file:
                                theta_mock, wtheta_mock = np.loadtxt(file).T
                                if theta is None:
                                    theta = theta_mock
                                elif not np.array_equal(theta, theta_mock):
                                    raise ValueError("Theta arrays are inconsistent across mock files.")
                                all_wtheta.append(wtheta_mock)

                        wtheta = np.mean(all_wtheta, axis=0)

                else:
                    file_in_zip = f"wtheta_mock{self.mock_id}_bin{bin_z}_DeltaTheta{self.delta_theta}.txt"
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        with zf.open(file_in_zip) as filename_wtheta:
                            theta, wtheta = np.loadtxt(filename_wtheta).T
                            
            elif "DESIY1_LRG" in self.dataset:
                if self.mock_id == "mean":
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        # pattern = re.compile(r"deltaz0p02_deltath0p4_thetacuts/twoangcorr_mock_\d+\.npz")
                        pattern = re.compile(r"twoangcorr_mock_\d+\.npz")
                        mock_files = [name for name in zf.namelist() if pattern.match(name)]
            
                        if bin_z == 0:
                            self.n_mocks = len(mock_files)
                            print(f"Averaging the w(theta) over {self.n_mocks} mocks!")
            
                        all_wtheta = []
                        theta = None
            
                        for mock_file in mock_files:
                            with zf.open(mock_file) as file:
                                npz_data = np.load(file)
            
                                wtheta_mock = npz_data.get(f"z{bin_z}")
            
                                if theta is None:
                                    theta = npz_data.get("theta") * np.pi / 180
            
                                all_wtheta.append(wtheta_mock)
                                
                        wtheta = np.mean(all_wtheta, axis=0)
            
                else:
                    # file_in_zip = f"deltaz0p02_deltath0p4_thetacuts/twoangcorr_mock_{self.mock_id}.npz"
                    file_in_zip = f"twoangcorr_mock_{self.mock_id}.npz"
            
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        with zf.open(file_in_zip) as filename_wtheta:
                            npz_data = np.load(filename_wtheta)
            
                            wtheta = npz_data.get(f"z{bin_z}")
                            theta = npz_data.get("theta") * np.pi / 180

            indices_theta_individualbin = np.where(
                (theta > self.theta_min[bin_z] * np.pi / 180) &
                (theta < self.theta_max[bin_z] * np.pi / 180)
            )[0]

            theta_wtheta_data[bin_z] = theta[indices_theta_individualbin]
            
            if bin_z in self.bins_removed:
                wtheta_data[bin_z] = np.zeros(len(indices_theta_individualbin))
            else:
                wtheta_data[bin_z] = wtheta[indices_theta_individualbin]

        theta_wtheta_data_concatenated = np.concatenate([theta_wtheta_data[bin_z] for bin_z in range(self.nbins)])
        wtheta_data_concatenated = np.concatenate([wtheta_data[bin_z] for bin_z in range(self.nbins)])

        len_datavector = sum(len(theta_wtheta_data[bin_z]) for bin_z in range(self.nbins) if bin_z not in self.bins_removed)
        print(f"Length of data vector (calculated from the w(theta)): {len_datavector}")

        return theta_wtheta_data, wtheta_data

    def load_covariance_matrix(self):
        theta_cov = {}

        path_cov = f"datasets/{self.dataset}/cov_{self.cov_type}"

        if self.dataset in ["DESY6", "DESY6_dec_below-23.5", "DESY6_dec_above-23.5", "DESY6_DR1tiles_noDESI", "DESY6_DR1tiles_DESIonly"]:
            if self.cov_type == "cosmolike":
                for bin_z in range(self.nbins):
                    theta_cov[bin_z] = np.loadtxt(f"{path_cov}/delta_theta_{self.delta_theta}_binning.txt")[:, 2] * np.pi / 180 # same for all of them!
                cov = np.loadtxt(
                    f"{path_cov}/cov_Y6bao_data_DeltaTheta{str(self.delta_theta).replace('.', 'p')}_mask_g_{self.cosmology_covariance}.txt"
                )
            elif self.cov_type == "mocks":
                for bin_z in range(self.nbins):
                    theta_cov[bin_z] = np.loadtxt(f"{path_cov}/theta_DeltaTheta{self.delta_theta}.txt") # same for all of them!
                cov = np.loadtxt(
                    f"{path_cov}/cov_COLA_DeltaTheta{self.delta_theta}.txt"
                )
            else:
                raise NotImplementedError("Such covariance does not exist.")

        elif self.dataset in ["DESY6_COLA", "DESY6_COLA_dec_below-23.5", "DESY6_COLA_dec_above-23.5", "DESY6_COLA_DR1tiles_noDESI", "DESY6_COLA_DR1tiles_DESIonly"]:
            if self.cov_type == "cosmolike":
                if self.cosmology_covariance == "mice":
                    for bin_z in range(self.nbins):
                        theta_cov[bin_z] = np.loadtxt(f"{path_cov}/delta_theta_{self.delta_theta}_binning.txt")[:, 2] * np.pi / 180 # same for all of them!
                    cov = np.loadtxt(
                        f"{path_cov}/cov_Y6bao_cola_deltatheta{str(self.delta_theta).replace('.', 'p')}_mask_g_area2_biasv2.txt"
                    )
                else:
                    raise NotImplementedError("Such covariance does not exist.")
            elif self.cov_type == "mocks":
                for bin_z in range(self.nbins):
                    theta_cov[bin_z] = np.loadtxt(f"{path_cov}/theta_DeltaTheta{self.delta_theta}.txt") # same for all of them!
                cov = np.loadtxt(
                    f"{path_cov}/cov_COLA_DeltaTheta{self.delta_theta}.txt"
                )
            else:
                raise NotImplementedError("Such covariance does not exist.")

        if "DESIY1_LRG" in self.dataset:
            if self.cov_type == "mocks":
                for bin_z in range(self.nbins):
                    theta_cov[bin_z] = np.loadtxt(f"{path_cov}/theta.txt") * np.pi / 180
                cov = np.loadtxt(f"{path_cov}/EZcovariance_matrix.txt")
            else:
                raise NotImplementedError("Such covariance does not exist.")

        # cov_adjusted = np.zeros_like(cov)
        # for bin_z1 in range(self.nbins):
        #     for bin_z2 in range(self.nbins):
        #         slice_1 = slice(bin_z1 * len(theta_cov[bin_z1]), (bin_z1 + 1) * len(theta_cov[bin_z1]))
        #         slice_2 = slice(bin_z2 * len(theta_cov[bin_z2]), (bin_z2 + 1) * len(theta_cov[bin_z2]))
        #         if bin_z1 == bin_z2 or (bin_z1 not in self.bins_removed and bin_z2 not in self.bins_removed):
        #             cov_adjusted[slice_1, slice_2] = cov[slice_1, slice_2]
        # cov = cov_adjusted

        cov_adjusted = np.zeros_like(cov)
        for bin_z1 in range(self.nbins):
            len_1 = len(theta_cov[bin_z1])
            for bin_z2 in range(self.nbins):
                len_2 = len(theta_cov[bin_z2])
                slice_1 = slice(bin_z1 * len_1, (bin_z1 + 1) * len_1)
                slice_2 = slice(bin_z2 * len_2, (bin_z2 + 1) * len_2)
                if bin_z1 == bin_z2 or (bin_z1 not in self.bins_removed and bin_z2 not in self.bins_removed):
                    cov_adjusted[slice_1, slice_2] = cov[slice_1, slice_2]
        cov = cov_adjusted

        if self.remove_crosscov == "y":
            cov_adjusted = np.zeros_like(cov)
            for bin_z in range(self.nbins):
                slice_ = slice(bin_z * len(theta_cov[bin_z]), (bin_z + 1) * len(theta_cov[bin_z]))
                cov_adjusted[slice_, slice_] = cov[slice_, slice_]
            cov = cov_adjusted

        if self.diag_only == "y":
            cov = np.diag(np.diag(cov))

        # Let's do the scale cuts
        theta_cov_cut = {}
        indices_theta_allbins_concatenated = []
        for bin_z in range(self.nbins):

            indices_theta_individualbin = np.where(
                (theta_cov[bin_z] > self.theta_min[bin_z] * np.pi / 180) &
                (theta_cov[bin_z] < self.theta_max[bin_z] * np.pi / 180)
            )[0]

            theta_cov_cut[bin_z] = theta_cov[bin_z][indices_theta_individualbin]
            
            for bin_z2 in range(bin_z):
                indices_theta_individualbin += len(theta_cov[bin_z2])

            indices_theta_allbins_concatenated.append(indices_theta_individualbin)

        indices_theta_allbins_concatenated = np.concatenate(indices_theta_allbins_concatenated)
        
        cov_cut = cov[indices_theta_allbins_concatenated[:, None], indices_theta_allbins_concatenated]

        len_datavector = sum(len(theta_cov_cut[bin_z]) for bin_z in range(self.nbins) if bin_z not in self.bins_removed)
        print(f"Length of data vector (calculated from the covariance): {len_datavector}")
        
        if self.cov_type == "mocks":
            if "DESIY1_LRG" in self.dataset:
                hartlap = (1000 - len_datavector - 2) / (1000 - 1) # it's always 1000 since it's the number of EZ mocks
            elif "DESY6" in self.dataset:
                hartlap = (1952 - len_datavector - 2) / (1952 - 1)
            cov_cut /= hartlap
            print(f"Applying the Hartlap correction to the covariance matrix from the mocks (cov -> cov/{hartlap})")
            
            if "DESIY1_LRG_Abacus" in self.dataset: # only used for the Abacus
                if hasattr(self, 'n_mocks'): # if it exists then it means we have averaged the w(theta) over n_mocks and then we need to re-scale the covariance matrix
                    cov_cut /= self.n_mocks
                    print(f"Re-scaling the covariance matrix to fit the mean of the mocks (cov -> cov/{self.n_mocks})")
        
        return theta_cov_cut, cov_cut

    def process(self):
        theta_data, wtheta_data = self.load_wtheta_data()
        theta_cov, cov = self.load_covariance_matrix()
    
        if set(theta_data.keys()) != set(theta_cov.keys()):
            warnings.warn("Theta keys mismatch between data and covariance.")
            sys.exit("Aborting: theta_data and theta_cov keys differ!")
    
        for key in theta_data:
            if not np.allclose(theta_data[key], theta_cov[key], rtol=1e-4, atol=1e-4):
                warnings.warn(f"Theta mismatch in bin {key} (not close enough).")
                print(theta_data[key])
                print(theta_cov[key])
                sys.exit("Aborting: theta_data and theta_cov values differ!")

        return theta_data, wtheta_data, cov
    