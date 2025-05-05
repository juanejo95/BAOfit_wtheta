import numpy as np
import sys
import zipfile
import re

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
        if self.dataset in ["DESY6", "DESY6_dec<-23.5", "DESY6_dec>-23.5"]: # they have the same n(z), but in different paths
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
        elif self.dataset == "DESY6_COLA":
            self.nz_type = "widebin"
            if self.nz_flag == "mocks":
                file_path = f"datasets/{self.dataset}/nz/nz_Y6COLA.txt"
                self.z_edges = {
                    0: [0.6, 0.7], 1: [0.7, 0.8], 2: [0.8, 0.9], 3: [0.9, 1.0], 4: [1.0, 1.1], 5: [1.1, 1.2]
                }
            else:
                raise ValueError(f"Unknown nz_flag: {self.nz_flag} for dataset: datasets/{self.dataset}")
        elif self.dataset in ["DESIY1_LRG_Abacus", "DESIY1_LRG_EZ"]:
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
            result[mask] = 1 / (z_max - z_min) # it is just a top hat function
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
        - theta_min (float): Minimum theta value.
        - theta_max (float): Maximum theta value.
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
        self.bins_removed = bins_removed
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
                        
            elif self.dataset in ["DESY6_noDESI_-23.5", "DESY6_DESI_-23.5"]:
                file_in_zip = (f"wtheta_data_bin{bin_z}_DeltaTheta{self.delta_theta}_weights{self.weight_type}.txt")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    with zf.open(file_in_zip) as filename_wtheta:
                        theta, wtheta = np.loadtxt(filename_wtheta).T[:2]

            elif self.dataset == "DESY6_COLA":
                if self.mock_id == "mean":
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        # Find all mock files for the given redshift bin
                        pattern = re.compile(f"wtheta_mock[0-9]+_bin{bin_z}_DeltaTheta{self.delta_theta}.txt")
                        mock_files = [name for name in zf.namelist() if pattern.match(name)]
                        
                        if bin_z == 0:
                            print(f"Averaging the w(theta) over {len(mock_files)} mocks!")
                        
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
                            
            elif self.dataset == "DESIY1_LRG_EZ":
                if self.mock_id == "mean":
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        # pattern = re.compile(r"deltaz0p02_deltath0p4_thetacuts/twoangcorr_mock_\d+\.npz")
                        pattern = re.compile(r"twoangcorr_mock_\d+\.npz")
                        mock_files = [name for name in zf.namelist() if pattern.match(name)]
            
                        if bin_z == 0:
                            print(f"Averaging the w(theta) over {len(mock_files)} mocks!")
            
                        all_wtheta = []
                        theta = None
            
                        for mock_file in mock_files:
                            with zf.open(mock_file) as file:
                                npz_data = np.load(file)
            
                                wtheta_mock = npz_data.get(f"z{bin_z+1}")
            
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
            
                            wtheta = npz_data.get(f"z{bin_z+1}")
                            theta = npz_data.get("theta") * np.pi / 180

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
        if self.cov_type == "cosmolike":
            path_cov = f"datasets/{self.dataset}/cov_{self.cov_type}"
            if self.dataset in ["DESY6", "DESY6_dec<-23.5", "DESY6_dec>-23.5"]:
                cov = np.loadtxt(
                    f"{path_cov}/cov_Y6bao_data_DeltaTheta{str(self.delta_theta).replace('.', 'p')}_mask_g_{self.cosmology_covariance}.txt"
                )
            elif self.dataset == "DESY6_COLA":
                if self.cosmology_covariance == "mice":
                    cov = np.loadtxt(
                        f"{path_cov}/cov_Y6bao_cola_deltatheta{str(self.delta_theta).replace('.', 'p')}_mask_g_area2_biasv2.txt"
                    )
                # if self.mock_id == "mean":
                #     cov /= 1952
            theta_cov = np.loadtxt(f"{path_cov}/delta_theta_{self.delta_theta}_binning.txt")[:, 2] * np.pi / 180
            if self.dataset == "DESY6_dec<-23.5":
                cov *= 1.456
            elif self.dataset == "DESY6_dec>-23.5":
                cov *= 3.193

        elif self.cov_type == "mocks":
            path_cov = f"datasets/{self.dataset}/cov_{self.cov_type}"
            if self.dataset == "DESIY1_LRG_EZ":
                cov = np.loadtxt(f"{path_cov}/EZcovariance_matrix.txt")
                theta_cov = np.loadtxt(f"{path_cov}/theta.txt") * np.pi / 180
        else:
            raise NotImplementedError("Such covariance does not exist.")

        theta_cov_concatenated = np.concatenate([theta_cov] * self.nbins)

        if abs(theta_wtheta_data_concatenated - theta_cov_concatenated[indices_theta_allbins_concatenated]).max() > 10**-4:
            print("The covariance matrix and the w(theta) do not have the same theta binning.")
            sys.exit()

        cov_adjusted = np.zeros_like(cov)
        for bin_z1 in range(self.nbins):
            for bin_z2 in range(self.nbins):
                slice_1 = slice(bin_z1 * len(theta_cov), (bin_z1 + 1) * len(theta_cov))
                slice_2 = slice(bin_z2 * len(theta_cov), (bin_z2 + 1) * len(theta_cov))
                if bin_z1 == bin_z2 or (bin_z1 not in self.bins_removed and bin_z2 not in self.bins_removed):
                    cov_adjusted[slice_1, slice_2] = cov[slice_1, slice_2]
        cov = cov_adjusted

        if self.remove_crosscov == "y":
            cov_adjusted = np.zeros_like(cov)
            for bin_z in range(self.nbins):
                slice_ = slice(bin_z * len(theta_cov), (bin_z + 1) * len(theta_cov))
                cov_adjusted[slice_, slice_] = cov[slice_, slice_]
            cov = cov_adjusted

        if self.diag_only == "y":
            cov = np.diag(np.diag(cov))

        cov_orig = np.copy(cov)
        cov = cov_orig[indices_theta_allbins_concatenated[:, None], indices_theta_allbins_concatenated]

        # if self.cov_type == "mocks":
        #     if self.dataset == "DESIY1_LRG_EZ":
        #         print("Applying the Hartlap correction to the covariance matrix from the mocks")
        #         hartlap = (1000 - len(cov) - 2) / (1000 - 1)
        #         cov /= hartlap
        
        return cov

    def process(self):
        theta_data, wtheta_data, indices_theta_allbins_concatenated, theta_wtheta_data_concatenated = self.load_wtheta_data()
        cov = self.load_covariance_matrix(indices_theta_allbins_concatenated, theta_wtheta_data_concatenated)
        return theta_data, wtheta_data, cov
    