class path_template:
    def __init__(self, include_wiggles, nz_flag, cosmology_template, verbose=True):
        """
        Initialize the path_template class.

        Args:
            include_wiggles (str): Indicates whether wiggles are included (e.g., "yes" or "no").
            nz_flag (str): The flag used for the n(z) configuration.
            cosmology_template (str): The cosmology template identifier.
            verbose (bool): Whether to print the save directory message. Default is True.
        """
        self.include_wiggles = include_wiggles
        self.nz_flag = nz_flag
        self.cosmology_template = cosmology_template
        self.verbose = verbose

    def __call__(self):
        """
        Print a message with the save directory path when the instance is called, 
        if verbose is True.
        """
        path_string = f'wtheta_template{self.include_wiggles}/nz_{self.nz_flag}/wtheta_{self.cosmology_template}'
        if self.verbose:
            print(f"Saving output to: {path_string}")
        return path_string

class path_baofit:
    def __init__(self, include_wiggles, dataset, weight_type, nz_flag, cov_type, cosmology_template, 
                 cosmology_covariance, delta_theta, theta_min, theta_max, n_broadband, bins_removed, verbose=True):
        """
        Initialize the path_baofit class to generate a save directory path based on various parameters.

        Args:
            include_wiggles (str): Indicates whether wiggles are included (e.g., "yes" or "no").
            dataset (str): The dataset identifier (e.g., "COLA" or others).
            weight_type (str): The weight type (e.g., "unweighted", "weighted").
            nz_flag (str): The flag used for the n(z) configuration.
            cov_type (str): The type of covariance.
            cosmology_template (str): The cosmology template identifier.
            cosmology_covariance (str): The cosmology covariance.
            delta_theta (float): The delta theta value.
            theta_min (float): The minimum theta value.
            theta_max (float): The maximum theta value.
            n_broadband (int): The number of broadband bins.
            bins_removed (int): The number of bins removed.
            verbose (bool): Whether to print the save directory message. Default is True.
        """
        self.include_wiggles = include_wiggles
        self.dataset = dataset
        self.weight_type = weight_type
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

    def __call__(self):
        """
        Generate the save directory path and print the message when the instance is called, 
        if verbose is True.
        """
        if self.dataset == 'COLA':
            path_string = (
                f"fit_results{self.include_wiggles}/{self.dataset}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}"
            )
        elif self.dataset == 'DESY6':
            path_string = (
                f"fit_results{self.include_wiggles}/{self.dataset}_{self.weight_type}/nz{self.nz_flag}_cov{self.cov_type}_"
                f"{self.cosmology_template}temp_{self.cosmology_covariance}cov_deltatheta{self.delta_theta}_"
                f"thetamin{self.theta_min}_thetamax{self.theta_max}_{self.n_broadband}broadband_binsremoved{self.bins_removed}"
            )
        
        # Print if verbose is True
        if self.verbose:
            print(f"Saving output to: {path_string}")
        
        return path_string