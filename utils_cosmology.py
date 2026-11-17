from cosmoprimo import Cosmology
from cosmoprimo.fiducial import DESI

class CosmologicalParameters:
    def __init__(self, cosmology_template="planck", verbose=True):
        """
        Initialize the cosmological parameters for a given cosmology and return a Cosmology object.
        
        Parameters:
        - cosmology_template (str): Cosmology for the template.
        - verbose (bool): Whether to print messages.
        """
        if cosmology_template in ["planck", "planck_old"]:
            H_0 = 67.6  # Hubble constant in km/s/Mpc
            h = H_0 / 100

            Omega_b = 0.022 / h**2
            Omega_m = 0.31

            A_s = 2.02730058 * 10**-9
            n_s = 0.97
            sigma_8 = 0.8

            Omega_nu_massive = 0.000644 / h**2
            num_nu_massive = 1

            self.cosmo = Cosmology(
                h=h,
                Omega_cdm=Omega_m - Omega_b - Omega_nu_massive,
                Omega_b=Omega_b,
                sigma8=sigma_8,
                n_s=n_s,
                Omega_ncdm=Omega_nu_massive,
                engine="class"
            )

        elif cosmology_template in ["mice", "mice_old"]:
            H_0 = 70.0  # Hubble constant in km/s/Mpc
            h = H_0 / 100

            Omega_b = 0.044
            Omega_m = 0.25

            A_s = 2.445 * 10**-9
            n_s = 0.95
            sigma_8 = 0.8

            Omega_nu_massive = 0
            num_nu_massive = 0

            self.cosmo = Cosmology(
                h=h,
                Omega_cdm=Omega_m - Omega_b - Omega_nu_massive,
                Omega_b=Omega_b,
                sigma8=sigma_8,
                n_s=n_s,
                Omega_ncdm=Omega_nu_massive,
                engine="class"
            )

        elif cosmology_template == "desifid":
            self.cosmo = DESI()

        elif cosmology_template == "desifid_w0wa":
            "Best fit values for a flat w0waCDM cosmology from CMB + DESI BAO DR2 + DESY5 (https://arxiv.org/pdf/2503.14738)."

            H_0 = 66.73428704
            h = H_0 / 100
            
            Omega_b = 0.02221485621 / h**2
            Omega_m = 0.3191980194

            logA = 3.038847745
            n_s = 0.9644215278

            w0_fld = -0.7536302620
            wa_fld = -0.8574714585

            self.cosmo = Cosmology(
                h=h,
                Omega_m=Omega_m,
                Omega_b=Omega_b,
                logA=logA,
                n_s=n_s,
                w0_fld=w0_fld,
                wa_fld=wa_fld,
                engine="class"
            )

        else:
            raise ValueError("Cosmology not recognized. Please choose 'planck', 'mice', or 'desifid'.")

        if verbose:
            print(f"Initialized cosmology: {cosmology_template}.")

    def get_cosmology(self):
        """
        Return the initialized cosmoprimo.Cosmology object.
        """
        return self.cosmo
        