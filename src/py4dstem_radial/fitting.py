"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import numpy as np
from .profiles import radial_profile
from scipy.optimize import curve_fit
def simulate_radial_profile_1D(
    q_centers,
    a,
    hkllist,
    k_broadening,
    int_scale,
    b0,
    b1,
    b2
):
    """
        Simulate a one-dimensional radial diffraction intensity profile for a cubic
        crystal using Gaussian-broadened reflections and a quadratic background.

        Parameters:
        q_centers: np.ndarray
            One-dimensional array of reciprocal-space positions at which to
            evaluate the simulated intensity, typically in Å^-1.
        a: float
            Cubic lattice parameter in Å. The predicted reflection positions are
            calculated as sqrt(hkllist) / a.
        hkllist: array-like
            Values of h^2 + k^2 + l^2 for the reflections to simulate.
        k_broadening: float
            Gaussian broadening parameter in reciprocal-space units. Must be
            nonzero for the Gaussian calculation.
        int_scale: float or array-like
            Intensity scale. A single value is shared by all reflections; otherwise,
            values are paired with reflections in hkllist.
        b0: float
            Constant term of the polynomial background.
        b1: float
            Linear coefficient of the polynomial background.
        b2: float
            Quadratic coefficient of the polynomial background.

        Function purpose:
            Builds a simulated radial diffraction profile by summing Gaussian
            peaks at the theoretical reflection positions and adding a quadratic
            polynomial background.

        Returns: np.ndarray
            Simulated intensity values with the same shape as q_centers.
        """
    hkllist = np.atleast_1d(np.asarray(hkllist, dtype=float))
    int_scale = np.atleast_1d(np.asarray(int_scale, dtype=float))
    theoq = np.sqrt(hkllist) / a
    I_sim = np.zeros_like(q_centers, dtype=float)
    if int_scale.size == 1:
        for g in zip(theoq):
            I_sim += int_scale.item() * np.exp(-(q_centers - g)**2 / (2 * k_broadening**2))
    else:
        for g, I0 in zip(theoq, int_scale):
            I_sim += I0 * np.exp(-(q_centers - g)**2 / (2 * k_broadening**2))    
    # Polynomial background
    I_sim += b0 + b1 * q_centers + b2 * q_centers**2
    return I_sim

def fit_lattice_parameter_1D(
    q_exp,
    I_exp,
    hkllist,
    latpar,
    k_broadening=0.02,
    fit_all_intensities=True,
    constrain_range=False,
    a_ref_fixed=4.0782,
    a_bounds=(0.5, 1.5),
):
    """
       Fit a cubic lattice parameter and other profile parameters to an
       experimental one-dimensional radial diffraction profile.

       Parameters:
       q_exp: np.ndarray
           One-dimensional experimental reciprocal-space axis, typically in Å^-1.
       I_exp: np.ndarray
           Experimental intensity values corresponding to q_exp.
       hkllist: array-like
           Values of h^2 + k^2 + l^2 defining the theoretical reflection positions.
       latpar: float
           Initial guess for the cubic lattice parameter, in Å.
       k_broadening: float, optional
           Initial Gaussian broadening parameter. Defaults to 0.02.
       fit_all_intensities: bool, optional
           If True, fit an independent intensity scale for each reflection. If
           False, use one shared intensity scale for all reflections.
       constrain_range: bool, optional
           If True, constrain the lattice parameter to ±5% of a_ref_fixed.
           If False, use a_bounds as fractional bounds relative to latpar.
       a_ref_fixed: float, optional
           Reference lattice parameter, in Å, used to set the ±5% fitting range
           when constrain_range is True. Defaults to 4.0782.
       a_bounds: tuple[float, float], optional
           Fractional lower and upper bounds relative to latpar when
           constrain_range is False. Defaults to (0.5, 1.5).

       Function purpose:
           Uses scipy.optimize.curve_fit to fit the simulated radial profile to
           the experimental data. The fitted parameters include the lattice
           parameter, Gaussian broadening, a quadratic background, and either
           individual or shared reflection intensity scales. The fit is weighted
           using a sigma array derived from I_exp.

       Returns: tuple[np.ndarray, np.ndarray]
           popt: Optimized parameter array. Its entries begin with the lattice
               parameter, broadening, and background coefficients, followed by
               fitted intensity scale(s).
           pcov: Covariance matrix estimated by curve_fit for the fitted parameters.
       """
    latpar = float(np.asarray(latpar).reshape(-1)[0])
    k_broadening = float(np.asarray(k_broadening).reshape(-1)[0])
    a_ref_fixed = float(np.asarray(a_ref_fixed).reshape(-1)[0])
    hkllist = np.atleast_1d(np.asarray(hkllist, dtype=float))
    if constrain_range == True:
        a_bounds = a_ref_fixed*0.95,a_ref_fixed*1.05
        n_peaks = len(hkllist)
        sigma=1/(I_exp + 0.3*np.max(I_exp))
        if fit_all_intensities == True:
            def fitfun(q, a, k_broadening, b0, b1, b2, *int_scale):
                return simulate_radial_profile_1D(
                    q, a, hkllist, k_broadening, np.array(int_scale), b0, b1, b2)
            p0 = [latpar, k_broadening, 0.0, 0.0, 0.0] + [1.0] * n_peaks
            lower_bounds = [a_bounds[0], 0.0, -np.inf, -np.inf, -np.inf] + [0.0] * n_peaks
            upper_bounds = [a_bounds[1], 0.1, np.inf, np.inf, np.inf] + [np.inf] * n_peaks
        else:
            def fitfun(q, a, k_broadening, b0,b1,b2, int_scale):
                return simulate_radial_profile_1D(
                    q, a, hkllist, k_broadening, np.full(n_peaks, int_scale), b0, b1, b2)
            p0 = [latpar, k_broadening, 0.0, 0.0, 0.0, 1.0]
            lower_bounds = [a_bounds[0], 0.00001, -np.inf, -np.inf, -np.inf, 0.0]
            upper_bounds = [a_bounds[1], 0.1, np.inf, np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(fitfun,q_exp,I_exp,p0=p0,bounds=(lower_bounds, upper_bounds),sigma=sigma,absolute_sigma=False)
        return popt, pcov
    else:
        n_peaks = len(hkllist)
        sigma=1/(I_exp + 0.3*np.max(I_exp))
        if fit_all_intensities == True:
            def fitfun(q, a, k_broadening,b0,b1,b2, *int_scale):
                return simulate_radial_profile_1D(q,a,hkllist,k_broadening,np.array(int_scale), b0, b1, b2)
            p0 = [latpar, k_broadening, 0.0, 0.0, 0.0] + [1.0] * n_peaks
            lower_bounds = [a_bounds[0], 0.00001, -np.inf, -np.inf, -np.inf] + [0.0] * n_peaks
            upper_bounds = [a_bounds[1], 0.1, np.inf, np.inf, np.inf] + [np.inf] * n_peaks
        else:
            def fitfun(q, a, k_broadening,b0,b1,b2, int_scale):
                return simulate_radial_profile_1D(q,a,hkllist,k_broadening,np.full(n_peaks, int_scale), b0, b1, b2)
            p0 = [latpar, k_broadening, 0.0, 0.0, 0.0, 1.0]
            lower_bounds = [a_bounds[0], 0.00001, -np.inf, -np.inf, -np.inf, 0.0]
            upper_bounds = [a_bounds[1], 0.1, np.inf, np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(fitfun,q_exp,I_exp,p0=p0,bounds=(lower_bounds, upper_bounds),sigma=sigma,absolute_sigma=False)
        return popt, pcov
def estimate_a_for_pixel_from_bvm(
        bvm_pixel:np.ndarray, #This array should be the shape of the Bragg Vector Map for the pixel of interest 
        q_r:np.ndarray,  #This array represents the distance to the origin in A^-1 or |q| and is of the same size as bvm_pixel. 
        hkllist:list[int],
        npeaks:int=5,
        q_range:tuple[float,float]=(0.2, 1.2),
        a_bounds:tuple[float,float]=(3.615, 4.0782),
        return_details:bool=False,
        num_bins:int=150,
        k_broadening:float=0.005,
        a_ref_fixed=None,
        latpar:float=4.0782,
        )->tuple[float,float]:
    """
        Estimate the cubic lattice parameter from the radial intensity profile of
        a single pixel's Bragg Vector Map.

        Parameters:
        bvm_pixel: np.ndarray, shape (Q_y, Q_x)
            Two-dimensional Bragg Vector Map containing intensities for one scan
            position.
        q_r: np.ndarray, shape matching bvm_pixel
            Reciprocal-space distance |q| for each Bragg Vector Map pixel, typically
            in Å^-1.
        hkllist: list[int] or array-like
            Values of h^2 + k^2 + l^2 for the reflections included in the profile fit.
        npeaks: int, optional
            Intended maximum number of peaks to use. The current implementation
            does not use this parameter in the fitting calculation. Defaults to 5.
        q_range: tuple[float, float], optional
            Lower and upper reciprocal-space limits used to calculate the radial
            profile. Defaults to (0.2, 1.2).
        a_bounds: tuple[float, float], optional
            Bounds passed to the fitting routine for the lattice parameter. Defaults
            to (3.615, 4.0782); the fitting routine interprets these as fractional
            bounds relative to latpar when constrain_range is False. In this
            function, constrain_range is True, so the effective fit bounds are
            instead set to ±5% of a_ref_fixed.
        return_details: bool, optional
            If True, also return a dictionary containing the radial profile and fit
            diagnostics. Defaults to False.
        num_bins: int, optional
            Number of radial bins used to construct the profile. Defaults to 150.
        k_broadening: float, optional
            Broadening parameter passed to radial_profile. Defaults to 0.005.
        a_ref_fixed: float or None, optional
            Reference lattice parameter passed to the fitting routine. The current
            implementation passes this value as the initial lattice parameter and
            as the fixed reference for constrained fitting. A numeric value is
            therefore expected for the fit to run successfully.
        latpar: float, optional
            Lattice parameter used to normalize the reported uncertainty percentage.
            Defaults to 4.0782.

        Function purpose:
            Calculates a radial profile from the selected pixel's Bragg Vector Map,
            fits a cubic lattice parameter, and estimates a percentage uncertainty
            from the fitted covariance matrix.

        Returns: tuple
            If return_details is False, returns (best_a, sigma_a_percent), where
            best_a is the fitted lattice parameter in Å and sigma_a_percent is
            twice the square root of the fitted lattice-parameter variance,
            expressed as a percentage of latpar. If return_details is True, returns
            (best_a, sigma_a_percent, details), where details contains q_centers,
            I_profile, popt, pcov, hkllist, and latpar.
        """
    #First, we compute the radial profile of the Bragg Vector Map of the pixel.
    q_min, q_max = q_range
    q_centers, I_profile = radial_profile(bvm_pixel, q_r, q_range=q_range, num_bins=num_bins,k_broadening=k_broadening)
    popt, pcov = fit_lattice_parameter_1D(
        q_centers,
        I_profile,
        hkllist,
        latpar=a_ref_fixed,
        fit_all_intensities=True,
        constrain_range=True,
        a_bounds=a_bounds,
        a_ref_fixed=a_ref_fixed,
    )
    best_a = popt[0]
    k_broadening = popt[1]
    # int_scales = popt[2:]
    sigma_a=np.sqrt(pcov[0,0])*2
    sigma_a_percent=sigma_a/latpar*100
    if return_details:
        return best_a,sigma_a_percent,{
            'q_centers': q_centers,
            'I_profile': I_profile,
            'popt': popt,
            'pcov': pcov,
            'hkllist': hkllist,
            'latpar': latpar,
        }
    else:
        return best_a,sigma_a_percent
def calculate_lattice_parameter_maps(
    per_pixel_bvms,
    q_r,
    hkllist,
    npeaks,
    q_range,
    a_bounds,
    k_broadening,
    a_ref_fixed,
):
    """
        Calculate a lattice-parameter map and an associated uncertainty map from
        Bragg Vector Maps for each position in a scan.

        Parameters:
        per_pixel_bvms: np.ndarray, shape (R0, R1, Q_y, Q_x)
            Bragg Vector Maps, with one two-dimensional map for each scan position.
        q_r: np.ndarray, shape (Q_y, Q_x)
            Reciprocal-space distance |q| for each Bragg Vector Map pixel.
        hkllist: array-like
            Values of h^2 + k^2 + l^2 defining the reflections used in fitting.
        npeaks: int
            Maximum number of peaks intended for the fit. Passed to
            estimate_a_for_pixel_from_bvm; that function currently does not use it.
        q_range: tuple[float, float]
            Reciprocal-space interval used to construct each radial profile.
        a_bounds: tuple[float, float]
            Lattice-parameter bounds passed to the per-pixel fitting function.
        k_broadening: float
            Broadening parameter used when calculating radial profiles.
        a_ref_fixed: float
            Reference lattice parameter used for the constrained fit.

        Function purpose:
            Iterates over scan positions with nonzero Bragg Vector Map intensity,
            estimates the lattice parameter for each selected pixel, and records
            failed fits as NaN. The nonzero mask identifies pixels whose maps have
            a positive total intensity.

        Returns: tuple[np.ndarray, np.ndarray, np.ndarray]
            a_map: Two-dimensional float32 array of fitted lattice parameters, with
                NaN for empty pixels or fits that raise RuntimeError or ValueError.
            strain_std_map: Two-dimensional float32 array containing the reported
                percentage uncertainty for each fitted lattice parameter, with NaN
                where no result was recorded.
            nonzero_mask: Two-dimensional boolean array indicating which input
                Bragg Vector Maps have a sum greater than zero.
    """    
    R0, R1 = per_pixel_bvms.shape[:2]

    nonzero_mask = (
        per_pixel_bvms.sum(axis=(2, 3)) > 0
    )
    fit_mask = (
        nonzero_mask
    )
    a_map = np.full(
        (R0, R1),
        np.nan,
        dtype=np.float32,
    )

    strain_std_map = np.full(
        (R0, R1),
        np.nan,
        dtype=np.float32,
    )

    for i in range(R0):
        for j in range(R1):
            if not fit_mask[i, j]:
                continue

            try:
                est_a, sig_a = estimate_a_for_pixel_from_bvm(
                bvm_pixel=per_pixel_bvms[i, j],
                q_r=q_r,
                hkllist=hkllist,
                npeaks=npeaks,
                q_range=q_range,
                a_bounds=a_bounds,
                return_details=False,
                k_broadening=k_broadening,
                a_ref_fixed=a_ref_fixed,
            )

                a_map[i, j] = est_a
                strain_std_map[i, j] = sig_a

            except (RuntimeError, ValueError):
                a_map[i, j] = np.nan
                strain_std_map[i, j] = np.nan

    return a_map, strain_std_map, nonzero_mask