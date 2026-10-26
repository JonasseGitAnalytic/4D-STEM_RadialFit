"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import py4DSTEM
import numpy as np
from py4DSTEM import show
from scipy.optimize import curve_fit


def calibrate_pixel_size_with_uncertainty(
    crystal,
    bragg_peaks,
    scale_pixel_size=1.0,
    bragg_k_power=1.0,
    bragg_intensity_power=1.0,
    k_min=0.0,
    k_max=None,
    k_step=0.002,
    k_broadening=0.002,
    fit_all_intensities=False,
    set_calibration_in_place=False,
    verbose=True,
    plot_result=False,
    figsize=(12, 6),
    returnfig=False,
):
    """
    Calibrate reciprocal-space pixel size using crystal structure factors. Modelled after
    py4DSTEM's Crystal.calibrate_pixel_size() function, with the addition of the 1-sigma uncertainty 
    obtained from the covariance matrix returned by scipy.optimize.curve_fit()

    Parameters: 
        
    crystal: crystal object 
        Crystal object containing structure factors corresponding to the measured lattice.
    bragg_peaks: tuple(int,int)
        The shape in pixels of the BraggVectors object. 
    scale_pixel_size: float
        Scale factor with which to change pixel size. 
    bragg_k_power: float
        Multiplies intensities by k^(bragg_k_power).
    bragg_intensity_power: float
        Changes intensities according to I^(bragg_intensity_power).
    k_min, kmax: float
        Minimum- and maximum of the used fitting range in reciprocal space units.
    k_step: float
        Step size of fitting model in reciprocal space units.
    k_broadening: float
        1 sigma broadening of input crystal structure factors. 
    fit_all_intensities: bool
        Whether to use one intensity scaling for all reflections or allow each peak to scale individually.
    set_calibration_in_place: bool
        Whether to change the calibration of the input bragg_peaks in place.
    verbose: bool
        Whether to print fitted pixel size and associated uncertainty. 
    plot_result: bool
        Whether to plot the result.
    figsize: tuple (int,int)
        The size of the figure.
    returnfig: bool
        Whether to return the figure.
        
    Function purpose: This function fits a reference lattice to a measured set of 
    Bragg peaks and yields the best-fit lattice parameter and corresponding uncertainty.
    
    Returns: 
    bragg_peaks_cali
        Calibrated copy of the BraggVectors object.

    pixel_size_new : float
        Calibrated Q pixel size in Å^-1.

    pixel_size_sigma : float
        1-sigma uncertainty of the calibrated Q pixel size in Å^-1.

    fit_scale : float
        Fitted scale_pixel_size parameter.

    fit_scale_sigma : float
        1-sigma uncertainty of fit_scale.

    fig, ax : optional
        Returned only when returnfig=True and plot_result=True.
    """

    assert hasattr(crystal, "struct_factors"), (
        "Compute structure factors first..."
    )

    # Prepare experimental data
    k, int_exp = crystal.calculate_bragg_peak_histogram(
        bragg_peaks,
        bragg_k_power,
        bragg_intensity_power,
        k_min,
        k_max,
        k_step,
    )
    calc_1D_profile = crystal.calibrate_pixel_size.__globals__["calc_1D_profile"]
    
    # Fit function
    def fit_profile(k, *coefs):
        scale_pixel_size = coefs[0]
        k_broadening = coefs[1]
        int_scale = coefs[2:]

        int_sf = calc_1D_profile(
            k,
            crystal.g_vec_leng * scale_pixel_size,
            crystal.struct_factors_int,
            k_broadening=k_broadening,
            int_scale=int_scale,
            normalize_intensity=False,
        )

        return int_sf

    # Initial parameters
    if fit_all_intensities:
        coefs = (
            scale_pixel_size,
            k_broadening,
            *tuple(np.ones(crystal.g_vec_leng.shape[0])),
        )
    else:
        coefs = (
            scale_pixel_size,
            k_broadening,
            1.0,
        )

    bounds = (0.0, np.inf)

    # Fit
    popt, pcov = curve_fit(
        fit_profile,
        k,
        int_exp,
        p0=coefs,
        bounds=bounds,
    )

    # Fitted parameters
    scale_pixel_size = popt[0]
    k_broadening = popt[1]
    int_scale = np.array(popt[2:])

    # 1-sigma uncertainty of fitted scale parameter
    scale_pixel_size_sigma = np.sqrt(pcov[0, 0])

    # Get calibrated pixel size
    pix_size_prev = bragg_peaks.calibration.get_Q_pixel_size()

    pixel_size_new = pix_size_prev / scale_pixel_size

    # Propagate uncertainty:
    pixel_size_sigma = (
        pixel_size_new
        * scale_pixel_size_sigma
        / scale_pixel_size
    )

    # Apply calibration in place if requested
    if set_calibration_in_place:
        bragg_peaks.calibration.set_Q_pixel_size(pixel_size_new)
        bragg_peaks.calibration.set_Q_pixel_units("A^-1")

    # Calibrated Bragg peaks
    bragg_peaks_cali = bragg_peaks.copy()
    bragg_peaks_cali.calibration.set_Q_pixel_size(pixel_size_new)
    bragg_peaks_cali.calibration.set_Q_pixel_units("A^-1")

    if verbose:
        print(
            f"Calibrated pixel size = "
            f"{pixel_size_new:.8f} ± {pixel_size_sigma:.8f} A^-1 "
            f"(1σ)"
        )

    # Plotting
    if plot_result:
        if int_scale.shape[0] < crystal.g_vec_leng.shape[0]:
            int_scale = np.hstack(
                (
                    int_scale,
                    np.ones(
                        crystal.g_vec_leng.shape[0] - int_scale.shape[0]
                    ),
                )
            )
        elif int_scale.shape[0] > crystal.g_vec_leng.shape[0]:
            int_scale = int_scale[: crystal.g_vec_leng.shape[0]]

        if returnfig:
            fig, ax = crystal.plot_scattering_intensity(
                bragg_peaks=bragg_peaks_cali,
                figsize=figsize,
                k_broadening=k_broadening,
                int_power_scale=1.0,
                int_scale=int_scale,
                bragg_k_power=bragg_k_power,
                bragg_intensity_power=bragg_intensity_power,
                k_min=k_min,
                k_max=k_max,
                returnfig=True,
            )
        else:
            crystal.plot_scattering_intensity(
                bragg_peaks=bragg_peaks_cali,
                figsize=figsize,
                k_broadening=k_broadening,
                int_power_scale=1.0,
                int_scale=int_scale,
                bragg_k_power=bragg_k_power,
                bragg_intensity_power=bragg_intensity_power,
                k_min=k_min,
                k_max=k_max,
            )

    # Return
    if returnfig and plot_result:
        return (
            bragg_peaks_cali,
            pixel_size_new,
            pixel_size_sigma,
            scale_pixel_size,
            scale_pixel_size_sigma,
            (fig, ax),
        )

    return (
        bragg_peaks_cali,
        pixel_size_new,
        pixel_size_sigma,
        scale_pixel_size,
        scale_pixel_size_sigma,
    )

def fit_origin(
    braggpeaks,
    probe,
    sampling,
    fitfunction="parabola",
    robust=True,
    show_result=False,
):
    """
    Determines the exact sub-pixel center of the undiffracted electron beam
    for each diffraction pattern. 
    
    Parameters:
        
    braggpeaks : py4DSTEM BraggVectors
        Bragg peak dataset with a fitted origin.
    probe: py4DSTEM probe
        Vacuum probe template from py4DSTEM.
    sampling: integer
        Whether to up- or downsample data in the histogram calculations.
    fitfunction: str
        The function used to fit to measured center beam positions.
    robust: bool
        Whether to iteratively remove outliers from the fit function.
    show_result: bool
        Whether to show the resulting fit.
    
    Function purpose: This function measures the center of all undiffracted electron 
    beam positions for each pixel and subsequently fits a function to those postions.
    
    Returns: tuple (qx0_fit, qy0_fit, qx0_residuals, qy0_residuals)
        Contains fit center beam positions and their residuals. 
        
    """

    # Determine the vacuum-probe center in the Bragg-vector coordinate system
    _, qx0_pr, qy0_pr = probe

    # Measure origin at every scan position using the same fixed center guess
    qx0_meas, qy0_meas, mask_meas = braggpeaks.measure_origin(
        center_guess=(qx0_pr, qy0_pr),
        score_method="distance",
    )

    if show_result:
        show(
            qx0_meas,
            title="Measured qx0",
        )
        show(
            qy0_meas,
            title="Measured qy0",
        )

    # Fit the spatially varying origin
    qx0_fit, qy0_fit, qx0_residuals, qy0_residuals = braggpeaks.fit_origin(
        fitfunction=fitfunction,
        robust=robust,
    )

    if show_result:
        bvm_o = braggpeaks.histogram(
            mode="cal",
            sampling=sampling,
        )
        show(
            bvm_o,
            circle={
            'center' : bvm_o.origin,
            'R' : 4*sampling,
            'fill' : False,
            'linewidth' : 1
            },
            vmax=0.999
            )
    return qx0_fit, qy0_fit, qx0_residuals, qy0_residuals


def fit_elliptical_correction(
    braggpeaks,
    qint=10,
    q_ell_range=15,
    sampling=1,
    binning=1,
    p0=None,
    mask=None,
    show_result=False,
    vmax=0.999,
):
    """
    Fit elliptical distortion from an amorphous diffraction ring according to 
    py4DSTEM's process.calibration.fit_ellipse_amorphous_ring() function. 

    Parameters:
   
    braggpeaks : py4DSTEM BraggVectors
        Bragg peak dataset with a fitted origin.
    qint : int
        Radial distance from center beam to use for ellipticity fitting.
    q_ell_range : int
        Radial range used for the elliptical-ring fit in pixels.
    sampling : int
        Sampling used when generating the calibrated BVM.
    binning: int
        The binning value used for the diffraction patterns. 
    p0 : array-like or None
        Initial ellipse parameters.
    mask : array-like or None
        Optional mask for the ellipse fit.
    show_result : bool
        Whether to display the elliptical fit.
    vmax : float
        Maximum display value for the diagnostic plot.
        
    Function purpose: This function fits an ellipse to the amorphous background
    of the BVM to yield to value for an elliptical distortion.

    Returns: p_ellipse
        Fitted ellipse parameters.
    """
    q_int_range = (
        round(qint / sampling),
        round((qint + q_ell_range) / sampling),)
    bvm_c = braggpeaks.histogram(
        mode="cal",
        sampling=sampling,
    )

    p_ellipse = py4DSTEM.process.calibration.fit_ellipse_amorphous_ring(
        data=bvm_c.data,
        center=bvm_c.origin,
        fitradii=q_int_range,
        p0=p0,
        mask=mask,
    )

    if show_result:
        py4DSTEM.visualize.show_elliptical_fit(
            bvm_c,
            q_int_range,
            p_ellipse[0],
            cmap="gray",
            vmax=vmax,
        )

    braggpeaks.calibration.set_p_ellipse(
        p_ellipse[0]
    )

    braggpeaks.setcal()

    return p_ellipse[0]


def calibrate_braggpeaks(
    braggpeaks,
    probe,
    qint,
    q_ell_range,
    binning,
    sampling,
    pixelsize=None,
    overwrite=False,
    show_result=False,
):
    """
    Calls fit_origin, fit_ellipse and calibrate_pixel_size_with_uncertainty based
    on whether overwrite is True or no existing calibration can be found.

    Parameters
    ----------
    braggpeaks : py4DSTEM BraggVectors
        Bragg peak dataset.
    probe : py4DSTEM probe
        Probe object of py4DSTEM
    qint : int
        Radial distance from center beam to use for ellipticity fitting.
    q_ell_range : int
        Radial range used for the elliptical-ring fit in pixels.
    binning : int
        Diffraction-pattern binning.
    sampling : int
      Sampling used when generating the calibrated BVM.
    pixelsize : float or None
        Initial reciprocal-space pixel size.
    overwrite : bool
        If True, redo the calibration.
    show_result : bool
        Prints calibration diagnostics and final calibration information.
        
    Function purpose: This function acts like a wrapper to call origin, ellipse,
    and pixel size calibrations.

    Returns: py4DSTEM BraggVectors
        Calibrated BraggVectors object.
    """

    #################################################################################################
    ### # ----- Check existing calibrations -----
    #################################################################################################

    existing_origin = braggpeaks.calibration.get_origin()
    existing_ellipse = braggpeaks.calibration.get_ellipse()
    existing_pixel_size = braggpeaks.calibration.get_Q_pixel_size()

    has_origin = existing_origin is not None
    has_ellipse = existing_ellipse is not None
    has_pixel_size = existing_pixel_size is not None

    has_calibration = (
        has_origin
        and has_ellipse
        and has_pixel_size
    )

    #################################################################################################
    ### # ----- Use existing calibrations -----
    #################################################################################################

    if has_calibration and not overwrite:

        braggpeaks.setcal()

        if show_result:
            print(
                "Pixel size =",
                braggpeaks.calibration.get_Q_pixel_size(),
                "A^-1",
            )
            print(
                "Calibration state:",
                braggpeaks.calstate,
            )

        return braggpeaks

    #################################################################################################
    ### # ----- Fit origin -----
    #################################################################################################

    fit_origin(
        braggpeaks,
        probe=probe,
        show_result=show_result,
        sampling=sampling,
    )

    #################################################################################################
    ### # ----- Fit ellipticity -----
    #################################################################################################

    fit_elliptical_correction(
        braggpeaks,
        qint=qint,
        q_ell_range=q_ell_range,
        sampling=sampling,
        show_result=show_result,
    )

    #################################################################################################
    ### # ----- Set pixel size or fit pixel size -----
    #################################################################################################

    if pixelsize is not None:

        pixel_size_inv_Ang = pixelsize * binning

        braggpeaks.calibration.set_Q_pixel_size(
            pixel_size_inv_Ang
        )

        braggpeaks.calibration.set_Q_pixel_units(
            "A^-1"
        )
    else:
        q_range=(0.4,1.2)
        from py4dstem_radial.visualization import reference_crystal_diagnostic
        crystal, fig, ax = reference_crystal_diagnostic(braggpeaks=braggpeaks,lattice_parameter=4.0,atom_num=78,q_range=(0.4,1.2))
        (braggpeaks_cali,Q_pixel_size,Q_pixel_size_sigma,scale,scale_sigma,(figCalCryst, axCalCryst),) = calibrate_pixel_size_with_uncertainty(
            crystal,
            bragg_peaks=braggpeaks,
            bragg_k_power=1.0,
            plot_result=True,
            k_broadening=0.02,
            fit_all_intensities=False,
            verbose=True,
            set_calibration_in_place=True,
            k_min=q_range[0],
            k_max=q_range[1],
            k_step=0.001,
            returnfig=True,
        )
        # print(axCalCryst)
        axCalCryst.set_xlim(q_range[0],q_range[1])
        print("Pixel size =",braggpeaks.calibration.get_Q_pixel_size(),"±",Q_pixel_size_sigma,"A^-1",)
        axCalCryst.set_xlim(q_range[0],q_range[1])
        figCalCryst.set_size_inches(8,5)
        # axScat.set_title(f'Reference scattering with latpar = {latpar}')
        axCalCryst.set_ylabel('Normalized intensity (a.u.)')
        axCalCryst.set_xlabel("Radial distance $|q|$ (Å$^{-1}$)")

    #################################################################################################
    ### # ----- Apply calibration to braggvectors -----
    #################################################################################################

    braggpeaks.setcal()

    #################################################################################################
    ### # ----- Evaluate calibration results -----
    #################################################################################################

    if show_result:
        print(
            "Pixel size =",
            braggpeaks.calibration.get_Q_pixel_size(),
            "A^-1",
        )
        print(
            "Calibration state:",
            braggpeaks.calstate,
        )

    return braggpeaks