"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import os
import py4DSTEM as py4DSTEM
import numpy as np
import scipy
def detect_braggpeaks_py4dstem(datacube, probe, detect_params, data=None):
    """
    Detects disk positions and corresponding intensities according to the cross-
    correlation method as outlined by py4DSTEM. 
    
    Parameters:
        
    datacube: py4DSTEM datacube
       The datacube of the sample of interest.
    probe: py4DSTEM probe
        Vacuum probe template from py4DSTEM.
    detect_params: dictionary
        The to be used detection parameters as defined in the py4DSTEM find_bragg_disks
        function. 
    data: tuple (x,y) or list of tuples in shape [(x1,y2),...] or None
        Real space positions to run detection for. If None, runs on the whole
        datacube and saves automatically.
        
    Function purpose: 
        This function performs disk detection according to the py4DSTEM cross-
        correlation method and gives subpixel resolution of disk position and
        intensities. 

    Returns: 
        if data is (x,y) 
            py4DSTEM Qpoints
        if data is list of (x,y) 
            list of py4DSTEM Qpoints
        if data is None 
            py4DSTEM BraggVectors
    """
    kwargs = dict(template=probe.kernel, **detect_params)
    if data is not None:
        kwargs["data"] = data
    return datacube.find_Bragg_disks(**kwargs)


def get_braggpeaks(
    samplename,
    datacube,
    probe,
    detect_params,
    dirpath2,
    autodiskparams=None,
    overwrite=False,
    detector_func=detect_braggpeaks_py4dstem,
    mask=None,
):
    """
    Detects disk positions and corresponding intensities according to the chosen
    disk detection method.
    
    Parameters:
    samplename: str
        The name of the sample to be loaded. 
    datacube: py4DSTEM datacube
       The datacube of the sample of interest.
    probe: py4DSTEM probe
        Vacuum probe template from py4DSTEM.
    detect_params: dictionary
        The to be used detection parameters as defined in the py4DSTEM find_bragg_disks
        function. 
    dirpath2: str
        Path to where the detected positions should be stored or already are stored.
    autodiskparams: dictionary
        The to be used detection parameters as defined in the autodisk.py module.
    overwrite: bool
        Whether to overwrite existing disk positions and intensities, if applicable.
    detector_func: func
        The disk detection function to be used. Currently, AutoDisk and py4DSTEM
        are implemented. Note that the detected positions need to be returned in
        a py4DSTEM data structure, even if a different method is used. 
        
    Function purpose: 
        This function acts as a wrapper to call a desired detector function for
        a datacube of interest. 
        
    Returns: braggpeaks
        py4DSTEM BraggVectors. Is also attached to datacube of interest. 
    """
    filepath_basic_02 = os.path.join(dirpath2, f"{samplename}_basic02.h5")
    #################################################################################################
    ### # ----- Check existing disk positions -----
    #################################################################################################
    if os.path.exists(filepath_basic_02) and not overwrite:
        print(f"[diskdetection] Loading from {filepath_basic_02}")
        basic_02 = py4DSTEM.read(filepath_basic_02)
        print(type(basic_02))
        print(basic_02.tree())
        print(hasattr(basic_02, "braggvectors"))
        try:
            braggpeaks = basic_02.tree("braggvectors")
        except Exception:
            braggpeaks = None
        if braggpeaks is not None:
            print("[diskdetection] Found saved braggvectors.")
            return braggpeaks

        print("[diskdetection] No saved braggvectors found; running detection.")
    #################################################################################################
    ### # ----- Run disk detection -----
    #################################################################################################
    print("[diskdetection] Running Bragg disk detection...")
    if autodiskparams is not None:
        braggpeaks = detector_func(
        datacube,
        probe,
        detect_params,
        data=None,
        autodiskparams=autodiskparams,
        mask=mask,
    )
    else:
        braggpeaks = detector_func(
            datacube,
            probe,
            detect_params,
            data=None,
        )
    #################################################################################################
    ### # ----- Format data in py4DSTEM structure -----
    #################################################################################################
    # Give the BraggVectors object the name expected by the loader
    braggpeaks.name = "braggvectors"

    # Attach BraggVectors to the DataCube's data tree
    datacube.attach(braggpeaks)
    # attach under the standard name in basic02
    datacube.braggvectors = braggpeaks
    # Save everything under the DataCube.
    py4DSTEM.save(
        filepath_basic_02,
        datacube,
        tree=None,   #everything under datacube
        mode='o'
    )
    py4DSTEM.print_h5_tree(filepath_basic_02)
    print(f"[diskdetection] Saved braggvectors to {filepath_basic_02}")

    return braggpeaks

#################################################################################################
### # ----- AutoDisk method and helper functions -----
#################################################################################################
def refine_peaks_with_autodisk_lattice(
    pattern,
    peaks_yxs,
    center_yx,
    disk_radius,
    min_peaks=8,
    max_shift=2.5,
):
    """
   Refines detected diffraction-disk positions using AutoDisk lattice fitting.

   Parameters:
   pattern: 2D ndarray
       Diffraction pattern used by AutoDisk during lattice fitting.
   peaks_yxs: ndarray, shape (N, 3)
       Detected peak positions and scores, with columns [y, x, score].
   center_yx: array-like, shape (2,)
       Approximate central-disk position in [y, x] order.
   disk_radius: float
       Approximate disk radius in pixels.
   min_peaks: int
       Minimum number of peaks required before attempting lattice fitting.
   max_shift: float
       Maximum accepted displacement of a fitted peak from its original
       position, in pixels.

   Function purpose:
       Refines peak coordinates by fitting a lattice with AutoDisk. If there
       are too few peaks, the fit is invalid, or fitting raises a handled
       exception, the original peaks are retained. Individual fitted
       coordinates are accepted only when their displacement does not exceed
       max_shift. The score column is preserved.

   Returns: ndarray, shape (N, 3)
       Peak positions and scores in [y, x, score] format.
   """
    from . import autodisk
    peaks = np.asarray(peaks_yxs, dtype=float)

    if peaks.size == 0:
        return peaks.reshape(0, 3)

    peaks = peaks.reshape(-1, 3)
    original = peaks.copy()

    # Lattice fitting needs multiple reliable disks.
    if len(peaks) < min_peaks:
        return original

    if not np.all(np.isfinite(peaks)):
        return original

    try:
        # AutoDisk expects [y, x, score].
        # detAng uses the score column to identify the central disk.
        angle, peaks_angle = autodisk.detAng(
            peaks.copy(),
            np.asarray(center_yx, dtype=float),
            disk_radius,
        )

        if len(peaks_angle) != len(original):
            return original

        peaks_rot = autodisk.rotCtr(
            pattern,
            peaks_angle,
            angle,
        )

        if len(peaks_rot) != len(original):
            return original

        (
            vec_a,
            vec_b,
            fitted_rot,
            middle_row,
            angle_correction,
        ) = autodisk.latFit(
            pattern,
            peaks_rot,
            disk_radius,
        )

        if fitted_rot.ndim != 2 or len(fitted_rot) != len(original):
            return original

        if fitted_rot.shape[1] < 3:
            return original

        # latFit performs an additional rotation internally.
        # Transform the fitted coordinates back to the original
        # detector coordinate system.
        fitted = autodisk.rotCtr(
            pattern,
            fitted_rot,
            -(angle + angle_correction),
        )

        if fitted.ndim != 2 or len(fitted) != len(original):
            return original

        if not np.all(np.isfinite(fitted[:, :3])):
            return original

        # Avoid allowing a poor lattice fit to move a peak too far.
        shifts = np.linalg.norm(
            fitted[:, :2] - original[:, :2],
            axis=1,
        )

        accepted = shifts <= max_shift

        result = original.copy()
        result[accepted, :2] = fitted[accepted, :2]

        return result

    except (ValueError, IndexError, TypeError, ZeroDivisionError,
            np.linalg.LinAlgError, FloatingPointError):
        # Keep the successful detection/RGM result if lattice fitting
        # cannot produce a usable result.
        return original
def rescale_dp(pattern, new_min=1.0, new_max=100.0):
    """
    Linearly rescales the intensity values of a diffraction pattern.

    Parameters:
        
    pattern: 2D ndarray
        Diffraction pattern whose intensity values will be rescaled.
    new_min: float
        Minimum value of the target intensity range. Defaults to 1.0.
    new_max: float
        Maximum value of the target intensity range. Defaults to 100.0.

    Function purpose:
        Maps the finite intensity range of the input pattern linearly to
        [new_min, new_max]. If all finite values are equal, returns an array
        filled with the midpoint of the target range. Raises a ValueError if
        the calculated input intensity range is not finite.

    Returns: ndarray
        Diffraction pattern with rescaled intensity values.
    """
    pattern = np.asarray(pattern, dtype=np.float64)

    p_min = np.nanmin(pattern)
    p_max = np.nanmax(pattern)

    if not np.isfinite(p_min) or not np.isfinite(p_max):
        raise ValueError("Diffraction pattern contains no finite intensity range.")

    if p_max == p_min:
        return np.full_like(pattern, (new_min + new_max) / 2.0)

    return new_min + (pattern - p_min) * (
        (new_max - new_min) / (p_max - p_min)
    )

def preprocess_dp(pattern, sigma=0):
    """
    Preprocesses a diffraction pattern before disk detection.

    Parameters:
        
    pattern: 2D ndarray
        Diffraction pattern to preprocess.
    sigma: float or None
        Standard deviation for optional Gaussian smoothing. Smoothing is
        applied only when sigma is not None and greater than zero. Defaults
        to 0, which disables smoothing.

    Function purpose:
        Converts the pattern to float64, optionally applies a Gaussian filter,
        rescales intensities to the range [1, 100], and clips negative values
        to zero.

    Returns: ndarray
        Preprocessed diffraction pattern.
    """
    pattern = np.asarray(pattern, dtype=np.float64)

    if sigma is not None and sigma > 0:
        pattern = scipy.ndimage.gaussian_filter(
            pattern,
            sigma=sigma,
        )

    pattern = rescale_dp(
        pattern,
        new_min=1.0,
        new_max=100.0,
    )

    return np.maximum(pattern, 0.0)
def masked_average_diffraction_pattern(
    datacube,
    mask=None,
    sigma=0,
):
    """
    Calculates the average of preprocessed diffraction patterns selected
    from a datacube using a scan-space mask.
    
    Parameters:
        
    datacube: py4DSTEM DataCube
        Datacube containing diffraction patterns in datacube.data.
    mask: 2D boolean ndarray or None
        Scan-space mask with the same shape as the first two dimensions of
        datacube.data. Patterns at True positions are included. If None,
        all scan positions are included.
    sigma: float or None
        Gaussian smoothing parameter passed to preprocess_dp for each
        diffraction pattern. Defaults to 0, disabling smoothing.
    
    Function purpose:
        Preprocesses each selected diffraction pattern individually, sums the
        resulting patterns, and divides by the number selected. Raises a
        ValueError if the mask has an incorrect shape, contains no True
        positions, or no diffraction patterns are selected.
    
    Returns: ndarray
        Average of the selected preprocessed diffraction patterns.
    """
    data = datacube.data
    Rshape = data.shape[:2]

    if mask is None:
        positions = np.ndindex(Rshape)
    else:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != Rshape:
            raise ValueError(
                f"mask must have shape {Rshape}, "
                f"got {mask.shape}"
            )

        if not np.any(mask):
            raise ValueError("mask contains no True positions.")

        positions = zip(*np.nonzero(mask))

    total = None
    count = 0

    for rx, ry in positions:
        pattern = preprocess_dp(data[rx, ry], sigma=sigma)

        if total is None:
            total = np.zeros_like(pattern, dtype=np.float64)

        total += pattern
        count += 1

    if count == 0:
        raise ValueError("No diffraction patterns selected.")

    return total / count

def identify_center_disk(peaks_yx, center_yx):
    """
    Identifies the detected peak closest to an estimated central-disk position.

    Parameters:
    peaks_yx: ndarray, shape (N, 2)
        Detected peak coordinates with columns [y, x].
    center_yx: array-like, shape (2,)
        Estimated central-disk coordinates in [y, x] order.

    Function purpose:
        Calculates the Euclidean distance from each detected peak to the
        estimated center and identifies the index of the closest peak.

    Returns: int or None
        Index of the closest peak, or None if no peaks are provided.
    """
    peaks_yx = np.asarray(peaks_yx, dtype=float)
    center_yx = np.asarray(center_yx, dtype=float)

    if len(peaks_yx) == 0:
        return None

    distances = np.linalg.norm(peaks_yx - center_yx, axis=1)
    return int(np.argmin(distances))

def detect_braggpeaks_autodisk(
    datacube,
    probe,
    detect_params,
    data=None,
    autodiskparams=None,
    mask=None,
    rgarefine=False,
):
    """
    Detects Bragg disks and optionally refines their positions using AutoDisk.

    Parameters:
    datacube: py4DSTEM DataCube
        Datacube containing the diffraction patterns to process.
    probe: py4DSTEM probe
        Probe object. Its probe attribute is used to determine the approximate
        disk radius and probe center through datacube.get_probe_size().
    detect_params: dict
        Parameters forwarded to datacube.find_Bragg_disks() for candidate
        disk detection.
    data: None, tuple, or boolean ndarray
        Specifies which scan positions to process. None selects the full
        datacube; a tuple (rx, ry) selects one position when both values are
        scalars, or multiple positions when they are coordinate arrays; a
        boolean array selects positions where the mask is True.
    autodiskparams: dict or None
        Optional AutoDisk and refinement parameters. Recognized keys include
        rgm_ra, rgm_threshold, rgm_rn, rgm_np, sigma, lattice_refine,
        lattice_min_peaks, lattice_max_shift, and subpixel_step.
    mask: unused
        Present in the function signature but not used by the current
        implementation.
    rgarefine: bool
        If True, applies AutoDisk radial-gradient-maximum subpixel refinement
        to the candidate peaks. Defaults to False.

    Function purpose:
        Detects candidate peaks at the requested scan positions, optionally
        refines their coordinates with AutoDisk RGA and lattice fitting, and
        stores the coordinates and intensities in py4DSTEM QPoints objects.
        For a full datacube, combines the point lists into a BraggVectors
        object. Candidate detection is performed through
        datacube.find_Bragg_disks().

    Returns: py4DSTEM.BraggVectors, py4DSTEM.QPoints, or list of QPoints
        Returns BraggVectors for the full datacube, a single QPoints object
        for one scalar scan position, or a list of QPoints for multiple
        selected positions.
    """

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    if autodiskparams is None:
        autodiskparams = {}
    rgm_ra = autodiskparams.get("rgm_ra", None)
    rgm_threshold = autodiskparams.get("rgm_threshold", None)
    rgm_rn = autodiskparams.get("rgm_rn", None)
    rgm_np = autodiskparams.get("rgm_np", None)
    sigma = autodiskparams.get("sigma", None)
    lattice_refine = autodiskparams.get("lattice_refine", None)
    lattice_min_peaks = autodiskparams.get("lattice_min_peaks", None)
    lattice_max_shift = autodiskparams.get("lattice_max_shift", None)

    r,x0,y0=datacube.get_probe_size(probe.probe)
    # r=int(r*1)
    

    # --------------------------------------------------------
    # Determine requested scan positions
    # --------------------------------------------------------

    full_dataset = data is None
    single_position = False

    if data is None:
        Rshape = datacube.data.shape[:2]
        positions = [
            (rx, ry)
            for rx in range(Rshape[0])
            for ry in range(Rshape[1])
        ]

    elif isinstance(data, tuple) and len(data) == 2:
        rx, ry = data

        if np.isscalar(rx) and np.isscalar(ry):
            positions = [(int(rx), int(ry))]
            single_position = True

        else:
            rx = np.asarray(rx)
            ry = np.asarray(ry)

            if rx.shape != ry.shape:
                raise ValueError(
                    "rx and ry must have the same shape."
                )

            positions = [
                (int(x), int(y))
                for x, y in zip(rx.flat, ry.flat)
            ]

    elif isinstance(data, np.ndarray) and data.dtype == bool:
        Rshape = datacube.data.shape[:2]

        if data.shape != Rshape:
            raise ValueError(
                "Boolean R-space mask must have shape "
                f"{Rshape}, got {data.shape}."
            )

        positions = [
            (int(rx), int(ry))
            for rx, ry in zip(*np.nonzero(data))
        ]

    else:
        raise TypeError(
            "data must be None, a 2-tuple of scan coordinates, "
            "or an R-space boolean mask."
        )

    # --------------------------------------------------------
    # Output dtype
    # --------------------------------------------------------

    dtype = np.dtype([
        ("qx", np.float64),
        ("qy", np.float64),
        ("intensity", np.float64),
    ])

    qpoints_list = []

    # --------------------------------------------------------
    # Detection loop: AutoDisk functions only
    # --------------------------------------------------------
    # Module-level counters: reset when the Python process restarts.
    AUTODISK_RUN_COUNT = 0
    # AUTODISK_PATTERN_COUNT = 0
    for rx, ry in positions:
        AUTODISK_RUN_COUNT += 1
        # pattern = preprocess_dp(
        #     datacube.data[rx, ry],
        #     sigma=sigma,
        # )
        print(AUTODISK_RUN_COUNT)
        # Detect candidate disks using py4DSTEM
        
        # Candidate detection using py4DSTEM
        candidate_qpoints = datacube.find_Bragg_disks(
            data=(rx, ry),
            template=probe.kernel,
            **detect_params,
        )
        
        candidate_data = candidate_qpoints.data
        # Keep original py4DSTEM candidate coordinates and intensities.
        candidate_qy = candidate_data["qy"]
        candidate_qx = candidate_data["qx"]
        candidate_intensity = candidate_data["intensity"]
        candidate_blobs = np.column_stack((
            candidate_qy,
            candidate_qx,
        ))
        center_yx = np.array([y0, x0], dtype=float)
        
        center_idx = identify_center_disk(
            candidate_blobs,
            center_yx,
        )
        if center_idx is not None:
            center_qy = candidate_qy[center_idx]
            center_qx = candidate_qx[center_idx]
        if len(candidate_blobs) == 0:
            qpoints_list.append(
                py4DSTEM.QPoints(
                    np.empty(0, dtype=dtype),
                    name="AutoDisk Bragg peaks",
                )
            )
            continue

        # --------------------------------------------------------
        # Independent refinement options
        # --------------------------------------------------------
        lattice_refine = bool(autodiskparams.get("lattice_refine", False))
        lattice_min_peaks = autodiskparams.get("lattice_min_peaks", 8)
        lattice_max_shift = autodiskparams.get("lattice_max_shift", 2.5)

        # Start with the original py4DSTEM candidate positions.
        qy = candidate_qy.copy()
        qx = candidate_qx.copy()
        intensity = candidate_intensity.copy()

        # --------------------------------------------------------
        # 1. Optional RGA subpixel refinement
        # --------------------------------------------------------
        if rgarefine:
            from . import autodisk
            original_dp = datacube.data[rx, ry]

            peaks, kept_indices = autodisk.radGradMaxSubpixel(
                original_dp,
                candidate_blobs,
                r,
                rn=rgm_rn,
                ra=rgm_ra,
                n_p=rgm_np,
                threshold=rgm_threshold,
                subpixel_step=autodiskparams.get("subpixel_step", 0.08),
                return_indices=True,
            )

            if len(peaks) == 0:
                qpoints_list.append(
                    py4DSTEM.QPoints(
                        np.empty(0, dtype=dtype),
                        name="AutoDisk Bragg peaks",
                    )
                )
                continue

            peaks = np.asarray(peaks, dtype=float)
            kept_indices = np.asarray(kept_indices, dtype=np.intp)

            if len(peaks) != len(kept_indices):
                raise RuntimeError(
                    "RGA returned different numbers of peaks and indices."
                )

            if (
                np.any(kept_indices < 0)
                or np.any(kept_indices >= len(candidate_intensity))
            ):
                raise RuntimeError(
                    "RGA returned an invalid candidate index."
                )

            # Use RGA-refined coordinates and preserve original intensities.
            qy = peaks[:, 0].copy()
            qx = peaks[:, 1].copy()
            intensity = candidate_intensity[kept_indices].copy()

            # Preserve the original py4DSTEM center-disk position.
            if center_idx is not None:
                center_matches = np.flatnonzero(
                    kept_indices == center_idx
                )
                if len(center_matches) > 0:
                    i = center_matches[0]
                    qy[i] = center_qy
                    qx[i] = center_qx

        # --------------------------------------------------------
        # 2. Optional lattice refinement
        #    Runs independently of RGA.
        # --------------------------------------------------------
        if lattice_refine:
            pattern = preprocess_dp(
                datacube.data[rx, ry],
                sigma=sigma if sigma is not None else 0,
            )

            peaks_yxs = np.column_stack((qy, qx, intensity))

            peaks_refined = refine_peaks_with_autodisk_lattice(
                pattern=pattern,
                peaks_yxs=peaks_yxs,
                center_yx=np.array([y0, x0], dtype=float),
                disk_radius=r,
                min_peaks=lattice_min_peaks,
                max_shift=lattice_max_shift,
            )

            qy = peaks_refined[:, 0]
            qx = peaks_refined[:, 1]
            intensity = peaks_refined[:, 2]


        # Build QPoints with explicitly initialized intensities.
        qdata = np.empty(len(qx), dtype=dtype)
        qdata["qx"] = qx
        qdata["qy"] = qy
        qdata["intensity"] = intensity

        qpoints_list.append(
            py4DSTEM.QPoints(
                qdata,
                name="AutoDisk Bragg peaks",
            )
        )
        continue
    # --------------------------------------------------------
    # Return a single QPoints object or a list of QPoints
    # --------------------------------------------------------

    if single_position:
        return qpoints_list[0]

    if not full_dataset:
        return qpoints_list

    # --------------------------------------------------------
    # Full DataCube: construct BraggVectors
    # --------------------------------------------------------

    pointlists = py4DSTEM.PointListArray(
        dtype=dtype,
        shape=Rshape,
        name="AutoDisk Bragg peaks",
    )

    for (rx, ry), qpoints in zip(positions, qpoints_list):
        if len(qpoints.data) > 0:
            pointlists.get_pointlist(rx, ry).add(qpoints.data)

    braggvectors = py4DSTEM.BraggVectors(
        Rshape=Rshape,
        Qshape=datacube.data.shape[-2:],
        name="AutoDisk BraggVectors",
    )

    braggvectors.set_raw_vectors(pointlists)

    return braggvectors