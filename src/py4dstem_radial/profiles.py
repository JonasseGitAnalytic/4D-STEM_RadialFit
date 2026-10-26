"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

def radial_profile(
        bvm: np.ndarray,
        q_r: np.ndarray,
        q_range: tuple[float, float] = (0.2, 1.2),
        num_bins: int = 150,
        k_broadening: float = 0.005,
        normalize='max'
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate a binned radial intensity profile from a Bragg Vector Map.

    Parameters:
        
    bvm : np.ndarray
        Bragg Vector Map containing intensity values.
    q_r : np.ndarray
        Array of reciprocal-space radial distances |q|, with the same shape as
        ``bvm``; values are expected in inverse ångströms.
    q_range : tuple of float, default=(0.2, 1.2)
        Inclusive lower and upper bounds of the reciprocal-space range to include.
    num_bins : int, default=150
        Number of equally spaced radial bins.
    k_broadening : float, default=0.005
        Gaussian broadening width in reciprocal-space units. It is converted to
        a standard deviation measured in bins before smoothing.
    normalize : {'max', 'sum', 'k', None}, default='max'
        Normalization method. ``'max'`` scales by the maximum intensity; ``'sum'``
        scales by the integrated profile area; ``'k'`` divides by q and then scales
        by the maximum; ``None`` leaves the profile unnormalized.

    Function purpose:
    Bin finite BVM intensities by radial distance, interpolate empty bins, optionally
    normalize the resulting profile, and apply optional Gaussian broadening.

    Returns:
    q_centers : np.ndarray
        One-dimensional array containing the center of each radial bin.
    I_profile : np.ndarray
        One-dimensional radial intensity profile evaluated at ``q_centers``.
        If no finite data lie within ``q_range``, all profile values are NaN.

    """

    q_min_include, q_max_include = q_range

    bvm = np.asarray(bvm, dtype=float)
    q_r = np.asarray(q_r, dtype=float)

    if bvm.shape != q_r.shape:
        raise ValueError("bvm and q_r must have the same shape.")

    # Only retain finite values inside the requested q range.
    mask = (
        (q_r >= q_min_include)
        & (q_r <= q_max_include)
        & np.isfinite(q_r)
        & np.isfinite(bvm)
    )

    q_vals = q_r[mask]
    I_vals = bvm[mask]

    bins = np.linspace(
        q_min_include,
        q_max_include,
        num_bins + 1,
    )

    q_centers = 0.5 * (bins[:-1] + bins[1:])

    
    I_profile = np.full(
        q_centers.shape,
        np.nan,
        dtype=float,
    )

    inds = np.digitize(q_vals, bins)

    for b in range(len(q_centers)):
        bin_mask = inds == (b + 1)

        if np.any(bin_mask):
            I_profile[b] = np.mean(I_vals[bin_mask])

    # ---------------------------------------------------------
    # Fill empty bins by interpolation.
    # ---------------------------------------------------------
    valid = np.isfinite(I_profile)

    if not np.any(valid):
        return q_centers, np.full_like(q_centers, np.nan)

    if np.sum(valid) == 1:
        I_profile[:] = I_profile[valid][0]
    elif not np.all(valid):
        I_profile = np.interp(
            q_centers,
            q_centers[valid],
            I_profile[valid],
        )

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    if normalize == 'k':

        q_safe = np.where(
            q_centers == 0,
            np.nan,
            q_centers,
        )

        I_profile = I_profile / q_safe

        finite = np.isfinite(I_profile)

        if np.any(finite):
            max_val = np.max(I_profile[finite])

            if max_val > 0:
                I_profile /= max_val

    elif normalize == 'max':

        max_val = np.nanmax(I_profile)

        if np.isfinite(max_val) and max_val > 0:
            I_profile /= max_val

    elif normalize == 'sum':

        area = np.trapezoid(
            np.nan_to_num(I_profile, nan=0.0),
            q_centers,
        )

        if np.isfinite(area) and area > 0:
            I_profile /= area

    elif normalize is None:
        pass

    else:
        raise ValueError(
            "normalize must be 'max', 'sum', 'k', or None."
        )

    # ---------------------------------------------------------
    # Gaussian broadening
    # ---------------------------------------------------------
    if k_broadening > 0 and len(q_centers) > 1:

        dq = q_centers[1] - q_centers[0]

        if dq > 0 and np.isfinite(dq):

            sigma_bins = k_broadening / dq

            if sigma_bins > 0:
                I_profile = gaussian_filter1d(
                    I_profile,
                    sigma=sigma_bins,
                    mode='nearest',
                )

    return q_centers, I_profile

def radial_coordinates(dims):
    """
   Calculate radial reciprocal-space coordinates.

   Parameters:
       dims (tuple): Pair of coordinate arrays, dimx and dimy.

   Function purpose:
       Create coordinate grids from the input arrays and calculate the radial
       distance from the origin at each grid point.

   Returns:
       tuple: The qx coordinate grid, qy coordinate grid, and radial-distance
       array.
   """
    dimx, dimy = dims
    qx, qy = np.meshgrid(dimx, dimy, indexing='ij')
    return qx, qy, np.sqrt(qx**2 + qy**2)