"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import numpy as np
from scipy.ndimage import convolve
from py4DSTEM import show
from scipy.ndimage import sobel

def smoothingkernel(datacube):
    """
   Smooth diffraction patterns in a 4D-STEM datacube using a custom kernel.

   Parameters:
       datacube (py4DSTEM.DataCube): Datacube containing the diffraction patterns.

   Function purpose:
       Normalize a custom weighted kernel and convolve the datacube along its
       diffraction dimensions. Display diffraction-pattern summaries and a
       selected pattern before and after smoothing, and replace datacube.data
       with the smoothed data.

   Returns:
       numpy.ndarray: The smoothed datacube data.
   """
    data=datacube.data
    dp_mean,dp_max = datacube.get_dp_mean(),datacube.get_dp_max()
    show([dp_mean,dp_max],title='Mean (left) and max (right) diffraction pattern\n before smoothing')
    dp = datacube[24,25]
    show(dp,title='Diffraction pattern before \nsmoothing with custom kernel')
    show(dp)
    # Weighted kernel (assuming a circular electron probe of twice the size of 
    # each pixel and the beam centered on each pixel position.)
    kernel = np.array([
        [0.0954, 0.44,   0.0954],
        [0.44,   1.0,    0.44],
        [0.0954, 0.44,   0.0954]  
    ], dtype=np.float64)
    # We then normalize the the kernel so that the final sum of intensities in
    # the destination pixel is correct. 
    kernel = kernel / kernel.sum()
    # We then match the kernel shape to that of the datacube, making sure that 
    # we only smooth over the diffraction pattern data, and not the annular 
    # dark field image.
    kernel = kernel[:, :, None, None]     # shape becomes (3,3,1,1)
    # We then apply the smoothing over our dataset. 
    data_smoothed = convolve(data, kernel, mode='nearest')
    # We then replace the datacube contents and recheck the diffraction pattern
    # at the same pixel position.
    datacube.data=data_smoothed
    dp_mean,dp_max = datacube.get_dp_mean(),datacube.get_dp_max()
    show([dp_mean,dp_max],title='Mean (left) and max (right) diffraction pattern\n after smoothing')
    show([dp_mean,dp_max])
    dp = datacube[24,25]
    show(dp,title='Diffraction pattern after \nsmoothing with custom kernel')
    show(dp)
    return datacube.data

def sobel_edge_magnitude(dp):
    """
    Calculate the Sobel gradient magnitude of a 2D diffraction pattern.

    Parameters:
        dp (array-like): Two-dimensional diffraction-pattern intensity data.

    Function purpose:
        Convert the input to a float32 NumPy array, calculate Sobel gradients
        along both dimensions, and combine them into a gradient-magnitude array.
        The input diffraction pattern is not modified.

    Returns:
        np.ndarray: Two-dimensional Sobel gradient-magnitude array.

    Raises:
        ValueError: If dp is not a two-dimensional array.
    """
    dp = np.asarray(dp, dtype=np.float32)

    if dp.ndim != 2:
        raise ValueError("dp must be a 2D diffraction pattern.")

    grad_y = sobel(dp, axis=0, mode="nearest")
    grad_x = sobel(dp, axis=1, mode="nearest")

    return np.hypot(grad_y, grad_x)

def smoothingkernelsobel(datacube):
    """
    Apply Sobel edge filtering to the diffraction patterns in a datacube.
    
    Parameters:
        datacube (py4DSTEM.DataCube): Datacube containing the diffraction patterns.
    
    Function purpose:
        Calculate Sobel gradients along the two diffraction dimensions and
        combine them into a gradient magnitude. Replace datacube.data with the
        filtered data and display diffraction-pattern summaries before and after
        filtering. Assumes data has shape (scan_x, scan_y, diffraction_x, diffraction_y).
    
    Returns:
        numpy.ndarray: The Sobel-filtered datacube data.
    """
    data = datacube.data

    dp_mean, dp_max = datacube.get_dp_mean(), datacube.get_dp_max()
    show([dp_mean, dp_max],
         title='Mean (left) and max (right) diffraction pattern before Sobel filtering')

    dp = datacube[24, 25]
    show(dp, title='Diffraction pattern before Sobel filtering')

    # Apply Sobel gradients along the two diffraction dimensions only.
    # Assumes data.shape = (scan_x, scan_y, diffraction_x, diffraction_y)
    grad_x = sobel(data, axis=2, mode='nearest')
    grad_y = sobel(data, axis=3, mode='nearest')

    # Combine both directional gradients into a gradient magnitude.
    data_sobel = np.hypot(grad_x, grad_y)

    # Replace the datacube data with the Sobel-filtered data.
    datacube.data = data_sobel

    dp_mean, dp_max = datacube.get_dp_mean(), datacube.get_dp_max()
    show([dp_mean, dp_max],
         title='Mean (left) and max (right) diffraction pattern after Sobel filtering')

    dp = datacube[24, 25]
    show(dp, title='Diffraction pattern after Sobel filtering')

    return datacube.data
