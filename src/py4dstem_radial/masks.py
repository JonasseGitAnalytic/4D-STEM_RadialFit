"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from pathlib import Path

def pixel_aligned_boundaries(
    mask:np.ndarray
    )->list:
    """
    Determines the outline of the input pixel region.
    
    Parameters:
        
    mask: 2D array of bools
        This array contains boolean values for each pixel in an image of interest
        representing whether to use that pixel for determining the outline. 
 
    Function purpose: This function determines the outline of a connected pixel 
    region of interest. 
    
    Returns: list of tuples
        This list contains tuples that define the outer edges of the pixels of interest. 
    """
    segments = []
    ny, nx = mask.shape
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x]:
                continue
            # left edge
            if x == 0 or not mask[y, x-1]:
                segments.append(((x, y), (x, y+1)))
            # right edge
            if x == nx-1 or not mask[y, x+1]:
                segments.append(((x+1, y), (x+1, y+1)))
            # bottom edge
            if y == 0 or not mask[y-1, x]:
                segments.append(((x, y), (x+1, y)))
            # top edge
            if y == ny-1 or not mask[y+1, x]:
                segments.append(((x, y+1), (x+1, y+1)))
    return segments
def lowest_intensity_roi(
    adf_image,
    sizes=None,
    border=1,
):
    """
    Find the rectangular ROI with the lowest mean intensity.

    The ROI shape is selected automatically from a set of candidate
    rectangular sizes. The complete ROI must remain at least `border`
    pixels away from the image boundary.

    Parameters:
        
    adf_image : np.ndarray
        2D ADF image.
    sizes : list of tuple of int, optional
        Candidate ROI shapes as (height, width).
    border : int
        Minimum distance between the ROI and the image border.

    Returns:
        
    mask : np.ndarray
        Boolean mask containing the selected ROI.
    roi_x, roi_y : int
        Top-left coordinates of the selected ROI.
    """
    image = adf_image.astype(float)

    if sizes is None:
        sizes = [
            (2, 4),
            (3, 3), (3, 5), (3, 7),
            (5, 3), (5, 5), (5, 7),
            (7, 3), (7, 5), (7, 7),
        ]

    height, width = image.shape

    # Integral image for fast rectangular sums.
    integral = np.pad(
        np.cumsum(
            np.cumsum(image, axis=0),
            axis=1,
        ),
        ((1, 0), (1, 0)),
        mode="constant",
    )

    best_mean = np.inf
    best_roi = None

    for roi_height, roi_width in sizes:

        # Valid top-left coordinates.
        y_start = border
        y_stop = height - border - roi_height + 1

        x_start = border
        x_stop = width - border - roi_width + 1

        if y_stop <= y_start or x_stop <= x_start:
            continue

        # Rectangular sums for every possible ROI position.
        sums = (
            integral[
                roi_height + y_start : roi_height + y_stop,
                roi_width + x_start : roi_width + x_stop,
            ]
            - integral[
                y_start:y_stop,
                roi_width + x_start : roi_width + x_stop,
            ]
            - integral[
                roi_height + y_start : roi_height + y_stop,
                x_start:x_stop,
            ]
            + integral[
                y_start:y_stop,
                x_start:x_stop,
            ]
        )

        means = sums / (roi_height * roi_width)

        # Find the lowest-intensity ROI for this shape.
        index = np.unravel_index(
            np.argmin(means),
            means.shape,
        )

        roi_mean = means[index]

        if roi_mean < best_mean:
            best_mean = roi_mean

            roi_y = y_start + index[0]
            roi_x = x_start + index[1]

            best_roi = (
                roi_x,
                roi_y,
                roi_width,
                roi_height,
            )

    if best_roi is None:
        raise ValueError(
            "No valid ROI fits within the image with the requested border."
        )

    roi_x, roi_y, roi_width, roi_height = best_roi

    mask = np.zeros_like(image, dtype=bool)
    mask[
        roi_y:roi_y + roi_height,
        roi_x:roi_x + roi_width,
    ] = True

    return mask, roi_x, roi_y

def match_adf_shape(im_adf, a_map):
    """
    Match the shape of the ADF image to the lattice-parameter map.

    Parameters:

    im_adf : np.ndarray
        ADF image.
    a_map : np.ndarray
        Lattice-parameter map used as the reference shape.

    Function purpose: This function checks whether the ADF image and
    lattice-parameter map have the same shape. If the shapes differ, it
    transposes the ADF image if that produces a matching shape. A
    ValueError is raised if neither orientation matches.

    Returns: np.ndarray
        The original or transposed ADF image with a shape matching a_map.
    """
    if im_adf.shape == a_map.shape:
        return im_adf

    print("Warning: ADF and a_map shapes differ, generating transpose.")

    if im_adf.T.shape == a_map.shape:
        return im_adf.T

    raise ValueError(
        f"Warning: Shape mismatch ADF={im_adf.shape}, "
        f"a_map={a_map.shape}"
    )
def apply_adf_mask(im_adf, a_map, strain_std_map, otsucutoff=0.5):
    """
   Create a particle mask from the ADF image and apply it to data maps.

   Parameters:

   im_adf : np.ndarray
       ADF image used to determine the particle boundary.
   a_map : np.ndarray
       Lattice-parameter map to be masked.
   strain_std_map : np.ndarray
       Map of strain uncertainty or standard deviation to be masked.
   otsucutoff : float
       Multiplicative factor applied to the Otsu threshold. Default is 0.5.

   Function purpose: This function calculates an Otsu threshold from the
   ADF image, scales the threshold by otsucutoff, and identifies particle
   pixels with intensities above the resulting threshold. Pixels outside
   the particle are set to NaN in the lattice-parameter and strain-error
   maps.

   Returns: tuple
       A tuple containing the Boolean particle mask, the masked
       lattice-parameter map, the masked strain-error map, and the
       scaled threshold value.
   """
    thresh_adf = threshold_otsu(im_adf)
    thresh_adf *= otsucutoff

    mask_particle = im_adf > thresh_adf

    a_map_masked = np.where(
        mask_particle,
        a_map,
        np.nan,
    )

    strain_std_map_masked = np.where(
        mask_particle,
        strain_std_map,
        np.nan,
    )

    print(
        f"ADF mask applied using Otsu threshold={thresh_adf:.3f}"
    )

    return (
        mask_particle,
        a_map_masked,
        strain_std_map_masked,
        thresh_adf,
    )

def largest_contiguous_region(mask):
    """
    Identify the largest contiguous region in a binary mask.

    Parameters:

    mask : np.ndarray
        Boolean mask containing one or more connected regions.

    Function purpose: This function labels connected regions in the mask,
    calculates their sizes, and selects the region containing the most
    pixels. A ValueError is raised if no connected region is found.

    Returns: np.ndarray
        Boolean mask containing only the largest connected region.
    """
    labeled_mask, num_features = ndimage.label(mask)

    if num_features == 0:
        raise ValueError("No contiguous region found in mask.")

    sizes = ndimage.sum(
        mask,
        labeled_mask,
        range(1, num_features + 1),
    )

    largest_label = np.argmax(sizes) + 1

    return labeled_mask == largest_label
def apply_largest_particle_mask(
    a_map_masked,
    strain_std_map,
):
    """
   Retain only the largest contiguous particle region in the data maps.

   Parameters:

   a_map_masked : np.ndarray
       Lattice-parameter map in which pixels outside the particle are
       represented by NaN.
   strain_std_map : np.ndarray
       Map of strain uncertainty or standard deviation.

   Function purpose: This function identifies the largest contiguous
   region of finite values in a_map_masked and masks both input maps
   to retain only that region.

   Returns: tuple
       A tuple containing the cleaned Boolean particle mask, the cleaned
       lattice-parameter map, and the masked strain-error map.
   """
    particle_mask = np.isfinite(a_map_masked)

    clean_particle_mask = largest_contiguous_region(
        particle_mask
    )

    a_map_masked = np.where(
        clean_particle_mask,
        a_map_masked,
        np.nan,
    )

    strain_std_map_masked = np.where(
        clean_particle_mask,
        strain_std_map,
        np.nan,
    )

    return (
        clean_particle_mask,
        a_map_masked,
        strain_std_map_masked,
    )
def create_shell_mask(
    particle_mask,
    shell_thickness,
):
    """
    Create a shell region along the boundary of a particle.

    Parameters:

    particle_mask : np.ndarray
        Boolean mask identifying the particle.
    shell_thickness : int or float
        Maximum distance, in pixels, from the particle boundary for
        pixels to be included in the shell.

    Function purpose: This function calculates the distance of each
    particle pixel from the nearest background pixel and selects particle
    pixels whose distance is less than or equal to shell_thickness.

    Returns: np.ndarray
        Boolean mask identifying the outer shell of the particle.
    """
    distance_to_surface = ndimage.distance_transform_edt(
        particle_mask
    )

    outer_shell_mask = (
        particle_mask
        & (distance_to_surface <= shell_thickness)
    )

    return outer_shell_mask
def create_particle_regions(
    a_map_masked,
    strain_std_map,
    shell_thickness,
):
    """
   Divide the particle into outer-shell and inner-particle regions.

   Parameters:

   a_map_masked : np.ndarray
       Masked lattice-parameter map, with NaN values outside the particle.
   strain_std_map : np.ndarray
       Map of strain uncertainty or standard deviation.
   shell_thickness : int or float
       Thickness of the outer shell in pixels.

   Function purpose: This function identifies the largest contiguous
   particle region, creates a mask for its outer shell, and defines the
   inner-particle region as the remaining particle pixels. It also
   creates lattice-parameter maps containing values only within the
   outer shell and inner-particle regions, respectively.

   Returns: tuple
       A tuple containing the cleaned particle mask, outer-shell mask,
       inner-particle mask, outer-shell lattice-parameter map, and
       inner-particle lattice-parameter map.
   """
    clean_particle_mask = largest_contiguous_region(
        np.isfinite(a_map_masked)
    )

    outer_shell_mask = create_shell_mask(
        clean_particle_mask,
        shell_thickness,
    )

    inner_particle_mask = (
        clean_particle_mask
        & ~outer_shell_mask
    )

    a_outer_map_masked = np.where(
        outer_shell_mask,
        a_map_masked,
        np.nan,
    )

    inner_particle = np.where(
        inner_particle_mask,
        a_map_masked,
        np.nan,
    )

    return (
        clean_particle_mask,
        outer_shell_mask,
        inner_particle_mask,
        a_outer_map_masked,
        inner_particle,
    )
def create_high_error_mask(
    inner_particle_mask,
    strain_std_map,
    a_map_masked,
    errorrange,
):
    """
  Identify inner-particle pixels with high strain uncertainty.

  Parameters:

  inner_particle_mask : np.ndarray
      Boolean mask identifying the inner-particle region.
  strain_std_map : np.ndarray
      Map of strain uncertainty or standard deviation.
  a_map_masked : np.ndarray
      Masked lattice-parameter map.
  errorrange : float
      Reference strain-error threshold. Pixels are selected when their
      strain uncertainty exceeds 0.8 times this value.

  Function purpose: This function identifies inner-particle pixels
  whose strain uncertainty exceeds the specified threshold and creates
  a lattice-parameter map containing values only at those pixels.

  Returns: tuple
      A tuple containing the Boolean high-error mask and the
      corresponding high-error lattice-parameter map.
  """
    high_error_inner_mask = (
        inner_particle_mask
        & (strain_std_map > errorrange * 0.8)
    )

    a_inner_higherror_map = np.where(
        high_error_inner_mask,
        a_map_masked,
        np.nan,
    )

    return (
        high_error_inner_mask,
        a_inner_higherror_map,
    )
def map_to_dataframe(image):
    """
    Convert finite image values into a single-column DataFrame.

    Parameters:

    image : np.ndarray
        Input image or two-dimensional data map.

    Function purpose: This function flattens the input array, removes
    NaN values, and stores the remaining values in a single-column
    pandas DataFrame.

    Returns: pandas.DataFrame
        DataFrame containing the finite image values in a column named
        "Column1".
    """
    import pandas as pd

    data = image.flatten()
    data = data[~np.isnan(data)]

    return pd.DataFrame({"Column1": data})
def map_to_dataframe_withcoordinates(image):
    """
   Convert finite image values and their pixel coordinates into a DataFrame.

   Parameters:

   image : np.ndarray
       Input image or two-dimensional data map.

   Function purpose: This function flattens the image and its pixel
   coordinates, removes entries whose image values are NaN, and stores
   the remaining coordinates and values in a pandas DataFrame.

   Returns: pandas.DataFrame
       DataFrame containing the pixel coordinates in columns "x" and
       "y" and the corresponding image values in "Column1".
   """
    import numpy as np
    import pandas as pd

    x,y = np.indices(image.shape)

    data = image.flatten()
    x = x.flatten()
    y = y.flatten()

    mask = ~np.isnan(data)

    return pd.DataFrame({
        "x": x[mask],
        "y": y[mask],
        "Column1": data[mask]
    })
def load_mask(mask_directory, samplename,mask_name, expected_shape, default_directory=None):
    """
   Load a saved Boolean mask from a NumPy file.

   Parameters:

   mask_directory : str, pathlib.Path, bool, or None
       Directory containing the mask file. If True, default_directory
       is used. If None, no mask is loaded.
   samplename : str
       Sample name used to construct the mask filename.
   mask_name : str
       Name of the mask to load.
   expected_shape : tuple of int
       Expected shape of the loaded mask.
   default_directory : str or pathlib.Path, optional
       Default directory used when mask_directory is True.

   Function purpose: This function constructs the mask filepath from
   the sample name and mask name, loads the mask if the file exists,
   and converts its values to Boolean. It raises a ValueError if the
   loaded mask does not match expected_shape.

   Returns: np.ndarray or None
       The loaded Boolean mask if the file exists and has the expected
       shape; otherwise, None.
   """
    if mask_directory is None:
        return None

    if mask_directory is True:
        mask_directory = default_directory

    filepath = Path(mask_directory) / f"{samplename}_{mask_name}.npy"

    if not filepath.exists():
        return None

    mask = np.load(filepath).astype(bool)

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask '{mask_name}' has shape {mask.shape}, "
            f"expected {expected_shape}."
        )

    print(f"[masks] Loaded {mask_name} from {filepath}")
    return mask
def get_particle_mask(sample, im_adf, a_map, strain_std_map, dirpath2,otsucutoff):
    """
  Retrieve or generate the particle mask for a sample.

  Parameters:

  sample : object
      Sample object containing the sample name and mask-directory settings.
  im_adf : np.ndarray
      ADF image used for threshold-based particle detection.
  a_map : np.ndarray
      Lattice-parameter map.
  strain_std_map : np.ndarray
      Map of strain uncertainty or standard deviation.
  dirpath2 : str or pathlib.Path
      Default directory used when the sample mask directory is True.
  otsucutoff : float
      Multiplicative factor applied to the Otsu threshold when generating
      a particle mask.

  Function purpose: This function attempts to load a saved particle
  mask for the sample. If no saved mask is available, it generates a
  mask from the ADF image using Otsu thresholding. The selected mask
  is then applied to the lattice-parameter and strain-error maps.

  Returns: tuple
      A tuple containing the particle mask, masked lattice-parameter
      map, masked strain-error map, and threshold value. The threshold
      value is None when a saved mask is used.
  """
    print(sample.mask_directory)
    mask = load_mask(sample.mask_directory,sample.name, "particle_mask", a_map.shape, dirpath2)
    if mask is not None:
        a_map_masked = np.where(mask, a_map, np.nan)
        strain_std_map_masked = np.where(mask, strain_std_map, np.nan)
        return mask, a_map_masked, strain_std_map_masked, None
    if mask is None:
        print(
            f"[masks] No saved particle mask found for {sample.name}. "
            "Falling back to ADF thresholding."
        )
    return apply_adf_mask(im_adf, a_map, strain_std_map,otsucutoff)


def get_clean_particle_mask(sample, a_map_masked, strain_std_map, dirpath2):
    """
   Retrieve or generate a mask containing only the largest particle region.

   Parameters:

   sample : object
       Sample object containing the sample name and mask-directory settings.
   a_map_masked : np.ndarray
       Masked lattice-parameter map.
   strain_std_map : np.ndarray
       Map of strain uncertainty or standard deviation.
   dirpath2 : str or pathlib.Path
       Default directory used when the sample mask directory is True.

   Function purpose: This function attempts to load a saved clean-particle
   mask. If no saved mask is available, it identifies the largest
   contiguous region of finite values in the lattice-parameter map.
   The resulting mask is applied to both data maps.

   Returns: tuple
       A tuple containing the clean-particle mask, cleaned
       lattice-parameter map, and masked strain-error map.
   """
    mask = load_mask(sample.mask_directory,sample.name, "clean_particle_mask", a_map_masked.shape, dirpath2)
    if mask is not None:
        a_map_masked = np.where(mask, a_map_masked, np.nan)
        strain_std_map_masked = np.where(mask, strain_std_map, np.nan)
        return mask, a_map_masked, strain_std_map_masked
    if mask is None:
        print(
            f"[masks] No saved clean particle mask found for {sample.name}. "
            "Falling back to ADF thresholding."
        )
    return apply_largest_particle_mask(a_map_masked, strain_std_map)


def get_particle_regions(sample, clean_particle_mask, a_map_masked, strain_std_map, shell_thickness,dirpath2):
    """
   Retrieve or generate the outer-shell and inner-particle masks.

   Parameters:

   sample : object
       Sample object containing the sample name and mask-directory settings.
   clean_particle_mask : np.ndarray
       Boolean mask identifying the cleaned particle region.
   a_map_masked : np.ndarray
       Masked lattice-parameter map.
   strain_std_map : np.ndarray
       Map of strain uncertainty or standard deviation.
   shell_thickness : int or float
       Thickness of the outer shell in pixels.
   dirpath2 : str or pathlib.Path
       Default directory used when the sample mask directory is True.

   Function purpose: This function attempts to load saved outer-shell
   and inner-particle masks. If either mask is unavailable, it generates
   the missing mask from the clean-particle mask and shell thickness.
   It then creates separate lattice-parameter maps for the outer shell
   and inner-particle regions.

   Returns: tuple
       A tuple containing the clean-particle mask, outer-shell mask,
       inner-particle mask, outer-shell lattice-parameter map, and
       inner-particle lattice-parameter map.
   """
    outer_shell_mask = load_mask(sample.mask_directory,sample.name, "outer_shell_mask", a_map_masked.shape,dirpath2)
    inner_particle_mask = load_mask(sample.mask_directory,sample.name, "inner_particle_mask", a_map_masked.shape,dirpath2)
    if inner_particle_mask is None:
        print(
            f"[masks] No saved particle inner shell mask found for {sample.name}. "
            "Falling back to ADF thresholding."
        )
    if outer_shell_mask is None:
        print(
            f"[masks] No saved particle outer shell mask found for {sample.name}. "
            "Falling back to ADF thresholding."
        )
    if outer_shell_mask is None:
        outer_shell_mask = create_shell_mask(clean_particle_mask, shell_thickness)

    if inner_particle_mask is None:
        inner_particle_mask = clean_particle_mask & ~outer_shell_mask

    a_outer_map_masked = np.where(outer_shell_mask, a_map_masked, np.nan)
    inner_particle = np.where(inner_particle_mask, a_map_masked, np.nan)


    return clean_particle_mask, outer_shell_mask, inner_particle_mask, a_outer_map_masked, inner_particle


def get_high_error_inner_mask(sample, inner_particle_mask, strain_std_map, a_map_masked, errorrange,dirpath2):
    """
    Retrieve or generate a mask of high-error pixels in the inner particle.

    Parameters:

    sample : object
        Sample object containing the sample name and mask-directory settings.
    inner_particle_mask : np.ndarray
        Boolean mask identifying the inner-particle region.
    strain_std_map : np.ndarray
        Map of strain uncertainty or standard deviation.
    a_map_masked : np.ndarray
        Masked lattice-parameter map.
    errorrange : float
        Reference strain-error threshold used to identify high-error pixels.
    dirpath2 : str or pathlib.Path
        Default directory used when the sample mask directory is True.

    Function purpose: This function attempts to load a saved high-error
    inner-particle mask. If no saved mask is available, it identifies
    inner-particle pixels whose strain uncertainty exceeds 0.8 times
    errorrange and creates the corresponding masked lattice-parameter map.

    Returns: tuple
        A tuple containing the Boolean high-error mask and the
        corresponding high-error lattice-parameter map.
    """
    mask = load_mask(sample.mask_directory,sample.name, "high_error_inner_mask", a_map_masked.shape,dirpath2)
    if mask is not None:
        a_inner_higherror_map = np.where(mask, a_map_masked, np.nan)
        return mask, a_inner_higherror_map
    if mask is None:
        print(
            f"[masks] No saved particle high error mask found for {sample.name}. "
            "Falling back to ADF thresholding."
        )
    return create_high_error_mask(inner_particle_mask, strain_std_map, a_map_masked, errorrange)