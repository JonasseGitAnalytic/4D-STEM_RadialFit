"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""


import time
import numpy as np
from tqdm import tqdm
from py4dstem_radial.config import (
    AnalysisConfig)
#################################################################################################
### # ----- Bragg vector module -----
#################################################################################################

def bvm_from_vectors_vectors(
        vdata:np.ndarray,
        Qshape:tuple[int,int],
        sampling:int, 
        calstate:dict[str:bool,str:bool,str:bool,str:bool], 
        calibration, 
        mode:str='cal'
        )->np.ndarray:
    """
    Computes a 2D histogram from all vectors present in a single pixel with the 
    same interpolation rules as those found in py4DSTEM, representing a Bragg
    Vector Map for that pixel.
    
    Parameters:
        
    vdata: Array of format [float,float,float]
        x,y and intensity values corresponding to vector positions and their 
        intensities. 
    Qshape: tuple(int,int)
        The shape in pixels of the Bragg Vector object. 
    sampling: integer
        Whether to up- or downsample data in the histogram calculations.
    calstate: dictionary
        Container of boolean values for whether the calibration object contains
        the respective calibrations. 
    calibration: calibration object from py4DSTEM
        Container of all calibrations associated with the Bragg Vector object.
    mode: str
        Whether to use calibrated or uncalibrated data. Options are 'cal' and 
        'raw'.
    
    Function purpose: This function calculates the 2D histogram for a specific 
    Bragg Vector object, reproducing py4DSTEMs internal BVM logic 
    to generate a Bragg Vector Map for a specific pixel rather than all pixels. 
    
    Returns: 2D numpy array with shape [Q_Nx,Q_Ny]
        A numpy array representing the Bragg Vector Map of the vectors found in
        a specific pixel.
    """
    if vdata.size == 0:
        return np.zeros(Qshape, dtype=np.float64)
    qx = vdata['qx'].astype(np.float64)
    qy = vdata['qy'].astype(np.float64)
    # It is also possible to weight by intensity of the vector or not. By default, the BVM
    # is intensity weighted. This is recommended, as erroneous detection of noise 
    # will strongly influence unweighted BVMs.
    I = np.ones(vdata.shape[0], dtype=np.float64)
    if AnalysisConfig.BVM_intens_weight == True:
        I = vdata['intensity'].astype(np.float64)
    if mode == 'raw':
        qx *= sampling
        qy *= sampling
    else:
        if calstate["pixel"] is True:
            qpix = calibration.get_Q_pixel_size()
            qx = qx / qpix
            qy = qy / qpix
        if calstate["center"] is True:
            origin = calibration.get_origin_mean()
            qx = qx + origin[0]
            qy = qy + origin[1]
        qx = qx * sampling
        qy = qy * sampling
    Q_Nx, Q_Ny = Qshape
    floorx,floory = np.floor(qx).astype(np.int64),np.floor(qy).astype(np.int64)
    ceilx,ceily  = np.ceil(qx).astype(np.int64),np.ceil(qy).astype(np.int64)
    mask = np.logical_and.reduce(((floorx >= 0), (floory >= 0), (ceilx < Q_Nx), (ceily < Q_Ny)))
    if not np.any(mask):
        return np.zeros(Qshape, dtype=np.float64)
    qx = qx[mask]; qy = qy[mask]; I = I[mask]
    floorx = floorx[mask]; ceilx = ceilx[mask]; floory = floory[mask]; ceily = ceily[mask]
    dx = qx - floorx
    dy = qy - floory
    inds00 = np.ravel_multi_index([floorx, floory], (Q_Nx, Q_Ny))
    inds01 = np.ravel_multi_index([floorx, ceily], (Q_Nx, Q_Ny))
    inds10 = np.ravel_multi_index([ceilx, floory], (Q_Nx, Q_Ny))
    inds11 = np.ravel_multi_index([ceilx, ceily], (Q_Nx, Q_Ny))
    total = ( np.bincount(inds00, I * (1.0 - dx) * (1.0 - dy), minlength=Q_Nx * Q_Ny)
            + np.bincount(inds01, I * (1.0 - dx) * dy, minlength=Q_Nx * Q_Ny)
            + np.bincount(inds10, I * dx * (1.0 - dy), minlength=Q_Nx * Q_Ny)
            + np.bincount(inds11, I * dx * dy, minlength=Q_Nx * Q_Ny) )
    return total.reshape(Q_Nx, Q_Ny)

def compute_per_pixel_bvm(
        bragg_obj, 
        mode:str='cal', 
        sampling:int=1,
        fit_mask:np.ndarray=None)->tuple: 
    """
    Calls the Bragg Vector Map calculation for each pixel represented in the Bragg Vector object and sums
    each individual map to mimick the mean map as obtained by py4DSTEM to check the internal 
    histogram logic.
    
    Parameters:
        
    bragg_obj: BraggVectors of py4DSTEM's braggvectors module
        Intensity of a bragg peak at a specific position.
    mode: str
        Whether to use calibrated or uncalibrated data. Options are 'cal' and 
        'raw'.
    sampling: int
        Whether to up- or downsample data in the histogram calculations.
        
    Function purpose: This wrapper function calls the 2D histogram calculation for all pixels 
    individually, in addition to summing them for comparison with py4DSTEM's 
    internal histogram logic. 

    Returns: tuple[np.ndarray,np.ndarray,tuple[int,int],float,tuple[float,float],str]
        A tuple containing the per pixel Bragg Vector Maps, the mean Bragg Vector
        map, the dimensions of that Bragg Vector Map, the pixelsize, the origin
        and the units of the Bragg Vector Map dimensions (e.g. pixels, nm, etc.). 
    """
    # This function can run both with calibrated 
    # and uncalibrated datasets. 
    assert mode in ('cal', 'raw')
    v_all = bragg_obj.cal if mode == 'cal' else bragg_obj.raw
    R0, R1 = bragg_obj.Rshape  # Shape of the real space image
    Q_Nx,Q_Ny = np.round(bragg_obj.Qshape[0]*sampling).astype(int), np.round(bragg_obj.Qshape[1]*sampling).astype(int)
    Qshape = (Q_Nx, Q_Ny)
    calstate = bragg_obj.calstate
    calibration = bragg_obj.calibration
    if fit_mask is None:
        fit_mask = np.ones((R0, R1), dtype=bool)
    per_pixel = np.zeros((R0, R1, Q_Nx, Q_Ny), dtype=np.float64)
    start = time.time()
    for i in tqdm(range(R0), desc='Rows'):
        for j in range(R1):
            if not fit_mask[i, j]:
                # Skip computation for masked‑out pixels, mark as NaN
               per_pixel[i, j] = np.nan
               continue
            vdata = v_all[i, j].data
            per_pixel[i, j] = bvm_from_vectors_vectors(vdata, Qshape, sampling, calstate, calibration, mode)
    elapsed = time.time() - start
    print(f"Per-pixel BVM computed in {elapsed:.1f} s. Shape: {per_pixel.shape}")
    # The next part of the function calculates the BVM map based on the 
    # indivdual peaks found in the BVMs found computed for each pixel. This is an important diagnostic to check 
    # whether the Bragg Vector Map calculations are performed correctly in each 
    # pixel indivdually and matches the py4DSTEM map as well. We also calculate a mask where we only include those 
    # diffraction patterns which have intensity. 
    nonzero_mask = (per_pixel.sum(axis=(2,3)) > -10000)
    if np.any(nonzero_mask):
        mean_bvm = per_pixel[nonzero_mask].sum(axis=0)
    else:
        mean_bvm = per_pixel.sum(axis=(0,1))
    # We also include the previously determined calibration if we choose to run
    # the function in cal mode.
    if mode == "cal" and calstate["center"] is True:
        origin = np.array(calibration.get_origin_mean(), dtype=float)
        origin = sampling * origin
        x0, y0 = origin[0], origin[1]
    else:
        x0, y0 = 0, 0
    if mode == "cal" and calstate["pixel"] is True:
        pixelsize = calibration.get_Q_pixel_size() / sampling
    else:
        pixelsize = 1.0 / sampling
    dimx,dimy = (np.arange(Q_Nx) - x0) * pixelsize,(np.arange(Q_Ny) - y0) * pixelsize
    dim_units = calibration.get_Q_pixel_units()
    return per_pixel, mean_bvm, (dimx, dimy), pixelsize, (x0, y0), dim_units