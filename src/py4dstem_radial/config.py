"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

from dataclasses import dataclass
@dataclass
class AnalysisConfig:
    # Peak detection / fitting
    # The allowed hkl values. Only cubic crystals are supported for now. 
    # hkllist: tuple = (3, 4, 8, 11, 12)
    hkllist: tuple = (3,4,8,11,12,16,19,20)
    # hkllist: tuple = (4,8)
    # hkllist: tuple = (4)
    # The maximum amount of peaks that are used for lattice parameter fitting.
    npeaks: int = 5
    # The range of reciprocal space that is used for fitting. 
    # q_range: tuple = (0.4, 0.76)
    q_range: tuple = (0.38, 1.2)
    # q_range: tuple = (0.1, 1.2)
    #The minimum vertical distance a peak needs to have to its lowest contour line.
    peak_prominence: float = 0.005
    #The bounding range of allowed lattice parameters in fitting.
    a_bounds: tuple = (3.615, 4.0782)
    # a_bounds: tuple = (3.615, 4.2782)
    # a_bounds: tuple = (3.815, 4.2782)
    # Radial profiles
    # Amount of bins to divide histograms in
    num_bins: int = 300
    #Broadening of radial plots. 
    k_broadening: float = 0.005
    # Strain / particle analysis
    #Depth (in pixels) from outer pixels of particle that define the shell
    shell_thickness: int = 3
    #Reference lattice parameter to compare the measured data with. 3.92 is for Pt
    # Used in strain calculations.
    # a_ref: float = 4.0 #Au-Pd cross-grating
    # a_ref: float = 3.92 #Pt
    a_ref: float = 3.9239 #Pt
    # a_ref: float = 4.0782 #Au
    # a_ref: float = 3.8902 #Pd
    #The atomic number for the reference crystal, here 78 = Pt. This only influences the crystal object.
    atom_num: int = 78
    # Data processing
    #The binning of the diffraction space data (e.g. binning of 2 means [512,512]->[256,256]).
    binning: int = 1
    #Whether to under- or oversample when creating the bragg vector maps.
    sampling: int = 1
    # Histograms / plotting
    #strain range used in the histograms and maps
    histrange: tuple = (-10, 7)
    #Amount of bins in the colorbars and the histograms
    histmapbins: int = 30
    #Range of the error map. 0.8 times this value is also the cutoff for all high error points included in histograms. 
    errorrange: float = 0.25
    # Whether to smooth the diffraction patterns according to a gaussian kernel
    smooth: bool = True
    # Whether to apply a soberfilter. This is applied prior to probe determination.
    soberfilter: bool = False
    # Whether to weight the per pixel BVMs by intensity or not.
    BVM_intens_weight = True

@dataclass
class SampleConfig:
    # Additional samples can be placed in sample.py. Below, their config files are defined. 
    #Name of sample. All 4D-STEM files are expected to be in a folder with this name.
    name: str 
    #We use the input lattice parameter of nanostructure of interest. 
    #Although not strictly necessary this speeds up the fitting procedure later. 
    lattice_parameter: float 
    #The pixel size of the sample. If it is set to None, the code will attempt to fit
    pixel_size: tuple
    #Depth and range around origin that is used for elliptical correction.
    q_int: int = 22 #depth
    q_ell_range: int = 15 #range
    # roi_range. Can be use to manually select a vacuum probe region if desired.
    roi_range: tuple = (2,6)
    overrideROI: bool = False
    #Diagnostic points. If desired, provide a list of tuples (x,y,color) of scan positions of interest.
    # Otherwise, code automatically selects diagnostic points to inspect. 
    diagnostic_points_override : tuple = None
    #Whether to load a pre-existing mask from for example imageJ and if so from where. 
    mask_directory: str | bool | None = None
    #Otsu threshold cutoff, higher values mean more cutoff.
    otsucutoff: float = 0.5
@dataclass #This dataclass is only used for the AutoDisk disk fitting approach. 
class AutoDiskConfig:
    c: float = 0.7 #The fraction used to reduce the probe for cross-correlation. Basically, this represents the reduced disk size.
    log_threshold: float = 0.1
    log_overlap: float = 0.6
    rgm_ra: int = 1
    rgm_rn: int = 80
    rgm_np: int = 40
    rgm_threshold: float = 10.0
    sigma: float = 2.0
    lattice_refine: bool = True
    lattice_min_peaks: int = 8
    lattice_max_shift: float = 4

@dataclass
class DiskDetectionConfig:
    minAbsoluteIntensity: float = 2 #standard is 2
    minRelativeIntensity: float = 0.02 #standard was 0.02
    relativeToPeak: int = 1 #standard is 1, which is the most intense disk after the undiffracted disk
    minPeakSpacing: int = 7 #Standard is 7, in pixels
    edgeBoundary: int = 2 #standard is 2, in pixels.
    sigma: float = 1.0 #The smoothing of the cross-correlation function. standard is 1, in standard deviation of the gaussian.
    maxNumPeaks: int = 100 #The maximum amount of disks that is retained. standard is 100. 
    subpixel: str = 'multicorr' #The subpixel correlation method. Multicorr is most accurate.
    corrPower: float = 1.0 #The strength of the cross-correlation function. Higher values require stronger correlation.
    sigma_dp: float = 1.0 #The smoothing of the disk diffraction patterns prior to disk detection, standard is 1, in standard deviation of the gaussian.
    CUDA: bool = True #Whether to use GPU accelaration. Is strongly recommended to run the code with appreciable speeds.
    # ML: bool = False # not implemented yet. 
@dataclass
class OutputConfig:
    # Where and whether to save the figures.
    save: bool = True
    save_data: bool = True
    save_figures: bool = True
    output_dir: str = '' 