"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""
#################################################################################################
### # ----- Imports all necessary python packages ----- #TODO: move this package list to an __init__.py file
#################################################################################################
import time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
import scipy
import py4DSTEM
from pprint import pformat
from py4DSTEM import show
#################################################################################################
### # ----- Imports all 4dstem_radial functions -----
#################################################################################################
from py4dstem_radial.config import (
    AnalysisConfig,
    DiskDetectionConfig,
    OutputConfig,
    AutoDiskConfig,
)
from py4dstem_radial.samples import samples #TODO: replace samples.py by a readable json file that is parsed by the script
from py4dstem_radial.vacuumprobe import get_vacuum_probe
from py4dstem_radial.masks import (
    lowest_intensity_roi,
    match_adf_shape,
    get_particle_mask,
    get_clean_particle_mask,
    get_particle_regions,
    get_high_error_inner_mask,
    map_to_dataframe,
    map_to_dataframe_withcoordinates,
)
from py4dstem_radial.calibrations import calibrate_braggpeaks
from py4dstem_radial.fitting import (
    fit_lattice_parameter_1D,
    calculate_lattice_parameter_maps,
)
from py4dstem_radial.saving import (
    save_strain_data,
    save_strain_figures,
)
from py4dstem_radial import (
    radial_profile,
    compute_per_pixel_bvm,
    straincalculator,
)
from py4dstem_radial.profiles import radial_coordinates
from py4dstem_radial.diskdetection import (
    get_braggpeaks,
    detect_braggpeaks_py4dstem,
)
from py4dstem_radial.visualization import (
    radialdiagnostic,
    mapfunction,
    imagehistogram,
    fitting_diagnostic,
    reference_crystal_diagnostic,
    bvm_diagnostic,
    disk_detection_diagnostic,
)
from py4dstem_radial.selection import get_diagnostic_points
from py4dstem_radial.loading import get_basic01
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", mpl.__version__)
print("py4DSTEM:", py4DSTEM.__version__)
#%%
###
"""
This code was developed by Just Pé Jonasse at Utrecht University during his PhD
research. 
"""
#################################################################################################
### # ----- Sets matplotlib parameters to make graphs show up nice -----
#################################################################################################
n = 30
color = plt.cm.inferno(np.linspace(0, 1, n))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['figure.titlesize'] = 22
mpl.rcParams['figure.labelsize'] = 18

#################################################################################################
### # ----- Define the directories that contain the datafiles -----
#################################################################################################
# Dirpath is the directory containing the raw 4DSTEM data and possibly a corresponding ADF image. Dirpath2 is the analyses will end up. 
#TODO: change this to a config file or make the directory paths parseable from the command line with argparser https://docs.python.org/3/library/argparse.html
dirpath = ""
dirpath2 = ""
#################################################################################################
### # ----- Load the config files. These are designed as dataclasses to allow for modification. -----
#################################################################################################
config = AnalysisConfig()
detect_config = DiskDetectionConfig()
autodiskdetect_config= AutoDiskConfig()
output_config = OutputConfig()
#################################################################################################
### # ----- Define the samples to analyze. Samples represent the folder name -----
### # ----- located at dirpath that contains two files; Diffraction SI.dm4 and an ADF image.-----
#################################################################################################
#TODO: make this a parseable input instead of a hardcoded list
samples_to_analyze = [
     "Particle8",
]
###
###
#%%
#################################################################################################
### # ----- Simple for loop with all necessary steps to generate strain maps of samples. -----
#################################################################################################
for samplename in samples_to_analyze:
    #################################################################################################
    ### # ----- Loading and preprocessing of data. -----
    #################################################################################################
    # First we define essential values. For this purpose, we use the sample config.py.
    # You can change or save different versions of this file to change the functionality. 
    # Inspect the config file to understand what each parameter represents.
    sample = samples[samplename]
    latpar = sample.lattice_parameter
    pixelsize = sample.pixel_size
    ROIRange = sample.roi_range
    overrideROI =sample.overrideROI
    npeaks = config.npeaks
    #Simple timing function to check speed.
    start_time = time.time()
    print(f"=== Processing {samplename} with Lattice parameter={latpar} ===")
    filepath_data_ADF = dirpath + samplename + '\\ADF Image.dm4'
    filepath_data = dirpath + samplename + '\\Diffraction SI.dm4'
    filepath_basics_01 = dirpath2 + samplename + "_basic01.h5"
    filepath_basics_02 = dirpath2 + samplename + "_basic02.h5"
    filepath_basics_03 = dirpath2 + samplename + "_basic03.h5"
    datacube = py4DSTEM.import_file(binfactor=config.binning, filepath=filepath_data)
    basics_01 = get_basic01(datacube,dirpath2=filepath_basics_01,overwrite=False,show_image=False)
    if Path(filepath_data_ADF).exists():
        print("Loading ADF...")
        defocused_cbed = py4DSTEM.import_file(filepath=filepath_data_ADF)
        im_adf=defocused_cbed.data
    else:
        im_adf=basics_01.tree('annular_dark_field')
    data = datacube.data
    # Clip extreme outliers. Sometimes, detectors can generate hot pixels which
    # interfere with downstream processing and should be handled.
    lo = np.nanpercentile(data, 0.00001)
    hi = np.nanpercentile(data, 99.99999)
    data = np.clip(data, lo, hi)
    datacube.data = data
    # Two toggles can be found in the config. Smooth represents whether to apply
    # smoothing on the diffraction patterns of adjecent pixels according to a 2D Gaussian kernel 
    # and soberfilter whether to apply a soberfilter.
    if config.smooth:
        from py4dstem_radial.smoothing import smoothingkernel
        datacube.data = smoothingkernel(datacube)
    if config.soberfilter:
        from py4dstem_radial.smoothing import smoothingkernelsobel
        datacube.data = smoothingkernelsobel(datacube)
    else:
        dp = datacube[24,25]
        dp_mean,dp_max = datacube.get_dp_mean(),datacube.get_dp_max()
        show(dp)
        show([dp_mean])
        show([dp_max])
    #%%
    #################################################################################################
    ### # ----- Determining vacuum probe for cross-correlation. -----
    #################################################################################################
    # We can then generate/load in the probe kernel and evaluate its shape, size and 
    # general quality. The quality of this probe directly influence the quality
    # of disk detection through cross-correlation. 
    mask, ROIx, ROIy = lowest_intensity_roi(im_adf)
    if overrideROI:
        mask = np.zeros(datacube.Rshape,dtype=bool)
        mask[ROIRange[0]:ROIRange[1],ROIRange[0]:ROIRange[1]] = 1
    # This wrapper gets the vacuum probe according to the method outlined in py4DSTEM. 
    probe = get_vacuum_probe(
    samplename=samplename,
    datacube=datacube,
    mask=mask,
    dirpath2=dirpath2,
    overwrite=True,
    basics_01=basics_01,
    show_image=True
    )
    # Use scan shape as dummy mask. This is necessary for the particle mask wrapper
    scan_shape = datacube.data.shape[:2]
    dummy_a_map = np.zeros(scan_shape, dtype=float)
    dummy_strain_std_map = np.zeros(scan_shape, dtype=float)
    # Align ADF to scan shape consistent with later use. 
    im_adf = match_adf_shape(im_adf, dummy_a_map)
    particle_mask_early, _, _, _ = get_particle_mask(sample,im_adf,dummy_a_map,dummy_strain_std_map,dirpath2,otsucutoff=sample.otsucutoff)
    #%%
    #################################################################################################
    ### # ----- Performing disk detection with cross-correlation. -----
    #################################################################################################
    # Load in the disk detection parameters as a dictionary to ensure compatibility
    # with py4DSTEM and AutoDisk wrappers.  
    detect_params = asdict(detect_config)
    autodisk_params = asdict(autodiskdetect_config)
    #%% Perform diagnostic detection to visually check the quality of disk detection. 
    # By default, six pixels at different positions are selected. 
    disks_selected = disk_detection_diagnostic(
        datacube=datacube,
        image=basics_01.tree('annular_dark_field'),
        probe=probe,
        detect_params=detect_params,
        detector_func=detect_braggpeaks_py4dstem,
        points=sample.diagnostic_points_override,
        showpoints=True,
        showclean=True,
    )
    # Detect all Bragg Disks for every pixel in the supplied image using the 
    # detection parameters loaded above and defined in the config file. You can
    # modify detector func to use a different disk detection approach. Make sure
    # to return a py4DSTEM braggvectors object so that downstream processing remains functional.
    braggpeaks = get_braggpeaks(
        samplename=samplename,
        datacube=datacube,
        probe=probe,
        detect_params=detect_params,
        dirpath2=dirpath2,
        overwrite=True,
        detector_func=detect_braggpeaks_py4dstem,
    )   
    #%%
    #################################################################################################
    ### # ----- Calibrating origin and elliptical distortions. -----
    #################################################################################################
    #%% Calibrate Bragg peak positions using py4DSTEMs inbuilt functions. This wrapper
    # calls both, with qint being the distance in pixels from the origin for the elliptical fit
    # and q_ell_range the thickness of the fitting region in pixels.
    braggpeaks = calibrate_braggpeaks(
    braggpeaks,
    probe=datacube.get_probe_size(probe.probe),
    qint=sample.q_int,
    q_ell_range=sample.q_ell_range,
    binning=config.binning,
    sampling=config.sampling,
    pixelsize=pixelsize,
    overwrite=True,
    show_result=True,
    )
    #%%
    #################################################################################################
    ### # ----- Mean Bragg Vector Map generation -----
    #################################################################################################
    # We then calculate the calibrated 2D histograms of these braggpeaks.
    # These 2D histograms are known as Bragg Vector Maps. 
    bvm_r = braggpeaks.histogram(mode='raw',sampling=config.sampling)
    bvm_c = braggpeaks.histogram(mode='cal',sampling=config.sampling)
    L = 25
    x, y = bvm_c.origin
    x0, xf = np.round([x-L, x+L]).astype(int)
    y0, yf = np.round([y-L, y+L]).astype(int)
    # print(braggpeaks.calstate)
    fig_r,ax_r=show([bvm_r,bvm_r[x0:xf, y0:yf]],vmax=0.999,returnfig=True)
    fig_r.set_size_inches(11, 5)
    for ax in ax_r.flat:
        ax.tick_params(axis="both", labelsize=22)
    fig_c,ax_c=show([bvm_c,bvm_c[x0:xf, y0:yf]],vmax=0.999,returnfig=True)
    fig_c.set_size_inches(11, 5)
    for ax in ax_c.flat:
        ax.tick_params(axis="both", labelsize=22)
    fig_z,ax_z=show([bvm_c],vmax=0.9995,returnfig=True)
    ax_z.tick_params(axis="both", labelsize=22)
    #%%
    #################################################################################################
    ### # ----- Per pixel Bragg Vector Map generation -----
    #################################################################################################
    # Generation of reference diffraction pattern and crystal.
    crystal, figCryst, axCryst = reference_crystal_diagnostic(braggpeaks,latpar,config.atom_num, config.q_range,)
    # Now we compute the Bragg Vector Maps for each pixel indivdually, and take 
    # their numerical sum to compare this to the bragg vector map as calculated
    # with py4DSTEM. We can supply a mask to this function to limit BVM calculations.
    per_pixel_bvms, mean_bvm, dims, pixelsize, origin, dim_units = compute_per_pixel_bvm(braggpeaks,mode='cal',sampling=config.sampling,fit_mask=particle_mask_early,)
    # This is to show a quick diagnostic of the BVM of a single pixel. 
    bvm_single, figBVM, figBVMComparison = bvm_diagnostic(
    per_pixel_bvms,
    mean_bvm,
    bvm_c,
    braggpeaks,
    dims,
    dim_units,
    points=sample.diagnostic_points_override,
    )
    rxs, rys, colors = get_diagnostic_points(per_pixel_bvms.shape[:2],override=sample.diagnostic_points_override,)
    i_fixed = rxs[0]
    j_fixed = rys[0]
    bvm_single = per_pixel_bvms[i_fixed, j_fixed, :, :]
    #%%
    #################################################################################################
    ### # ----- Radial profile and lattice parameter calculations from BVMs -----
    #################################################################################################
    # From these BVMs, we can compute a radial profile which we can use for
    # peak detection and strain calculations for each indivdiual pixel. 
    # We then generate an array with the same shape as our BVMs and calculate the
    # absolute distance |q| from the origin for each pixel position. 
    qx, qy, q_r = radial_coordinates(dims)
    # We can then calculate the radial profiles of the mean and a specific pixel 
    # to check the radial profiling script. 
    q_mean, I_mean = radial_profile(mean_bvm, q_r,num_bins=config.num_bins,k_broadening=config.k_broadening,normalize='max',q_range=config.q_range)
    q_single, I_single = radial_profile(bvm_single, q_r,num_bins=config.num_bins,k_broadening=config.k_broadening,normalize='max',q_range=config.q_range)
    # If desired, we can run a diagnostic function to see how our individual pixel compares
    # to the mean. Logically, peaks should be different in intensity and can vary in position. In polycrystalline materials,
    # some peaks may be absent with respect to the mean. 
    radialdiagnostic(q_single,I_single,q_mean,I_mean,q_range=config.q_range,col='tab:red',i_fixed=i_fixed,j_fixed=j_fixed)
    # We then calculate the average lattice parameter based on the mean BVM.
    a_mean, a_sigma_mean = fit_lattice_parameter_1D(
        q_mean,
        I_mean,
        config.hkllist,
        a_ref_fixed=latpar,
        a_bounds=config.a_bounds,
        latpar=sample.lattice_parameter
    )
    print("Reference a from mean pattern:", a_mean)
    # Now, after computing our lattice parameter from the radial profile of the mean BVM and that of an
    # individual pixel, we can extend the same analysis to the full image. 
    a_map, strain_std_map, nonzero_mask = (calculate_lattice_parameter_maps(per_pixel_bvms=per_pixel_bvms,q_r=q_r,hkllist=config.hkllist,npeaks=config.npeaks,q_range=config.q_range
                                            , a_bounds=config.a_bounds,k_broadening=config.k_broadening,a_ref_fixed=a_mean))
    # We also check the profiles that we actually have fitted and used to calculate
    # the lattice parameter for specific pixels, giving us an important diagnostic.
    # These are the same points as shown in the diagnostic disk detection step. 
    fitting_diagnostic(
        per_pixel_bvms,
        q_r,
        config.hkllist,
        config.npeaks,
        config.q_range,
        config.a_bounds,
        config.num_bins,
        config.k_broadening,
        latpar,
        a_ref_fixed=a_mean,
        points=sample.diagnostic_points_override,
    )
    #%%
    #################################################################################################
    ### # ----- Strain map plotting and dataframe creation -----
    #################################################################################################
    # Generate strain maps with various interesting features. 
    im_adf = match_adf_shape(im_adf, a_map)
    mask_particle, a_map_masked, strain_std_map_masked, thresh_adf = get_particle_mask(sample, im_adf, a_map, strain_std_map, dirpath2,otsucutoff=sample.otsucutoff)
    particle_mask, a_map_masked, strain_std_map_masked = get_clean_particle_mask(sample, a_map_masked, strain_std_map, dirpath2)
    a_map_higherror = np.where(strain_std_map > config.errorrange * 0.8, a_map_masked, np.nan)
    clean_particle_mask, outer_shell_mask, inner_particle_mask, a_outer_map_masked, inner_particle = get_particle_regions(sample, particle_mask, a_map_masked, strain_std_map, config.shell_thickness, dirpath2)
    high_error_inner_mask, a_inner_higherror_map = get_high_error_inner_mask(sample, inner_particle_mask, strain_std_map, a_map_masked, config.errorrange, dirpath2)
    # Map data to dataframe so numerical data is available. 
    a_map_data = map_to_dataframe(a_map_masked)
    a_map_data_coordinates=map_to_dataframe_withcoordinates(a_map_masked)
    a_error_data=map_to_dataframe(strain_std_map_masked)
    a_map_higherrordata = map_to_dataframe(a_map_higherror)
    a_inner_higherror_data = map_to_dataframe(a_inner_higherror_map)
    a_outer_data = map_to_dataframe(a_outer_map_masked)
    a_inner_data = map_to_dataframe(inner_particle)
    a_errorcompshelldata = map_to_dataframe(np.where(strain_std_map < config.errorrange * 0.8, a_outer_map_masked, np.nan))
    a_errorcompcoredata = map_to_dataframe(np.where(strain_std_map < config.errorrange * 0.8, inner_particle, np.nan))
    #%%
    #################################################################################################
    ### # ----- Strain map visualization -----
    #################################################################################################
    # We use a mapping function to visualize the various attributes that we computed.
    figAMap=mapfunction(a_map_masked,bounds=config.a_bounds,l1="Estimated lattice parameter (Å)",title=f"Lattice parameter map ({samplename})",plotpoints=True,points=sample.diagnostic_points_override,cmbins=config.histmapbins)
    figStrainMap=mapfunction(straincalculator(a_map_masked,a_ref=config.a_ref),bounds=(config.histrange[0],config.histrange[1]),l1="Estimated Strain (%)",title=f"Strain map ({samplename})",cmbins=config.histmapbins)
    figStrainStd = mapfunction(strain_std_map_masked,bounds=(0,config.errorrange),cmap='Reds',l1="Strain error (%)", title=f"Strain error map ({samplename})",cmbins=config.histmapbins)
    figStrainMapShell = mapfunction(straincalculator(a_map_masked,a_ref=config.a_ref),bounds=(config.histrange[0],config.histrange[1]),l1="Estimated Strain (%)",title=f"Strain shell map ({samplename})",shell=True,cmbins=config.histmapbins,outer_shell_mask=outer_shell_mask)
    #%%
    #################################################################################################
    ### # ----- Histogram visualization -----
    #################################################################################################
    figStrainHist,histoall = imagehistogram(straincalculator(inner_particle,a_ref=config.a_ref),bounds=(config.histrange[0],config.histrange[1]),binhist=config.histmapbins)
    figStrainStdHist,stdhistoall = imagehistogram(strain_std_map_masked,bounds=(0,config.errorrange),binhist=config.histmapbins)
    figStrainShellHist,strainhistoshellall,strainhistoshellshell = imagehistogram(straincalculator(inner_particle,a_ref=config.a_ref),straincalculator(a_outer_map_masked,a_ref=config.a_ref),bounds=(config.histrange[0],config.histrange[1]),shell=True,binhist=config.histmapbins)
    #%%
    #################################################################################################
    ### # ----- Data saving -----
    #################################################################################################
    save_strain_data(
    output_config=output_config,
    output_dir=dirpath2 + output_config.output_dir,
    samplename=samplename,
    a_map_data=a_map_data,
    a_map_data_coordinates=a_map_data_coordinates,
    a_map_higherrordata=a_map_higherrordata,
    a_inner_higherror_data=a_inner_higherror_data,
    a_outer_data=a_outer_data,
    a_inner_data=a_inner_data,
    a_error_data=a_error_data,
    a_errorcompshelldata=a_errorcompshelldata,
    a_errorcompcoredata=a_errorcompcoredata,
    )
    #%%
    #################################################################################################
    ### # ----- Figure saving -----
    #################################################################################################
    figures = {
        "StrainMap": figStrainMap,
        "StrainStdMap": figStrainStd,
        "StrainHist": figStrainHist,
        "StrainStdHist": figStrainStdHist,
        "StrainMapShell": figStrainMapShell,
        "StrainShellHist": figStrainShellHist,
    }
    
    save_strain_figures(
        output_config=output_config,
        output_dir=dirpath2 + output_config.output_dir,
        samplename=samplename,
        figures=figures,
        binning=config.binning,
        max_num_peaks=detect_params["maxNumPeaks"],
    )
    #%%
    #################################################################################################
    ### # ----- Save configurations -----
    #################################################################################################
    config = AnalysisConfig()
    detect_config = DiskDetectionConfig()
    output_config = OutputConfig()
    saveconfig = {
        "AnalysisConfig": asdict(config),
        "DiskDetectionConfig": asdict(detect_config),
        "OutputConfig": asdict(output_config),
        "SampleConfig": asdict(sample),
    }
    
    # Save configurations to a txt file to allow for easy copying back into python. 
    with open(f'{dirpath2}{output_config.output_dir}{samplename}_config.txt', "w", encoding="utf-8") as f:
        f.write(pformat(saveconfig, sort_dicts=False))
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.close('all')