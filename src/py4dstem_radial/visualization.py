"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import py4DSTEM
from .masks import pixel_aligned_boundaries
from py4DSTEM import show
from scipy.ndimage import label
from .fitting import (
    estimate_a_for_pixel_from_bvm,
)
from py4dstem_radial.selection import get_diagnostic_points
from py4dstem_radial.fitting import simulate_radial_profile_1D
def runcolumndiagnostic(
    stack:np.ndarray #[2048,52,52]
    ):
    """
    Plot the minimum and maximum values of diffraction-pattern columns.
    
    Parameters:
        stack (np.ndarray): Data array containing diffraction patterns, with
            the scan dimensions in the first two axes.
    
    Function purpose:
        Calculate the minimum and maximum values along the scan dimensions
        for each remaining column and plot the results to help identify
        problematic data, such as oversaturated pixels or unusually large values.
    
    Returns:
        None: The function generates diagnostic figures without returning a value.
    """
    col_max = stack.max(axis=(0,1))
    col_min = stack.min(axis=(0,1))
    figMax=plt.figure(figsize=(6,5))
    axMax = figMax.add_axes([1,1,1,1])
    axMax.plot(col_max)
    axMax.set_title("Column max over scan")
    axMax.set_ylim(0,np.max(col_max))
    figMin=plt.figure(figsize=(6,5))
    axMin = figMin.add_axes([1,1,1,1])
    axMin.plot(col_min)
    axMin.set_title("Column max over scan")
    axMin.set_ylim(np.min(col_min),0)
    return 
def radialdiagnostic(
        q_single:np.ndarray,
        I_single:np.ndarray,
        q_mean:np.ndarray,
        I_mean:np.ndarray,
        q_range:tuple=(0.2,1.2),
        col:str='tab:blue',
        i_fixed:int=1,
        j_fixed:int=1,
        )-> mpl.figure:
    """
   Plot and compare a single pixel's radial profile with the mean radial profile.

   Parameters:
       q_single (np.ndarray): Radial reciprocal-space coordinates for the selected pixel.
       I_single (np.ndarray): Intensity profile for the selected pixel.
       q_mean (np.ndarray): Radial reciprocal-space coordinates for the mean BVM.
       I_mean (np.ndarray): Intensity profile for the mean BVM.
       q_range (tuple): Minimum and maximum radial distances to display.
       col (str): Plot color parameter. Defaults to 'tab:blue'.
       i_fixed (int): First scan coordinate used in the pixel label.
       j_fixed (int): Second scan coordinate used in the pixel label.

   Function purpose:
       Normalize the single-pixel and mean radial profiles by their respective
       maximum intensities and display them together over the specified
       reciprocal-space range.

   Returns:
       mpl.figure: Figure containing the radial-profile comparison.
   """
    figComp = plt.figure(figsize=(9,5))
    axComp = figComp.add_axes([1,1,1,1])
    axComp.fill_between(x=q_mean,y1=I_mean/I_mean.max(), label='Mean', lw=2, ls='-',facecolor='tab:red',edgecolor='r',alpha=0.45)
    axComp.plot(q_single, I_single / I_single.max(), label=f'Pixel ({i_fixed}, {j_fixed})', lw=4, ls='--',color='g')
    axComp.set_xlim(q_range[0],q_range[1])
    axComp.set_xlabel("Radial distance $|q|$ (Å$^{-1}$)")
    axComp.set_ylabel("Normalized intensity (a.u.)")
    # axComp.set_title("Radial profile comparison")
    axComp.legend(frameon=True)
    return figComp
def mapfunction(
        map_masked,
        bounds:tuple,
        cmap='RdBu_r',
        cmbins=25,
        l1='default',
        title='default',
        shell:bool=False,
        plotpoints:str=False,
        points=None,
        outer_shell_mask:np.ndarray=1, 
        )-> mpl.figure:
    """
     Display a two-dimensional map with a colorbar and optional diagnostic overlays.
    
     Parameters:
         map_masked (np.ndarray): Two-dimensional array containing the map data.
         bounds (tuple): Minimum and maximum values used for the color scale.
         cmap (str): Matplotlib colormap name. Defaults to 'RdBu_r'.
         cmbins (int): Number of discrete levels in the colormap.
         l1 (str): Label displayed on the colorbar.
         title (str): Figure title parameter.
         shell (bool): If True, draw the boundary of the outer shell mask.
         plotpoints (str or bool): If truthy, plot diagnostic points on the map.
         points (sequence or None): Optional diagnostic points supplied as
             (x, y, color) entries.
         outer_shell_mask (np.ndarray): Mask used to determine and draw shell
             boundaries when shell is True.
    
     Function purpose:
         Plot the input map using the specified colormap and bounds. Optionally
         overlay diagnostic points and draw the boundaries of the outer shell
         using pixel-aligned segments.
    
     Returns:
         mpl.figure: Figure containing the plotted map and colorbar.
     """
    # scalebar = ScaleBar(0.8, "nm", length_fraction=0.25,width_fraction=0.03,location='lower right',font_properties={"size":18})
    rxs, rys, colors = get_diagnostic_points(
    map_masked.shape,
    override=points,
    )
    if shell == True:
        figMap = plt.figure(figsize=(8,8))
        axMap = figMap.add_axes([1,1,1,1])
        imMap = axMap.imshow(
            map_masked,
            origin='upper',
            interpolation='nearest',
            cmap=plt.cm.get_cmap(cmap, lut=cmbins),
            vmin=bounds[0],
            vmax=bounds[1]
        )
        # axMap.add_artist(scalebar)
        figMap.colorbar(imMap, ax=axMap, fraction=0.044, pad=0.04, label=l1)
        # axMap.set_title(title)
        # axMap.set_xlabel('X position')
        # axMap.set_ylabel('Y position')
        axMap.tick_params(axis='both', which='both', length=0, 
               labelbottom=False, labelleft=False)
        labels, num = label(outer_shell_mask)
        segments = pixel_aligned_boundaries(outer_shell_mask)
        for (x0, y0), (x1, y1) in segments:
            axMap.plot(
                [x0 - 0.5, x1 - 0.5],
                [y0 - 0.5, y1 - 0.5],
                color='yellow',
                linewidth=1.5
            )
        if plotpoints:
            for i, j, color in zip(rxs, rys, colors):
                axMap.scatter(x=i, y=j, color=color)
        return figMap
    else:
        figMap = plt.figure(figsize=(8,8))
        axMap = figMap.add_axes([1,1,1,1])
        imMap = axMap.imshow(
            map_masked,
            origin='upper',
            interpolation='nearest',
            cmap=plt.cm.get_cmap(cmap, lut=cmbins),
            vmin=bounds[0],
            vmax=bounds[1]
        )
        # axMap.add_artist(scalebar)
        figMap.colorbar(imMap, ax=axMap, fraction=0.044, pad=0.04, label=l1)
        # axMap.set_title(title)
        # axMap.set_xlabel('X position')
        # axMap.set_ylabel('Y position')
        axMap.tick_params(axis='both', which='both', length=0, 
               labelbottom=False, labelleft=False)
        if plotpoints:
            for i, j, color in zip(rxs, rys, colors):
                axMap.scatter(x=j, y=i, color=color)
        return figMap
def imagehistogram(
        map_masked:np.ndarray,
        map_masked_outer:np.ndarray=[0,0],
        bounds:tuple=(-7,7),
        binhist:int=24,
        shell:str='False',
        n:int=30,
        )-> mpl.figure:
    """
   Calculate and plot a histogram of values in a two-dimensional map.

   Parameters:
       map_masked (np.ndarray): Map data used to calculate the histogram.
       map_masked_outer (np.ndarray): Optional map data restricted to the outer
           shell region.
       bounds (tuple): Minimum and maximum values shown on the histogram.
       binhist (int): Number of points used to define the histogram bin edges.
       shell (str or bool): If True, plot the total and outer-shell distributions
           separately.
       n (int): Additional parameter retained in the function signature.

   Function purpose:
       Calculate the mean and standard deviation of the map data, select a
       histogram color based on the mean, and plot the distribution. If shell
       is True, compare the full map with the outer-shell data.

   Returns:
       tuple: If shell is True, return the figure, full-map histogram results,
       and outer-shell histogram results. Otherwise, return the figure and
       histogram results for the full map.
   """
    # Automatically select colors
    vmin,vmax=bounds[0],bounds[1]
    norm=mcolors.Normalize(vmin=vmin,vmax=vmax)
    avg=np.nanmean(map_masked.flatten())
    cmap=plt.cm.get_cmap('inferno',30)
    color_for_hist=cmap(norm(avg))
    bin_edges=np.linspace(bounds[0],bounds[1],binhist)
    if shell == True:
        figHist=plt.figure(figsize=(9,5))
        axHist = figHist.add_axes([1,1,1,1])
        l1=('Total strain' + ' = ' + str(round(np.nanmean(map_masked.flatten()),1)) +str(' $\pm$ ') 
        + str(round(np.nanstd(map_masked.flatten()),1)) + ' %')
        l2=('Shell strain' + ' = ' + str(round(np.nanmean(map_masked_outer.flatten()),1)) +str(' $\pm$ ') 
        + str(round(np.nanstd(map_masked_outer.flatten()),1)) + ' %')
        nshellall = axHist.hist(map_masked.flatten(), bins=bin_edges, alpha=0.4,edgecolor='black',color=color_for_hist, label=l1)
        nshell = axHist.hist(map_masked_outer.flatten(), bins=bin_edges, alpha=1,edgecolor='black',color=color_for_hist, label=l2)
        axHist.set_xlabel("Strain (%)")
        axHist.set_xlim(bounds[0],bounds[1])
        axHist.set_ylim(0,np.max(nshellall[0])*1.1)
        axHist.set_ylabel("Counts")
        axHist.legend(fontsize=20)
        return figHist, nshellall, nshell
    else:
        figHist = plt.figure(figsize=(9,5))
        axHist = figHist.add_axes([1,1,1,1])
        l1=('Average' + ' = ' + str(round(np.nanmean(map_masked.flatten()),1)) +str(' $\pm$ ') 
                            + str(round(np.nanstd(map_masked.flatten()),1)) + ' %')
        n = axHist.hist(x=map_masked.flatten(),bins=bin_edges,linewidth=0.5,
                        edgecolor='black',alpha=0.8,color=color_for_hist,label=l1)
        axHist.legend(fontsize=20)
        axHist.set_ylim(0,np.max(n[0])*1.1)
        axHist.set_xlim(bounds[0],bounds[1])
        axHist.set_ylabel('Counts')
        axHist.set_xlabel('Deviation (%)')
        return figHist,n
def fittingdiagnostic(
        per_pixel_bvms:np.ndarray,
        pixel:tuple,
        q_r:np.ndarray,
        hkllist:list,
        npeaks:int=5,
        q_range:tuple[float,float]=(0.2, 1.2),
        a_bounds:tuple[float,float]=(3.615, 4.0782),
        num_bins:int=150,
        k_broadening:float=0.005,
        latpar:float=4.0782,
        a_ref_fixed:float=None,
        col:str='tab:blue',
        )-> mpl.figure:
    """
   Plot the radial profile and fitted diffraction peaks for a selected pixel.

   Parameters:
       per_pixel_bvms (np.ndarray): Bragg Vector Maps calculated for individual scan pixels.
       pixel (tuple): Scan coordinates (i, j) of the pixel to analyze.
       q_r (np.ndarray): Radial reciprocal-space distances for the BVM.
       hkllist (list): List of sums of squared Miller indices for the reference lattice.
       npeaks (int): Maximum number of peaks considered during fitting.
       q_range (tuple[float, float]): Reciprocal-space range used for the radial profile.
       a_bounds (tuple[float, float]): Lower and upper bounds for the lattice parameter.
       num_bins (int): Number of radial bins.
       k_broadening (float): Gaussian broadening width of the radial profile.
       latpar (float): Reference lattice parameter described in the diagnostic plot.
       a_ref_fixed (float or None): Optional fixed reference lattice parameter passed
           to the fitting function.
       col (str): Color used to plot the fitted and measured profiles.

   Function purpose:
       Estimate the lattice parameter for the selected pixel, retrieve the
       fitted profile, simulate the fitted radial intensity, and plot it
       alongside the measured radial profile and theoretical reflection positions.

   Returns:
       mpl.figure: Figure containing the radial profile and fitting results.
   """
    best_a, sigma_a,details = estimate_a_for_pixel_from_bvm(
            per_pixel_bvms[pixel[0],pixel[1]], q_r, 
            hkllist, npeaks=npeaks, q_range=q_range,
            a_bounds=a_bounds, return_details=True,
            num_bins=num_bins,k_broadening=k_broadening,a_ref_fixed=a_ref_fixed)
    
    print("Pixel", (pixel[0],pixel[1]), "estimated a =", best_a)
    q_centers = details['q_centers']; I_profile = details['I_profile']
    # If we detect no peaks at this stage, the function ends here and we return
    # nans. 
    # Here we plot the fitted pattern.
    popt=details['popt']
    a_fit = popt[0]
    k_fit = popt[1]
    b0_fit, b1_fit, b2_fit = popt[2:5]
    int_fit = np.atleast_1d(popt[5:])
    I_fit = simulate_radial_profile_1D(
        q_centers,
        a_fit,
        hkllist,
        k_fit,
        int_fit,
        b0_fit,
        b1_fit,
        b2_fit
    )
    # We define important attributes of the peaks. 
    figFitProfile = plt.figure(figsize=(9,5))
    axFitProfile = figFitProfile.add_axes([1,1,1,1])
    theo_q = np.sqrt(np.atleast_1d(np.asarray(hkllist, dtype=float)))/best_a
    for n, t in enumerate(theo_q):
        if n == 0:
            axFitProfile.axvline(t, color='k', alpha=0.4,label='Fitted lattice')
        else:
            axFitProfile.axvline(t, color='k', alpha=0.4)
    axFitProfile.plot(q_centers,I_fit/np.max(I_fit),color=col,lw=5,label='Fitted profile',marker='',linestyle='--')
    axFitProfile.plot(q_centers, I_profile / (I_profile.max() if I_profile.max()>0 else 1), label='Radial profile',color=col,lw=4,alpha=0.6)
    axFitProfile.set_xlim(q_range)
    axFitProfile.set_xlabel("Radial distance $|q|$ (Å$^{-1}$)"); plt.ylabel("Normalized intensity (a.u.)")
    # axFitProfile.set_title(f"Radial profile of pixel ({pixel[0]},{pixel[1]}) and \ntheoretical refections of a={best_a:.4f} Å"+'\n'+f"Average a = {latpar:.4f}")
    axFitProfile.legend(frameon=True)
    # figFitProfile.patch.set_linewidth(10)
    # figFitProfile.patch.set_edgecolor(col)
    return figFitProfile

def disk_detection_diagnostic(
    datacube,
    image,
    probe,
    detect_params,
    detector_func,
    autodiskparams=None,
    points=None,
    vmin=0.6,
    vmax=1.0,
    scaling='log',
    showpoints=False,
    showclean=False,
    mask=None,
):
    """
   Run Bragg disk detection at selected scan positions and visualize the results.

   Parameters:
       datacube (py4DSTEM.DataCube): Input 4D-STEM datacube.
       image (np.ndarray): Navigation or virtual image used to determine the
           diagnostic image shape and optionally display selected points.
       probe: Probe object required by the detector function.
       detect_params (dict): Parameters passed to the disk detection function.
       detector_func (callable): Function used to detect Bragg disks.
       autodiskparams (object or None): Optional additional parameters for
           automatic disk detection.
       points (sequence or None): Optional diagnostic points supplied as
           (x, y, color) entries.
       vmin (float): Minimum display intensity.
       vmax (float): Maximum display intensity.
       scaling (str): Intensity scaling used for visualization.
       showpoints (bool): If True, display the diagnostic positions on the image.
       showclean (bool): If True, display the selected diffraction patterns
           without detected disk overlays.
       mask (np.ndarray or None): Optional mask passed to the detector function.

   Function purpose:
       Select diagnostic scan positions, run the specified Bragg disk detector,
       and optionally display the selected points, diffraction patterns, and
       detected disk positions.

   Returns:
       list: Detected Bragg peak objects returned by detector_func for the
       selected scan positions.
   """
    rxs, rys, colors = get_diagnostic_points(
    image.shape,
    override=points,
    )
    if autodiskparams is not None:
        disks_selected = detector_func(
        datacube,
        probe,
        detect_params,
        data=(rxs, rys),
        autodiskparams=autodiskparams,
        mask=mask,
    )
    else:
        disks_selected = detector_func(
            datacube,
            probe,
            detect_params,
            data=(rxs, rys),
        )
    # overlay points on ADF / navigation image
    if showpoints == True:
        py4DSTEM.visualize.show_points(
            image,   
            x=rxs,
            y=rys,
            pointcolor=colors,
            figsize=(8, 8),
            scalebar=True,
        )
    if showclean == True:
        py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: datacube[rxs[i], rys[i], :, :],
            H=2, W=3, axsize=(5, 5),
            vmin=0.6, vmax=1, scaling='log',
            get_bordercolor=lambda i: colors[i],
        )
    # show diffraction patterns with detected disks. Swap axis when using py4dstem/autodisk. 
    if detector_func.__name__ == "detect_braggpeaks_py4dstem":
        py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: datacube[rxs[i], rys[i], :, :],
            H=2, W=3, axsize=(5, 5),
            vmin=vmin, vmax=vmax, scaling=scaling,
            get_bordercolor=lambda i: colors[i],
            get_x=lambda i: disks_selected[i].data['qx'],
            get_y=lambda i: disks_selected[i].data['qy'],
            get_pointcolors=lambda i: colors[i],
            open_circles=True, scale=700,
        )
    else:
        py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: datacube[rxs[i], rys[i], :, :],
            H=2, W=3, axsize=(5, 5),
            vmin=vmin, vmax=vmax, scaling=scaling,
            get_bordercolor=lambda i: colors[i],
            get_x=lambda i: disks_selected[i].data['qx'],
            get_y=lambda i: disks_selected[i].data['qy'],
            get_pointcolors=lambda i: colors[i],
            open_circles=True, scale=700,
        )
    return disks_selected
def show_vacuum_probe(
    datacube,
    probe,
    basics_01=None,
    mask=None,
    show_image=False,
    vmin=0,
    vmax=7.2e3,
):
    """
     Visualize the vacuum probe, its estimated size, and its probe kernel.
    
     Parameters:
         datacube (py4DSTEM.DataCube): Datacube used to estimate the probe size.
         probe: Vacuum probe object containing probe.probe and probe.kernel.
         basics_01 (py4DSTEM.DataCube or None): Optional basic01 datacube containing
             the annular dark-field virtual image.
         mask (np.ndarray or None): Optional mask applied when displaying the image.
         show_image (bool): If True, display the annular dark-field image with the mask.
         vmin (float): Minimum intensity displayed for the probe.
         vmax (float): Maximum intensity displayed for the probe.
    
     Function purpose:
         Display the vacuum probe, mark its estimated center and semiangle,
         display its kernel, and optionally show the masked annular dark-field image.
    
     Returns:
         tuple: Probe semiangle, horizontal probe-center coordinate, and vertical
         probe-center coordinate.
     """

    alpha_pr, qx0_pr, qy0_pr = datacube.get_probe_size(probe.probe)

    fig1, ax1 = show(
    probe.probe,
    scaling='none',
    intensity_range='absolute',
    vmin=vmin,
    vmax=vmax,
    returnfig=True,
    )
    
    # ax1.set_xlabel(ax1.get_xlabel(), fontsize=14)
    # ax1.set_ylabel(ax1.get_ylabel(), fontsize=14)
    
    # ax1.tick_params(axis='both', labelsize=12)

    fig, ax = show(
    probe.probe,
    scaling='none',
    intensity_range='absolute',
    vmin=vmin,
    vmax=vmax,
    returnfig=True,
    circle={
        'center': (qx0_pr, qy0_pr),
        'R': alpha_pr,
        'alpha': 0.3,
        'fill': True,
    },
    )
    
    # ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    # ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    
    # ax.tick_params(axis='both', labelsize=12)

    figs,axs=py4DSTEM.visualize.show_kernel(
        probe.kernel,
        R=24,
        L=24,
        W=1,
        returnfig=True,
    )
       # Left panel
    axs[0].tick_params(labelsize=18)
    axs[0].xaxis.label.set_size(22)
    axs[0].yaxis.label.set_size(22)
    
    # Right panel
    axs[1].tick_params(labelsize=18)
    axs[1].xaxis.label.set_size(22)
    axs[1].yaxis.label.set_size(22)
    figs.set_size_inches(13, 5)
    if show_image:
        show(
            basics_01.tree('annular_dark_field'),
            mask=~mask,
            mask_alpha=0.667,
            mask_color='r',
            # vmin=0,
            # vmax=7.2e3,
            scalebar=False,
        )
    return alpha_pr, qx0_pr, qy0_pr

def reference_crystal_diagnostic(
    braggpeaks,
    lattice_parameter,
    atom_num,
    q_range,
):
    """
   Generate an FCC reference crystal and compare its reflections with measured Bragg peaks.

   Parameters:
       braggpeaks: Measured Bragg peaks used for comparison in the scattering plot.
       lattice_parameter (float): Lattice parameter of the reference crystal.
       atom_num: Atomic number or atom specification passed to the Crystal constructor.
       q_range (tuple): Minimum and maximum reciprocal-space values used for
           calculating and plotting the scattering intensity.

   Function purpose:
       Construct an FCC crystal, calculate its structure factors over the
       specified reciprocal-space range, and plot its scattering intensity
       alongside the measured Bragg peaks.

   Returns:
       tuple: The reference crystal object, Matplotlib figure, and Matplotlib axes.
   """

    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    crystal = py4DSTEM.process.diffraction.Crystal(
        pos,
        atom_num,
        lattice_parameter,
    )

    crystal.calculate_structure_factors(
        q_range[1],
        return_intensities=True,
    )

    fig, ax = crystal.plot_scattering_intensity(
        bragg_peaks=braggpeaks,
        bragg_k_power=1.0,
        remove_origin=True,
        returnfig=True,
        k_min=q_range[0],
        k_max=q_range[1],
    )

    ax.set_xlim(q_range[0], q_range[1])
    fig.set_size_inches(8, 5)
    ax.set_title("Comparison to reference reflections")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_xlabel("Radial distance $|q|$ (Å$^{-1}$)")
    return crystal, fig, ax

def bvm_diagnostic(
    per_pixel_bvms,
    mean_bvm,
    bvm_c,
    braggpeaks,
    dims,
    dim_units,
    points=None,
):
    """
   Compare the py4DSTEM BVM with the mean per-pixel BVM and inspect a selected pixel.

   Parameters:
       per_pixel_bvms (np.ndarray): Bragg Vector Maps calculated for individual scan pixels.
       mean_bvm (np.ndarray): Mean Bragg Vector Map calculated from the per-pixel maps.
       bvm_c: Bragg Vector Map object generated by py4DSTEM.
       braggpeaks: Bragg peak data containing calibrated peak positions for each scan pixel.
       dims (tuple): Coordinate arrays defining the BVM axes.
       dim_units (str): Units used for the BVM coordinate axes.
       points (sequence or None): Optional diagnostic points supplied as
           (x, y, color) entries.

   Function purpose:
       Display the selected pixel's Bragg peaks over the mean BVM and compare
       the py4DSTEM BVM with the mean per-pixel BVM and their absolute difference.

   Returns:
       tuple: The selected pixel's BVM, the figure showing its peaks over the
       mean BVM, and the figure comparing the BVM datasets.
   """
    rxs, rys, colors = get_diagnostic_points(
    per_pixel_bvms.shape[:2],
    override=points,
    )
    
    i_fixed = rxs[0]
    j_fixed = rys[0]
    dimx, dimy = dims

    bvm_single = per_pixel_bvms[i_fixed, j_fixed, :, :]
    vdata = braggpeaks.cal[i_fixed, j_fixed].data
    qx, qy = vdata["qx"], vdata["qy"]

    figBVM = plt.figure(figsize=(8, 8))
    axBVM = figBVM.add_axes([1, 1, 1, 1])

    imBVM = axBVM.imshow(
        np.log1p(mean_bvm.T),
        origin="lower",
        extent=[dimx[0], dimx[-1], dimy[0], dimy[-1]],
        cmap="magma",
        aspect="equal",
    )

    axBVM.scatter(
        qx,
        qy,
        s=80,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.5,
        label=f"Pixel ({i_fixed}, {j_fixed}) peaks",
    )

    axBVM.set_title(
        f"Local BVM peaks over mean BVM "
        f"(pixel {i_fixed}, {j_fixed})"
    )
    axBVM.set_xlabel(f"qx [{dim_units}]")
    axBVM.set_ylabel(f"qy [{dim_units}]")
    axBVM.legend()

    figBVM.colorbar(
        imBVM,
        ax=axBVM,
        fraction=0.046,
        pad=0.04,
        label="log(Intensity)",
    )

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(
        np.log1p(bvm_c.data.T),
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axs[0].set_title("py4DSTEM BVM")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(
        np.log1p(mean_bvm.T),
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axs[1].set_title("Mean per pixel BVM")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    diff = np.abs(
        np.log1p(bvm_c.data) -
        np.log1p(mean_bvm)
    )

    im2 = axs[2].imshow(
        np.log1p(diff.T),
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axs[2].set_title(
        "Difference\n(py4DSTEM - mean per pixel BVM)"
    )
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlim(0, bvm_c.data.shape[0] - 1)
        ax.set_ylim(0, bvm_c.data.shape[1] - 1)

    return bvm_single, figBVM, fig
def fitting_diagnostic(
    per_pixel_bvms,
    q_r,
    hkllist,
    npeaks,
    q_range,
    a_bounds,
    num_bins,
    k_broadening,
    latpar,
    a_ref_fixed=None,
    points=None,
):
    """
   Generate fitting diagnostic plots for selected scan positions.

   Parameters:
       per_pixel_bvms (np.ndarray): Bragg Vector Maps calculated for individual scan pixels.
       q_r (np.ndarray): Radial reciprocal-space distances for the BVM.
       hkllist (list): List of sums of squared Miller indices for the reference lattice.
       npeaks (int): Maximum number of peaks considered during fitting.
       q_range (tuple): Reciprocal-space range used for the radial profile.
       a_bounds (tuple): Lower and upper bounds for the lattice parameter.
       num_bins (int): Number of radial bins.
       k_broadening (float): Gaussian broadening width of the radial profile.
       latpar (float): Reference lattice parameter used in the diagnostic plots.
       a_ref_fixed (float or None): Optional fixed reference lattice parameter.
       points (sequence or None): Optional diagnostic points supplied as
           (x, y, color) entries.

   Function purpose:
       Select diagnostic scan positions, generate a fitting diagnostic plot
       for each selected pixel using fittingdiagnostic, and collect the figures.

   Returns:
       list: Figures generated for the selected diagnostic pixels.
   """
    rxs, rys, colors = get_diagnostic_points(
        per_pixel_bvms.shape[:2],
        override=points,
    )

    figures = []

    for i, j, color in zip(rxs, rys, colors):
        fig = fittingdiagnostic(
            per_pixel_bvms,
            (i, j),
            q_r,
            hkllist,
            npeaks,
            q_range,
            a_bounds,
            num_bins,
            k_broadening,
            latpar,
            a_ref_fixed=a_ref_fixed,
            col=color,
        )
        figures.append(fig)

    return figures