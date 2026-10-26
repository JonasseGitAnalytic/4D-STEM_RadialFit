"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""
import os
import py4DSTEM
from py4dstem_radial.visualization import show_vacuum_probe
def get_vacuum_probe(   
    samplename,
    datacube,
    mask,
    dirpath2,
    overwrite=False,
    calibration=None,
    basics_01=None,
    show_image=False,
):
    """
    Load an existing vacuum probe or generate and save a new one.

    Parameters:
        samplename (str): 
            Sample name used in the output filename.
        datacube (py4DSTEM.DataCube): 
            Input 4D-STEM datacube.
        mask (np.ndarray): 
            Region-of-interest mask used to determine the vacuum probe.
        dirpath2 (str or pathlib.Path): 
            Directory where the basic03 file is stored.
        overwrite (bool): 
            If True, regenerate the probe even if a saved file exists.
        calibration: 
            Optional calibration object assigned to the datacube before
            generating the probe.
        basics_01 (py4DSTEM.DataCube or None): 
            Optional basic01 datacube used for
            displaying the annular dark-field image.
        show_image (bool): 
            If True, display the masked annular dark-field image.

    Function purpose:
        Load the probe from an existing basic03 file when possible. If no saved
        probe is available, determine a vacuum probe from the masked region,
        calculate its size and center, generate a sigmoid probe kernel, attach
        the probe to the datacube, save the datacube, and display the probe
        diagnostics.

    Returns:
        probe: The loaded or newly generated vacuum probe.
    """
    filepath_basics_03 = os.path.join(
        dirpath2,
        f"{samplename}_basic03.h5",
    )

    # Load existing probe
    if os.path.exists(filepath_basics_03) and not overwrite:
        print(f"[vacuumprobe] Loading probe from {filepath_basics_03}")

        basics_03 = py4DSTEM.read(filepath_basics_03)

        try:
            probe = basics_03.tree('probe')
        except AssertionError:
            probe = None

        if probe is not None:
            if probe is not None:
                show_vacuum_probe(
                    datacube,
                    probe,
                    basics_01=basics_01,
                    mask=mask,
                    show_image=show_image,
                )
            return probe

        print("[vacuumprobe] No probe found; regenerating.")

    # Generate probe
    print("[vacuumprobe] Determining vacuum probe...")

    if calibration is not None:
        datacube.calibration = calibration

    probe = datacube.get_vacuum_probe(
        ROI=mask,
        align=True,
    )

    alpha_pr, qx0_pr, qy0_pr = datacube.get_probe_size(probe.probe)

    # Prepare probe kernel
    probe.get_kernel(
        mode='sigmoid',
        origin=(qx0_pr, qy0_pr),
        radii=(alpha_pr, 2 * alpha_pr),
    )

    # Attach probe and save it under the 'probe' tree
    datacube.probe = probe

    py4DSTEM.save(
        filepath_basics_03,
        datacube,
        tree=None,
        mode='o',
    )

    print(f"[vacuumprobe] Saved probe to {filepath_basics_03}")

    show_vacuum_probe(
        datacube,
        probe,
        basics_01=basics_01,
        mask=mask,
        show_image=show_image,
    )
    return probe