# src/py4dstem_radial/basic01.py
"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

from pathlib import Path

import py4DSTEM


def get_basic01(
    datacube,
    dirpath2,
    overwrite=False,
    show_image=False,
):
    """
   Load an existing basic01 file or generate and save one from a 4D-STEM datacube.

   Parameters:
       
   datacube : py4DSTEM.DataCube
       Input 4D-STEM datacube. Its data are used to calculate the mean diffraction
       pattern, determine probe geometry, and generate an annular virtual image.
   dirpath2 : str or pathlib.Path
       Path to the basic01 file to load or create. Despite the original description,
       the current implementation uses this value directly as the file path rather
       than appending a sample name.
   overwrite : bool, default=False
       If True, regenerate and save the basic01 file even when the path already
       exists. If False and the path exists, load it instead.
   show_image : bool, default=False
       If True, display the mean diffraction pattern using logarithmic scaling.

   Function purpose:
       Prepare the basic01 datacube by calculating the mean diffraction pattern,
       estimating probe size and center, configuring an annular detector, generating
       an annular dark-field virtual image, and saving the datacube.

   Returns:
       py4DSTEM.DataCube
       The datacube loaded from the existing or newly saved file.
   """

    dirpath2 = Path(dirpath2)
    filepath_basics_01 = dirpath2 

    # ------------------------------------------------------------
    # 1. Load existing file
    # ------------------------------------------------------------
    if filepath_basics_01.exists() and not overwrite:

        print("Loading existing basic01 file:")
        print(f"    {filepath_basics_01}")

        return py4DSTEM.read(filepath_basics_01)

    # ------------------------------------------------------------
    # 2. Generate basic01
    # ------------------------------------------------------------
    print("basic01 file not found. Generating it...")

    dp_mean = datacube.get_dp_mean()

    if show_image:
        from py4DSTEM.visualize import show

        show(
            dp_mean,
            scaling="log",
        )

    # Determine probe size and diffraction center
    probe_semiangle, probe_qx0, probe_qy0 = datacube.get_probe_size(
        dp_mean.data,
    )

    center = (probe_qx0, probe_qy0)

    # Annular detector geometry
    r_inner = probe_semiangle * 3
    r_outer = probe_semiangle * 15
    radii = (r_inner, r_outer)

    # Set virtual detector
    datacube.position_detector(
        mode="annular",
        geometry=(center, radii),
    )

    # Generate virtual image
    datacube.get_virtual_image(
        mode="annulus",
        geometry=(center, radii),
        name="annular_dark_field",
    )

    # ------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------
    print("Saving basic01 file:")
    print(f"    {filepath_basics_01}")

    py4DSTEM.save(
        filepath_basics_01,
        datacube,
        tree=None,
        mode="o",
    )

    # ------------------------------------------------------------
    # 4. Return generated file
    # ------------------------------------------------------------
    return py4DSTEM.read(filepath_basics_01)