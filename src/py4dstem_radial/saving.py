"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

from pathlib import Path


def save_dataframe_excel(dataframe, filepath):
    """
    Save a DataFrame to an Excel file.

    Parameters:
        dataframe (pandas.DataFrame): DataFrame to save.
        filepath (str or pathlib.Path): Destination path for the Excel file.

    Function purpose:
        Create the parent directory if it does not exist and save the DataFrame
        without its index.

    Returns:
        None: The function saves the file without returning a value.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(filepath, index=False)


def save_strain_data(
    output_config,
    output_dir,
    samplename,
    a_map_data=None,
    a_map_data_coordinates=None,
    a_map_higherrordata=None,
    a_inner_higherror_data=None,
    histogram_data=None,
    a_outer_data=None,
    a_inner_data=None,
    a_error_data=None,
    a_errorcompshelldata=None,
    a_errorcompcoredata=None,
):
    """
  Save available strain-related data as Excel files.

  Parameters:
      output_config: Configuration object controlling data saving.
      output_dir (str or pathlib.Path): Directory in which to save the files.
      samplename (str): Sample name used in the output filenames.
      a_map_data (pandas.DataFrame or None): Optional map data.
      a_map_data_coordinates (pandas.DataFrame or None): Optional map data with coordinates.
      a_map_higherrordata (pandas.DataFrame or None): Optional high-error map data.
      a_inner_higherror_data (pandas.DataFrame or None): Optional inner high-error data.
      histogram_data (pandas.DataFrame or None): Optional histogram data.
      a_outer_data (pandas.DataFrame or None): Optional outer-region data.
      a_inner_data (pandas.DataFrame or None): Optional inner-region data.
      a_error_data (pandas.DataFrame or None): Optional error data.
      a_errorcompshelldata (pandas.DataFrame or None): Optional shell error-component data.
      a_errorcompcoredata (pandas.DataFrame or None): Optional core error-component data.

  Function purpose:
      Save each supplied DataFrame to a separate Excel file when both
      output_config.save and output_config.save_data are True. Data arguments
      set to None are skipped.

  Returns:
      None: Returns without saving if data saving is disabled.
  """
    if not output_config.save or not output_config.save_data:
        return

    output_dir = Path(output_dir)

    if a_map_data is not None:
        save_dataframe_excel(
            a_map_data,
            output_dir / f"{samplename}_a_map_data.xlsx",
        )
    if a_map_data_coordinates is not None:
        save_dataframe_excel(
            a_map_data_coordinates,
            output_dir / f"{samplename}_a_map_coordinates_data.xlsx",
        )

    if a_map_higherrordata is not None:
        save_dataframe_excel(
            a_map_higherrordata,
            output_dir / f"{samplename}_a_map_higherrordata.xlsx",
        )

    if a_inner_higherror_data is not None:
        save_dataframe_excel(
            a_inner_higherror_data,
            output_dir / f"{samplename}_a_inner_higherror_data.xlsx",
        )

    if a_outer_data is not None:
        save_dataframe_excel(
            a_outer_data,
            output_dir / f"{samplename}_a_outer_data.xlsx",
        )

    if histogram_data is not None:
        save_dataframe_excel(
            histogram_data,
            output_dir / f"{samplename}_histogram_values.xlsx",
        )
    if a_inner_data is not None:
        save_dataframe_excel(
            a_inner_data,
            output_dir / f"{samplename}_a_inner_data.xlsx",
        )
    if a_error_data is not None:
        save_dataframe_excel(
            a_error_data,
            output_dir / f"{samplename}_a_error_data.xlsx",
            )
    if a_errorcompshelldata is not None:
        save_dataframe_excel(
            a_errorcompshelldata,
            output_dir / f"{samplename}_a_errorcompshelldata.xlsx",
            )
    if a_errorcompcoredata is not None:
        save_dataframe_excel(
            a_errorcompcoredata,
            output_dir / f"{samplename}_a_errorcompcoredata.xlsx",
            )
def save_strain_figures(
    output_config,
    output_dir,
    samplename,
    figures,
    binning,
    max_num_peaks,
):
    """
    Save strain-related figures as SVG files.
    
    Parameters:
        output_config: Configuration object controlling figure saving.
        output_dir (str or pathlib.Path): Directory in which to save the figures.
        samplename (str): Sample name used in the output filenames.
        figures (dict): Dictionary mapping figure names to figure objects.
        binning: Binning value included in the output filenames.
        max_num_peaks (int): Maximum number of peaks included in the filenames.
    
    Function purpose:
        Save the supplied figures when both output_config.save and
        output_config.save_figures are True. The output filenames include the
        sample name, binning value, peak count, and figure name.
    
    Returns:
        None: Returns without saving if figure saving is disabled.
    """
    if not output_config.save or not output_config.save_figures:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{samplename}_{binning}_"

    for name, figure in figures.items():
        filepath = output_dir / f"{prefix}{max_num_peaks}{name}.svg"
        figure.savefig(filepath, bbox_inches="tight")