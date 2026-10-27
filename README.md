# 4D-STEM_RadialFit

4DSTEM_Radial is a volume-averaged scalar strain analysis method that enables high-throughput nanobeam electron diffraction (NBED) measurements of polycrystalline nanomaterials.

## Installation:
It is recommended to create a fresh conda environment, download the repository and perform a editable pip install.

Example for `4DSTEM_Radial v0.1.0`:

### 1. Download the repository

Clone this repository from GitHub:

	git clone <repository-url>
	cd 4DSTEM_RadialAnalysis

Alternatively, download the repository as a ZIP file and navigate to the extracted project directory.

### 2. Install the dependencies
<!-- collapsed all requirements into environment.yml-->
Install the required packages using Conda:

	conda env create -f environment.yml
	conda activate 4DSTEMRadial

The pywin32 dependency is intended for Windows environments. Install it only if required by your workflow. If you have a Nvidia GPU available, you can significantly speed up disk detection operations. For GPU accelerated analyses, also install the following packages:

	conda install -c conda-forge cupy=13.6.0
	conda install -c conda-forge numba=0.61.2

### 3. Install 4DSTEM_Radial in editable mode
<!-- can you not just publish it as a pip package?-->
With the Conda environment activated and the terminal positioned at the project root, run:

	pip install -e .

Editable installation allows changes to the source code to be used without reinstalling the package after every edit.

## How to use

In the accompanying (interactive) python script, `4DSTEMRadialAnalysis.py`, we showcase its use for a complex polycrystalline Pt-shell AuCu-core nanoparticle. It documents the workflow and explains how to use the analysis functions provided in the source modules.
<!-- Tell me where these files are and how to use them in an example folder.-->
Three files are key to manage this package. The first is `config.py`, which documents the configuration of analysis, disk detection and the general sample configuration parameters. The next is `sample.py`, which has information on each indivdual sample you wish to analyze in the same format as the `sampleconfig` dataclass documented in `config.py`. Finally, the `4DSTEMRadialAnalysis.py` file documents how to use the most important functions. 

Open the files `4DSTEMRadialAnalysis.py`, `config.py` and `sample.py` in an IDE (e.g. spyder). Three things need to be changed before the script can be run:
1. Modify dirpath and dirpath2 to where your data is located and where you would like your analyses to end up, respectively.
2. Add the samplename to the sample_to_analyze list (or if using the example dataset, keep it 'Particle8'). Also add your new sample to `sample.py`.
3. In `config.py`, navigate to the dataclass `Outputconfig` and set the output_dir. This is tacked on dirpath2, to keep all analyses results together. 
4. Optionally, disable GPU-accelerated disk detection in `config.py` by changing CUDA in the `DiskDetectionConfig` dataclass to False.
5. Run the script in your IDE.

## Acknowledgement and citations 
For more details on the volume-average scalar strain analysis method developed by Just Jonasse, the reader is referred to: "4D-STEM Strain Mapping of Polycrystalline Core-shell Nanoparticles" article at https://doi.org/10.26434/chemrxiv.15008328/v1. 
For more details on the py4DSTEM package, the reader is referred to the respective paper: Savitzky, B. H., Pelz, P. M., Barnard, E. S., et al. (2021). py4DSTEM: A Software Package for Four-Dimensional Scanning Transmission Electron Microscopy Data Analysis. Microscopy and Microanalysis, 27(4), 712–745. https://doi.org/10.1017/S1431927621000477
For more details on the AutoDisk detection method, the reader is referred to the respective paper: Wang, S.; Eldred, T. B.; Smith, J. G.; Gao, W. AutoDisk: Automated Diffraction Processing and Strain Mapping in 4D-STEM. Ultramicroscopy 2022, 236, 113513. https://doi.org/10.1016/j.ultramic.2022.113513
