# 4D-STEM_RadialFit

4DSTEM_Radial is a volume-averaged scalar strain analysis method that enables high-throughput nanobeam electron diffraction (NBED) measurements of polycrystalline nanomaterials.

## Installation:
It is recommended to create a fresh conda environment, download the repository and perform a editable pip install. 

Example for 4DSTEM_Radial v0.1.0:

### 1. Create and activate the environment

Open Anaconda Prompt and run:

	conda create --name 4DSTEMRadial python=3.12.11
	conda activate 4DSTEMRadial

### 2. Install the dependencies

Install the required packages using Conda:

	conda install -c conda-forge numpy=1.26.4
	conda install -c conda-forge scipy=1.15.3
	conda install -c conda-forge matplotlib=3.10.6
	conda install -c conda-forge py4DSTEM=0.14.18
	conda install -c conda-forge pandas=3.0.6
	conda install -c conda-forge pywin32

The pywin32 dependency is intended for Windows environments. Install it only if required by your workflow. If you have a Nvidia GPU available, you can significantly speed up disk detection operations. For GPU accelerated analyses, also install the following packages:

	conda install -c conda-forge cupy=13.6.0
	conda install -c conda-forge numba=0.61.2


### 3. Download the repository

Clone this repository from GitHub:

	git clone <repository-url>
	cd 4DSTEM_RadialAnalysis

Alternatively, download the repository as a ZIP file and navigate to the extracted project directory.

### 4. Install 4DSTEM_Radial in editable mode

With the Conda environment activated and the terminal positioned at the project root, run:

	pip install -e .

Editable installation allows changes to the source code to be used without reinstalling the package after every edit.
## How to use

In the accompanying python script, 4DSTEMRadialAnalysis.py, we showcase its use for a complex polycrystalline Pt-shell AuCu-core nanoparticle. It documents the workflow and explains how to use the analysis functions provided in the source modules.

Three files are key to manage this package. The first is config.py, which documents the configuration of analysis, disk detection and the general sample configuration parameters. The next is sample.py, which has information on each indivdual sample you wish to analyze in the same format as the sampleconfig dataclass documented in config.py. Finally, the 4DSTEMRadialAnalysis.py file documents how to use the most important functions. 

## Acknowledgement and citations 
For more details on the volume-average scalar strain analysis method developed by Just Jonasse, the reader is referred to: "4D-STEM Strain Mapping of Polycrystalline Core-shell Nanoparticles" article at https://doi.org/10.26434/chemrxiv.15008328/v1. For more details on the py4DSTEM module, the reader is referred to the respective paper: Savitzky, B. H., Pelz, P. M., Barnard, E. S., et al. (2021). py4DSTEM: A Software Package for Four-Dimensional Scanning Transmission Electron Microscopy Data Analysis. Microscopy and Microanalysis, 27(4), 712–745.
