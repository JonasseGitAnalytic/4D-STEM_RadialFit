# 4D-STEM_RadialFit

4DSTEM_Radial is a volume-averaged scalar strain analysis method that enables high-throughput nanobeam electron diffraction (NBED) measurements of polycrystalline nanomaterials. In the provided notebook, we showcase its use for a complex polycrystalline Pt-shell AuCu-core nanoparticle. 

Installation:
It is recommended to create a fresh conda environment, download the repository and perform a editable pip install. 

Example for 4DSTEM_Radial v1.0:
1. First, open anaconda prompt.
2. Run the following commands in order:
conda create --name 4DSTEMRadial python=3.12.11
conda activate 4DSTEMRadial
conda install -c conda-forge NumPy=1.26.4
conda install -c conda-forge SciPy=1.15.3
conda install -c conda-forge Matplotlib=3.10.6
conda install -c conda-forge py4DSTEM=0.14.18
conda install -c conda-forge pywin32
conda install -c conda-forge cupy
4. In anaconda prompt, navigate to where you downloaded the repository using cd.
5. Still in your new environment, install the repository through pip:
  pip install -e .

For more details on the procedure developed by Just Jonasse, the reader is referred to: "4D-STEM Strain Mapping of Polycrystalline Core-shell Nanoparticles" article at https://doi.org/10.26434/chemrxiv.15008328/v1. 
