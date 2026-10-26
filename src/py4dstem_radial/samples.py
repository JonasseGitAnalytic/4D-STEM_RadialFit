"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

from .config import SampleConfig
samples = {
    "Particle8": SampleConfig(
        name="Particle8",
        lattice_parameter=3.73,
        pixel_size=0.00725,
        roi_range=(1, 6),
        q_int = 22,
        mask_directory = None,
        overrideROI = False,
        diagnostic_points_override = [(10,30,'r'),(12,26,'limegreen'),(16,22,'c'),(20,18,'g'),(24,14,'orange'),(30,10,'violet')]
    ),
}
