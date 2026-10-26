"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""
def straincalculator(a, a_ref=3.92):
    """
    Calculate the percentage strain relative to a reference lattice parameter.
    
    Parameters:
        a (float or np.ndarray): 
            Measured lattice parameter.
        a_ref (float): 
            Reference lattice parameter. Defaults to 3.92.
    
    Function purpose:
        Calculate the relative difference between the measured and reference
        lattice parameters and express it as a percentage.
    
    Returns:
        float or np.ndarray: 
            Strain expressed as a percentage.
    """
    return ((a - a_ref) / a_ref) * 100