"""
4dstem_radial version 1.0

@author: Just Pé Jonasse（j.p.jonasse@uu.nl）
"""

import numpy as np


def diagnostic_points(image_shape, fractions=None, colors=None):
    """
  Generate default diagnostic pixels and their plotting colors.

  Parameters:
      image_shape (tuple[int, int]): 
          Image dimensions (nx, ny).
      fractions (sequence or None): 
          Optional normalized (x, y) positions.
          Default positions are used if None.
      colors (sequence or None): 
          Optional plotting colors corresponding to
          the diagnostic positions. Default colors are used if None.

  Function purpose:
      Convert normalized positions to integer pixel coordinates, constrain
      the coordinates to the image boundaries, and associate each position
      with its plotting color.

  Returns:
      tuple[list[int], list[int], list]: 
          The x coordinates, y coordinates, and corresponding plotting colors.
  """
    nx,ny = image_shape

    if fractions is None:
        fractions = (
            (0.30, 0.30),
            (0.45, 0.35),
            (0.60, 0.40),
            (0.70, 0.55),
            (0.60, 0.70),
            (0.40, 0.65),
        )

    if colors is None:
        colors = [
            "r",
            "limegreen",
            "c",
            "g",
            "orange",
            "violet",
        ]

    rxs = np.array([x * nx for x, y in fractions]).round().astype(int)
    rys = np.array([y * ny for x, y in fractions]).round().astype(int)

    rxs = np.clip(rxs, 0, nx - 1)
    rys = np.clip(rys, 0, ny - 1)

    return rxs.tolist(), rys.tolist(), list(colors)


def get_diagnostic_points(image_shape, override=None):
    """
    Return diagnostic points, using overrides when provided.

    Parameters:
        image_shape (tuple[int, int]): 
            Image dimensions passed to diagnostic_points.
        override (sequence or None): 
            Optional sequence of (x, y, color) entries.

    Function purpose:
        If override is provided, separate its entries into x coordinates,
        y coordinates, and colors. Otherwise, generate the default diagnostic
        points for the specified image shape.

    Returns:
        tuple[list, list, list]: 
            The x coordinates, y coordinates, and colors.
    """
    if override is not None:
        rxs = [point[0] for point in override]
        rys = [point[1] for point in override]
        colors = [point[2] for point in override]

        return rxs, rys, colors

    return diagnostic_points(image_shape)