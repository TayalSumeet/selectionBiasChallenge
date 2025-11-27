"""
Step 5: Create a masked stippled image by applying the block letter mask.
This creates the "biased estimate" by systematically removing data points
where the mask is dark (representing selection bias).
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply a mask to a stippled image to create a biased estimate.
    
    Where the mask is dark (below threshold), stipples are removed (set to white).
    Where the mask is light (above threshold), stipples are kept as they are.
    This simulates selection bias by systematically removing data points.
    
    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image as 2D array (height, width) with values in [0, 1]
        Where 0.0 = black dot (stipple), 1.0 = white background
    mask_img : np.ndarray
        Mask image as 2D array (height, width) with values in [0, 1]
        Where 0.0 = black (mask area, remove stipples)
        Where 1.0 = white (keep area, preserve stipples)
    threshold : float
        Threshold value to determine what counts as "part of the mask"
        Pixels below this value are considered part of the mask (default 0.5)
    
    Returns
    -------
    masked_stipple : np.ndarray
        2D numpy array with the same shape as input images
        Stipples are removed (set to 1.0/white) where mask is dark
        Stipples are preserved where mask is light
    """
    # Ensure images have the same shape
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"Image shapes must match: stipple_img {stipple_img.shape} != mask_img {mask_img.shape}"
        )
    
    # Create a copy of the stipple image
    masked_stipple = stipple_img.copy()
    
    # Where mask is dark (below threshold), remove stipples (set to white/1.0)
    # Where mask is light (above threshold), keep stipples as they are
    mask_region = mask_img < threshold
    
    # Remove stipples in the mask region by setting to white
    masked_stipple[mask_region] = 1.0
    
    # Count how many stipples were removed
    removed_count = np.sum((stipple_img == 0.0) & mask_region)
    total_stipples = np.sum(stipple_img == 0.0)
    
    # print(f"Applied mask: removed {removed_count} of {total_stipples} stipples ({100*removed_count/max(total_stipples,1):.1f}%)")
    # print(f"Masked stipple shape: {masked_stipple.shape}")
    
    return masked_stipple

