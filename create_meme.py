"""
Create the final statistics meme by assembling all four panels.
Assembles the original image, stippled image, block letter, and masked image
into a professional four-panel layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white"
) -> None:
    """
    Create a four-panel statistics meme demonstrating selection bias.
    
    Parameters
    ----------
    original_img : np.ndarray
        Original grayscale image (2D array, values in [0, 1])
    stipple_img : np.ndarray
        Stippled version of the image (2D array, values in [0, 1])
    block_letter_img : np.ndarray
        Block letter mask image (2D array, values in [0, 1])
    masked_stipple_img : np.ndarray
        Masked stippled image (2D array, values in [0, 1])
    output_path : str
        Path where the meme image will be saved (e.g., "statistics_meme.png")
    dpi : int
        Resolution for the output image (default 150)
    background_color : str
        Background color for the meme (default "white")
        Can be "white", "pink", "lightgray", etc.
    
    Returns
    -------
    None
        Saves the meme image to output_path
    """
    # Ensure all images have the same dimensions
    h, w = original_img.shape
    
    # Resize images if they don't match (shouldn't happen, but just in case)
    if stipple_img.shape != (h, w):
        from PIL import Image
        stipple_pil = Image.fromarray((stipple_img * 255).astype(np.uint8))
        stipple_pil = stipple_pil.resize((w, h), Image.Resampling.LANCZOS)
        stipple_img = np.array(stipple_pil, dtype=np.float32) / 255.0
    
    if block_letter_img.shape != (h, w):
        from PIL import Image
        block_letter_pil = Image.fromarray((block_letter_img * 255).astype(np.uint8))
        block_letter_pil = block_letter_pil.resize((w, h), Image.Resampling.LANCZOS)
        block_letter_img = np.array(block_letter_pil, dtype=np.float32) / 255.0
    
    if masked_stipple_img.shape != (h, w):
        from PIL import Image
        masked_pil = Image.fromarray((masked_stipple_img * 255).astype(np.uint8))
        masked_pil = masked_pil.resize((w, h), Image.Resampling.LANCZOS)
        masked_stipple_img = np.array(masked_pil, dtype=np.float32) / 255.0
    
    # Create figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor(background_color)
    
    # Panel labels
    labels = ["Reality", "Your Model", "Selection Bias", "Estimate"]
    images = [original_img, stipple_img, block_letter_img, masked_stipple_img]
    
    # Display each panel
    for i, (ax, img, label) in enumerate(zip(axes, images, labels)):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save the meme
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
    # print(f"Saved statistics meme to: {output_path}")
    plt.close()
    
    
    return None

