"""
Step 4: Create a block letter matching image dimensions.
Generates a block letter (default "S") that represents the selection bias pattern.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9
) -> np.ndarray:
    """
    Create a block letter matching the image dimensions.
    
    Parameters
    ----------
    height : int
        Height of the output image in pixels
    width : int
        Width of the output image in pixels
    letter : str
        Letter to render (default "S")
    font_size_ratio : float
        Ratio of font size to image size (default 0.9)
        Controls how large the letter appears relative to the image
    
    Returns
    -------
    letter_img : np.ndarray
        2D numpy array (height, width) with values in [0, 1]
        Black letter (0.0) on white background (1.0)
    """
    # Create a white image
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Calculate font size based on image dimensions
    # Use the smaller dimension to ensure the letter fits
    min_dimension = min(height, width)
    font_size = int(min_dimension * font_size_ratio)
    
    # Try to find a suitable bold font
    font = None
    font_paths = [
        # Windows fonts
        "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
        "C:/Windows/Fonts/calibrib.ttf",  # Calibri Bold
        "C:/Windows/Fonts/timesbd.ttf",   # Times Bold
        "C:/Windows/Fonts/impact.ttf",   # Impact (bold by default)
        # macOS fonts
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        # Linux fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    # Try to load a font
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
    
    # If no font found, use default font (may not be bold)
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Get text bounding box to center the letter
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position to center the letter
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]
    
    # Draw the letter in black
    draw.text((x, y), letter, fill=0, font=font)
    
    # Convert to numpy array and normalize to [0, 1]
    # PIL: 0=black, 255=white
    # After normalization: 0.0=black (letter), 1.0=white (background)
    letter_array = np.array(img, dtype=np.float32) / 255.0
    
    # print(f"Created block letter '{letter}' with size {letter_array.shape}")
    return letter_array

