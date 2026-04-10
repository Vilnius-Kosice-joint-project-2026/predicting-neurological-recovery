import cv2
import random
import numpy as np

# 1
def adjust_brightness_contrast(image):
    """
    Randomly adjusts brightness and contrast within ±0.5 boundary.
    """
    img = image.copy()
    
    # Contrast: 0.5 to 1.5
    alpha = random.uniform(0.5, 1.5)
    
    # Brightness: small shift
    beta = random.uniform(-50, 50)
    
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# 2
def frequency_mask(image):
    """
    Applies frequency masking with up to 50% masking rate.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Mask width up to 50% of image width
    mask_width = int(w * random.uniform(0, 0.5))

    # Random starting point
    x_start = random.randint(0, w - mask_width)

    # Apply mask (set to black)
    img[:, x_start:x_start + mask_width] = 0

    return img

# 3
def gaussian_blur(image):
    """
    Applies Gaussian blur with kernel size between 3 and 7.
    """
    img = image.copy()
    
    # Kernel size must be odd (3,5,7)
    k = random.choice([3, 5, 7])
    
    return cv2.GaussianBlur(img, (k, k), 0)

# 4 
def time_shift(image):
    """
    Shifts the image horizontally by up to ±20%.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Calculate shift (±20% of width)
    shift = int(w * random.uniform(-0.2, 0.2))

    # Shift image horizontally
    shifted = np.roll(img, shift, axis=1)

    return shifted

# 5

def cutout(image):
    """
    Applies cutout by adding up to 8 random holes (max size 45x45).
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Number of holes (1 to 8)
    num_holes = random.randint(1, 8)

    for _ in range(num_holes):
        hole_w = random.randint(10, 45)
        hole_h = random.randint(10, 45)

        # Random position
        x = random.randint(0, w - hole_w)
        y = random.randint(0, h - hole_h)

        # Apply hole (black patch)
        img[y:y + hole_h, x:x + hole_w] = 0

    return img