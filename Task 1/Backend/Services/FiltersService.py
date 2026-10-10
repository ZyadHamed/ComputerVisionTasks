from UtilitiesService import ConvolveImage
import numpy as np


def low_pass_filter(img: np.array, filter_type, kernel_size=3, sigma=1.0):
    if filter_type == "Average":
     
     kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
     result = np.round((ConvolveImage(img, kernel))).astype(np.uint8)

    elif filter_type == "Gaussian":
     
     half = kernel_size // 2
     offsets = np.arange(-half, half + 1) #the position of each column relative to the center
     x, y = np.meshgrid(offsets, offsets) #np.meshgrid(a, b) builds a grid with one column for every entry in a and one row for every entry in b
     kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
     gaussian_kernel = kernel/kernel.sum() 
     #the kernel numbers add up to a number x > 1 
     #if it's used as is, the image would be x times brighter 
     #dividing every entry by the sum scales them so they total 1
     result = np.round(ConvolveImage(img, gaussian_kernel).astype(np.uint8))

    elif filter_type == "Median":
            pad = kernel_size // 2
            # pad rows and columns only; (0, 0) leaves the channel axis alone
            pad_width = [(pad, pad), (pad, pad)] + [(0, 0)] * (img.ndim - 2)
            padded = np.pad(img, pad_width, mode='reflect')

            result = np.zeros_like(img)
            height, width = img.shape[:2]

            for i in range(height):
                for j in range(width):
                    window = padded[i:i + kernel_size, j:j + kernel_size]
                    result[i, j] = np.median(window, axis=(0, 1))

    else:
       raise ValueError("Undefined filter type")
    
    return result
