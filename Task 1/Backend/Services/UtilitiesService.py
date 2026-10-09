import numpy as np

def ConvolveImage(image: np.array, kernel: np.array):
    kernel_height, kernel_width = kernel.shape

    # Calculate padding based on kernel dimensions
    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height - 1 - pad_top

    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width - 1 - pad_left

    # RGB
    if image.ndim == 3:
        img_height, img_width, channels = image.shape
        output_image = np.zeros_like(image, dtype=np.float64)

        for c in range(channels):
            channel = image[:, :, c]

            padded_channel = np.pad(
                channel,
                ((pad_top, pad_bottom),
                 (pad_left, pad_right)),
                mode="constant"
            )

            for i in range(img_height):
                for j in range(img_width):

                    region = padded_channel[
                        i:i + kernel_height,
                        j:j + kernel_width
                    ]

                    output_image[i, j, c] = np.sum(region * kernel)

        return output_image

    # Grayscale
    else:
        img_height, img_width = image.shape

        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom),
             (pad_left, pad_right)),
            mode="constant"
        )

        output_image = np.zeros_like(image, dtype=np.float64)

        for i in range(img_height):
            for j in range(img_width):

                region = padded_image[
                    i:i + kernel_height,
                    j:j + kernel_width
                ]

                output_image[i, j] = np.sum(region * kernel)

        return output_image


def NormalizeImage(img: np.array):
  max_val = img.max()
  min_val = img.min()
  if max_val == min_val:
    return np.zeros_like(img, dtype=np.uint8)
  normalized_image = 255 * ((img - min_val) / (max_val - min_val))
  return normalized_image.astype("uint8")

def TransformIntoGrayScale(rgbImg: np.array, mode: str):

    if mode == "weighted":
        gray_image = (0.2989 * rgbImg[:, :, 0] + 0.5870 * rgbImg[:, :, 1] + 0.1140 * rgbImg[:, :, 2])

    elif mode == "average":
        gray_image = (rgbImg[:, :, 0] + rgbImg[:, :, 1] + rgbImg[:, :, 2]) / 3

    else:
        raise ValueError("Mode must be either 'weighted' or 'average'")

    return gray_image.astype("uint8")