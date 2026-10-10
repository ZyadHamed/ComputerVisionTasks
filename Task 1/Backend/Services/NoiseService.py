import numpy as np
import cv2 as cv


def salt_and_pepper(img, p=0.05):

  noise = np.random.random(img.shape[:2])
  pepper_threshold = p/2
  salt_threshold = 1 - p/2

  noisy = img.copy()
  noisy[noise < pepper_threshold] = 0
  noisy[noise > salt_threshold] = 255

  return noisy


def add_noise(img, noise_type,  mean=0, sigma=25, low=-50, high=50, p=0.05):

  img_float = img.astype(np.float32) 

  if noise_type == 'Gaussian':
    noise = np.random.normal(loc=mean, scale=sigma, size=img_float.shape)
  elif noise_type == "Uniform":
    noise = np.random.uniform(low, high, size = img_float.shape)
  elif noise_type == "Salt and Pepper":
    return salt_and_pepper(img,p)
  else:
    raise ValueError("Noise type is not defined")

  noisy_image = img_float + noise
  clipped_image = np.clip(noisy_image,0, 255).astype(np.uint8)
  return clipped_image

img = cv.imread(r'Task 1\Samples\Edges_test.jpg')
noisy = add_noise(img, 'Salt and Pepper', p=0.02)
cv.imshow('noisy', noisy)
cv.waitKey(0)