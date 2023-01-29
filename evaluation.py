# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oTXD8vnT6-4BPM1sA1QZFHpw_GSJl1M_
"""

import numpy as np
import sys
import os
import cv2
from skimage.metrics import structural_similarity as ssim

def load_images(dir_path):
  # initialize an empty list to store the images
  images = []
  # iterate through all the files in the directory
  for file_name in os.listdir(dir_path):
    # check if the file is an image
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
      # open the image using the PIL library
      image = cv2.imread(os.path.join(dir_path, file_name))
      # add the image to the list
      images.append(image)
  return images

# Calculate the Peak Signal-to-Noise Ratio (PSNR)
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def evaluate_predictions(model_outputs,ground_truths):
  if model_outputs.shape[0] != ground_truths.shape[0]:
    raise Exception("Number of model outputs is unequal to number of ground truths")
  mses = []
  ssims = []
  psnrs = []
  for i in range(model_outputs.shape[0]):
    model_output = model_outputs[i]
    ground_truth = ground_truths[i]
    # Calculate the mean squared error
    mse = np.mean((ground_truth - model_output) ** 2)
    mses.append(mse)
    #print("Mean Squared Error: ", mse)

    # Calculate the Structural Similarity Index (SSIM)
    ssim_value = ssim(ground_truth, model_output, multichannel=True)
    ssims.append(ssim_value)
    #print("Structural Similarity Index: ", ssim_value)

    psnr_value = psnr(ground_truth, model_output)
    psnrs.append(psnr_value)
    #print("Peak Signal-to-Noise Ratio: ", psnr_value)
  return np.mean(mses), np.mean(ssims), np.mean(psnrs)

def main():
  # Load the model's output and ground truth image
  model_outputs = load_images('./data/model_outputs')
  ground_truths = load_images('./data/ground_truths')

  # Convert images to numpy arrays
  model_outputs = np.array(model_outputs)
  ground_truths = np.array(ground_truths)

  # Evaluate the model
  mean_mse, mean_ssim, mean_psnr = evaluate_predictions(model_outputs, ground_truths)
  print("Mean MSE: "+ str(mean_mse), "Mean SSIM: " + str(mean_ssim), "Mean PSNR: " + str(mean_psnr))

if __name__ == '__main__':
    main()