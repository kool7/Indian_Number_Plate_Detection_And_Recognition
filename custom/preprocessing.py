'''
This script perform basic image preprocessing.
'''


import cv2
import os
import numpy as np 
from PIL import Image

def croppedImage(image, count):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.bilateralFilter(gray, 11, 20, 20)
  gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
  gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 90)
  # gray = cv2.Canny(gray, 70, 100)
  pathPng='cropped'
  count += 1
  filename = os.path.join(pathPng,'{}.png'.format(count))
  cv2.imwrite(filename, gray)
  return filename

def imageToarray(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)