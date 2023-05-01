# colect here the pre-processing functions
import cv2
# import numpy as np

# otsu thresholding
def otsu_thresholding(image):
  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(image,(5,5),0)
  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return th3