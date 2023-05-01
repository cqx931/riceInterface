
# opencv code to test opening an image and displaying it
import cv2
import sys

def test(image_path):    
  # path
  # path = env.str("TEST_IMAGE_PATH")

  # Reading an image in default mode
  image = cv2.imread(image_path)

  # display image in window
  cv2.imshow("test", image)
  cv2.waitKey(0)
