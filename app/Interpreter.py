import cv2
from image_processing import *
from Debug import debug

class Interpreter:

  result = {}
  img_raw = None
  # mg_out = None

  def __init__(self, img_raw):
    self.img_raw = img_raw
    
  def test(self, img_input):
    self.result = { 'test': 'test'}
    self.img_out = img_input.copy()
    # opencv write text "test" on image
    
    img_otsu = otsu_thresholding(self.img_out)
    
    debug.push_image(img_otsu, "otsu")
    
    cv2.putText(self.img_out, "test", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 10)
    return self.img_out

  def get_results(self):
    return self.result

  def get_image(self):
    return self.img_out
