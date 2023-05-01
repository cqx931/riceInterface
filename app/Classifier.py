import cv2
import numpy as np
from Interpreter import Interpreter

class Classifier:

  # interpreter = Interpreter()
  results = {}
  order = []

  def __init__(self, mode):
    #self.model = mode
    self.mode = mode
  
  # this function will run on main thread with new image every frame if its on exhibition mode
  # on debug mode, it will run 1 time per image and deplay results in bulk? or just 1 time per image
  def run(self, img_raw):
    interpreter = Interpreter(img_raw)
    # this is only for simple test
    # if self.mode == 'test':
    img_out = interpreter.test(img_raw)
      # get test image
    return img_out
  
  def get_results(self):
    return self.results

