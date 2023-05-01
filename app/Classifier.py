import cv2
import numpy as np
from Debug import debug
from image_processing import *
import json
from json import JSONEncoder

class Classifier:

  # interpreter = Interpreter()
  results = {}
  order = []
  layers = []

  def __init__(self, mode):
    #self.model = mode
    self.mode = mode
  
  # this function will run on main thread with new image every frame if its on exhibition mode
  # on debug mode, it will run 1 time per image and deplay results in bulk? or just 1 time per image
  def run(self, img_raw):
    # this is only for simple test, writes "test" on the image
    if self.mode == 'test':
      img_out = self.test(img_raw)
    else:
      img_out = self.process(img_raw)
    
    return img_out
  
  def get_results(self):
    # numpyData = {"results": self.layers}
    json_object = json.dumps(self.layers, indent = 4) 
    return json_object

  def get_layers(self):
    #numpyData = {"layers": self.layers}
    json_object = json.dumps(self.layers)
    return json_object
  
  # process image
  def process(self, img_raw):
    
    img_out = img_raw.copy()
    print(len(img_out.shape))
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    
    # ---------------------------------------------- #
    # outer contour
    # ---------------------------------------------- #
    
    # brighter image for outer contour detection 
    step_name = "outer_contour"
    img_lighter = img_out # equalizeLight(img_out, 20) # todo implement
    img_otsu = otsu_thresholding(img_lighter)
    outer_contour = findMaxContour(img_otsu)
    rect = cv2.minAreaRect(outer_contour)
    step_name = "island_circles"
    self.layers.append({
      "name": step_name,
      "type": "contour",
      "data": outer_contour
    })
    
    # ---------------------------------------------- #
    # inner islands
    # ---------------------------------------------- #
    
    step_name = "island_contours"
    img_masked = getMaskedImage(img_raw, outer_contour)
    img_out = img_masked.copy()

    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    
    # darker image for island detection
    img_darker = img_out # equalizeLight(img_masked, -10)

    inner_contours = getInnerIslands(img_darker, outer_contour)
    self.layers.append({
      "name": step_name,
      "type": "contour",
      "data": inner_contours
    })
    
    # ---------------------------------------------- #
    # draw things 
    # ---------------------------------------------- #

    # draw island detections
    circles = []
    for c in inner_contours:
      center, radius = cv2.minEnclosingCircle(c)
      circles.append([center, radius])
      cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 3)
      cv2.drawContours(img_out, [c], 0, (255,0,0), 2)
    # draw rice contour
    cv2.drawContours(img_out, [outer_contour], 0, (255,0,0), 2)
    # draw bounding box
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_out,[box],0,(0,0,255),10)
    
    step_name = "bounding_box"
    self.layers.append({
      "name": step_name,
      "type": "contour",
      "data": [box]
    })
    
    step_name = "island_circles"
    self.layers.append({
      "name": step_name,
      "type": "circle",
      "data": circles
    })

    
    return img_out

  # opencv write text "test" on image
  def test(self, img_input):
    self.results = { 'test': 'test'}
    self.img_out = img_input.copy()
    cv2.putText(self.img_out, "test", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 10)
    return self.img_out