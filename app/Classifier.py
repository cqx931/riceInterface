import cv2
import numpy as np
from Debug import debug
from image_processing import *
import json
from utils import *

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
    return self.results

  def get_layers(self):
    output = []
    for layer in self.layers:   
      layer["data"] = json.dumps(layer["data"], cls=NumpyArrayEncoder)
      output.append(layer)
    return output
  
  # process image
  def process(self, img_raw):
    
    img_out = img_raw.copy()
    print(len(img_out.shape))
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    
    img_out = equalize_image(img_out)
    # debug.push_image(img_out, "equalized image")
    
    # ---------------------------------------------- #
    # outer contour
    # ---------------------------------------------- #
    
    # brighter image for outer contour detection 
    step_name = "outer_contour"
    img_lighter = img_out.copy() # equalizeLight(img_out, 20) # todo implement
    img_otsu = otsu_thresholding(img_raw)
    # debug.push_image(img_otsu, "otsu")
    outer_contour = findMaxContour(img_otsu)
    rect = cv2.minAreaRect(outer_contour)
    # convert contour to json object
    step_name = "island_circles"
    self.layers.append({
      "name": step_name,
      "type": "contours",
      "data": json.dumps([outer_contour], cls=NumpyArrayEncoder)
    })
  
    
    # ---------------------------------------------- #
    # inner islands
    # ---------------------------------------------- #
    
    step_name = "island_contours"
    
    img_masked = getMaskedImage(img_raw, outer_contour)
    # preprocess image
    # _, img_binary = cv2.threshold(img_masked, 125, 255, cv2.THRESH_BINARY)
    img_binary = threshold_and_mask(img_masked, exclude_percent=10)
  
    # debug.push_image(img_binary, "binary image")

    # img_out = img_masked.copy()
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)

    inner_contours = getInnerIslands(img_binary, outer_contour)
    self.layers.append({
      "name": step_name,
      "type": "contours",
      "data": json.dumps(inner_contours, cls=NumpyArrayEncoder)
    })
    
    # ---------------------------------------------- #
    # crack lines
    # ---------------------------------------------- #
      
    longest_line, lines = detect_trace(img_binary)
    
    angle = np.rad2deg(getOrientation(outer_contour, img_out)) 
    # print("angle", angle)

     # draw the longest line on the image
    if longest_line is not None:
      x1, y1, x2, y2 = longest_line[0]
      # cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 10)

    lines_vert = None
    lines_hori = None
    
    if lines is not None:  
      lines = filter_lines_by_distance(lines, min_distance=300)
      lines_vert = filter_lines_by_angle(lines, angle, tolerance=30)
      lines_hori = filter_lines_by_angle(lines, angle-90, tolerance=30)
    
    # vertical lines
    if lines_vert is not None:
      for line in lines_vert:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_out, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    # horizontal lines
    if lines_hori is not None:
      for line in lines_hori:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 10)
        # drawAxis(img_out, [x1, y1], [x2, y2], (255, 255, 0), 5)
    
    intersection_points = None
    if lines_hori is not None and lines_vert is not None:
      intersection_points = find_intersection_points(lines_vert, lines_hori)
      if intersection_points is not None:
        # draw intersection points as circles
        for point in intersection_points:
          cv2.circle(img_out, (int(point[0]), int(point[1])), 50, (125, 255, 255), 10)
    
    # self.layers.append({
    #   "name": "crack_lines",
    #   "type": "lines",
    #   "data": json.dumps(lines_a, cls=NumpyArrayEncoder)
    # })
    
    # ---------------------------------------------- #
    # embrio
    # ---------------------------------------------- #
    
    img_out, embrio_circle = fault_check(img_out, outer_contour)
    if embrio_circle is not None:
      (center, radius) = embrio_circle
      cv2.circle(img_out, center, radius, (0, 0, 255), -1)
      self.layers.append({
        "name": "embrio_circle",
        "type": "circle",
        "data": embrio_circle
      })
      
    # ---------------------------------------------- #
    # side faults 
    # ---------------------------------------------- #

    # ---------------------------------------------- #
    # draw things 
    # ---------------------------------------------- #

    # draw island detections
    circles = []
    for c in inner_contours:
      center, radius = cv2.minEnclosingCircle(c)
      circles.append([center, radius])
      cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 3)
      #cv2.drawContours(img_out, [c], 0, (0,255,0), 2)
    # draw rice contour
    cv2.drawContours(img_out, [outer_contour], 0, (255,0,0), 2)
    # draw bounding box
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(img_out,[box],0,(0,0,255),10)
    
    step_name = "bounding_box"
    self.layers.append({
      "name": step_name,
      "type": "contours",
      "data": json.dumps([box], cls=NumpyArrayEncoder)
    })
    
    step_name = "island_circles"
    self.layers.append({
      "name": step_name,
      "type": "circle",
      "data": json.dumps(circles, cls=NumpyArrayEncoder)
    })

    with open('layers.json', 'w', encoding='utf-8') as f:
      json.dump(self.layers, f, ensure_ascii=False, cls=NumpyArrayEncoder)
    
    return img_out

  # opencv write text "test" on image
  def test(self, img_input):
    self.results = { 'test': 'test'}
    self.img_out = img_input.copy()
    cv2.putText(self.img_out, "test", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 10)
    return self.img_out
