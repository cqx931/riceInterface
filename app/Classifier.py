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

  def get_json_layers(self):
    output = []
    for layer in self.layers:   
      layer["data"] = json.dumps(layer["data"], cls=NumpyArrayEncoder)
      output.append(layer)
    return output
  
  def get_layer_data(self, name):
    for layer in self.layers:
      if layer["name"] == name:
        return layer["data"]
    return []
  
  # process image
  def process(self, img_raw):
    
    img_out = img_raw.copy()
    print(len(img_out.shape))
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    img_out = equalize_image(img_out)
    
    # ---------------------------------------------- #
    # outer contour
    # ---------------------------------------------- #
    
    # brighter image for outer contour detection 
    img_lighter = img_out.copy() # equalizeLight(img_out, 20) # todo implement
    img_otsu = otsu_thresholding(img_raw)
    # debug.push_image(img_otsu, "otsu")
    outer_contour = findMaxContour(img_otsu)
    rect = cv2.minAreaRect(outer_contour)
    self.add_layer("outer_contour", "contours", outer_contour)
    
    # ---------------------------------------------- #
    # inner islands
    # ---------------------------------------------- #
    
    # preprocess image
    img_masked = getMaskedImage(img_raw, outer_contour)
    img_binary = threshold_and_mask(img_masked, exclude_percent=9)  
    img_binary_islands = threshold_and_mask(equalize_image(img_masked), exclude_percent=6)
  
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)

    inner_contours = getInnerIslands(img_binary_islands, outer_contour)
    self.add_layer("island_contours", "contours", inner_contours)
    
    # draw island detections
    island_circles = []
    for c in inner_contours:
      center, radius = cv2.minEnclosingCircle(c)
      island_circles.append([center, radius])
      #cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 3)
        #cv2.drawContours(img_out, [c], 0, (0,255,0), 2)
    self.add_layer("island_circles", "circle", island_circles)
    
    # ---------------------------------------------- #
    # crack lines
    # ---------------------------------------------- #
    
    angle = np.rad2deg(getOrientation(outer_contour, img_out)) 

    # vertical line
    lines_vert = detect_trace(img_binary, threshold=50, minLineLength=800, maxLineGap=100)
    if lines_vert is not None:
      lines_vert = filter_lines_by_distance(lines_vert, min_distance=200)
      lines_vert = filter_lines_by_angle(lines_vert, angle, tolerance=20)
      self.add_layer("lines_vertical", "lines", lines_vert)
      for line in lines_vert:
        x1, y1, x2, y2 = line[0]
        drawAxis(img_out, (x1, y1), (x2, y2), (255, 255, 0), 5)
        cv2.line(img_out, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    # horizontal line
    lines_hori = detect_trace(img_binary, threshold=80, minLineLength=500, maxLineGap=800)
    if lines_hori is not None:
      lines_hori = filter_lines_by_distance(lines_hori, min_distance=300)
      lines_hori = filter_lines_by_angle(lines_hori, angle-90, tolerance=30)
      self.add_layer("lines_horizontal", "lines", lines_hori)
      for line in lines_hori:
        x1, y1, x2, y2 = line[0]
        drawAxis(img_out, (x1, y1), (x2, y2), (0, 255, 255), 5)
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 10)

    
    # ---------------------------------------------- #
    # lines intersections
    # ---------------------------------------------- #
    
    intersection_points = None
    if lines_hori is not None and lines_vert is not None:
      intersection_points = find_intersection_points(lines_vert, lines_hori)
      if intersection_points is not None:
        # draw intersection points as circles
        self.add_layer("intersections", "circles", intersection_points)
        for point in intersection_points:
          cv2.circle(img_out, (int(point[0]), int(point[1])), 50, (125, 255, 255), 10)
    
    # ---------------------------------------------- #
    # circles lines intersections
    # ---------------------------------------------- #
    
    intersecting_circles, non_intersecting_circles = find_circle_line_intersections(lines_hori, island_circles)
    
    if intersecting_circles is not None:
      self.add_layer("intersecting_islands", "circles", intersecting_circles)
      for circle in intersecting_circles:
        center, radius = circle
        # cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 3)
    
    if non_intersecting_circles is not None:
      self.add_layer("non_intersecting_islands", "circles", non_intersecting_circles)
      for circle in non_intersecting_circles:
        center, radius = circle
        cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 4)    
    
    # ---------------------------------------------- #
    # embrio & faults
    # ---------------------------------------------- #
    
    img_out, embrio_circle, circle_faults = embrio_check(img_out, outer_contour)
    if embrio_circle is not None:
      (center, radius) = embrio_circle
      cv2.circle(img_out, center, radius, (0, 0, 255), -1)
      self.add_layer("embrio_circle", "circle", embrio_circle)
    
    # if circle_faults is not None:
    #   for (center, radius) in circle_faults:
    #     cv2.circle(img_out, (int(center[0]),int(center[1])), radius, (255, 0, 0), -1)
    #   self.layers.append({
    #     "name": "circle_faults",
    #     "type": "circles",
    #     "data": circle_faults
    #   })

    # ---------------------------------------------- #
    # draw things 
    # ---------------------------------------------- #

    # draw rice contour
    cv2.drawContours(img_out, [outer_contour], 0, (255,0,0), 2)
    # draw bounding box
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(img_out,[box],0,(0,0,255),10)
    
    self.add_layer("bounding_box", "contours", [box])

    with open('layers.json', 'w', encoding='utf-8') as f:
      json.dump(self.layers, f, ensure_ascii=False, cls=NumpyArrayEncoder)
    
    return img_out

  def add_layer(self, name, type, data):
    self.layers.append({
      "name": name,
      "type": type,
      "data": data
    })
  
  # opencv write text "test" on image
  def test(self, img_input):
    self.results = { 'test': 'test'}
    self.img_out = img_input.copy()
    cv2.putText(self.img_out, "test", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 10)
    return self.img_out

  def clear_layers(self):
    self.layers = []