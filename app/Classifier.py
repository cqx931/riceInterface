import cv2
import numpy as np
from Debug import debug
from image_processing import *
import json
from utils import *
LINES_PERC = 10
ISLAND_PERC = 6

class Classifier:

  # interpreter = Interpreter()
  results = {}
  order = []
  layers = []
  outer_contour = []
  island_circles = []
  inner_contours = []
  lines_hori = []
  lines_half_hori = []
  lines_vert = []
  intersecting_circles = []
  non_intersecting_circles = []
  embrio_circle = []
  intersection_points = []
  horizontal_intersection_points = []
  triangle_faults = []

  rotation_angle = 0
  width = 0
  height = 0

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
    _layers = self.layers.copy()
    for layer in _layers:   
      layer["data"] = json.dumps(layer["data"], cls=NumpyArrayEncoder)
      output.append(layer)
    return output
  
  def get_layer_data(self, name):
    _layers = self.layers.copy()
    for layer in _layers:
      if layer["name"] == name:
        return layer["data"]
    return []
   

  # process image
  def process(self, img_raw):
    img_out = img_raw.copy()
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    img_out = equalize_image(img_out)
    
    # ---------------------------------------------- #
    # outer contour
    # ---------------------------------------------- #
    self.outer_contour = []
    # brighter image for outer contour detection 
    img_lighter = img_out.copy() # equalizeLight(img_out, 20) # todo implement
    img_otsu = otsu_thresholding(img_raw)
    # cv2.imshow("img_otsu", img_otsu)
    # debug.push_image(img_otsu, "otsu")
    outer_contour = findMaxContour(img_otsu)
    if outer_contour is None: # if there is no outer contour, no sense doing anything else
      self.clear_vars()
      self.clear_layers()
      return False

    rice_area = cv2.contourArea(outer_contour)
    print("Rice Area:", rice_area)
    rect = cv2.minAreaRect(outer_contour)
    (x,y), (w,h), a = rect
    if (w > h):
      self.width = int(h)
      self.height = int(w)
    else:
      self.width = int(w)
      self.height = int(h)
    print("Width:",self.width, "Height:", self.height)
    # only check rice_area in stream mode
    if self.mode == 'stream' and (rice_area > MAX_RICE_AREA or rice_area < MIN_RICE_AREA):
      self.clear_vars()
      self.clear_layers()
      return False
    #self.outer_contour = [outer_contour]
    self.add_layer("outer_contour", "contour", [outer_contour]) #(#)#
    self.outer_contour = [outer_contour]
    #print("rice_area", rice_area)

    # ---------------------------------------------- #
    # inner islands
    # ---------------------------------------------- #
    
    # preprocess image
    img_masked = getMaskedImage(img_raw, outer_contour)
    img_binary_lines = threshold_and_mask(img_masked, exclude_percent=LINES_PERC)  
    img_binary_islands = threshold_and_mask(equalize_image(img_masked), exclude_percent=ISLAND_PERC)
    # cv2.imshow("img_binary_islands", img_binary_islands)
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)

    inner_contours = getInnerIslands(img_binary_islands, outer_contour)
    
    #if drawContours is not None:
      # self.add_layer("island_contours", "contours", inner_contours)
    
    # draw island detections
    self.island_circles = [] 
    island_circles = []
    for c in inner_contours:
      center, radius = cv2.minEnclosingCircle(c)
      island_circles.append([center, radius])
      #cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 3)
        #cv2.drawContours(img_out, [c], 0, (0,255,0), 2)
    if len(island_circles) > 0:
      self.add_layer("island_circles", "circles", island_circles)
      self.island_circles = island_circles
    
    # ---------------------------------------------- #
    # crack lines
    # ---------------------------------------------- #
    
    angle = int(np.rad2deg(getOrientation(outer_contour, img_out)))
    self.rotation_angle = 45-angle
    print("Current angle:", angle)
    if(self.rotation_angle > 15 or self.rotation_angle < -15):
      #sendMessage(rotation_angle)
      return True
      # continue when rotation is no longer needed

    # vertical line
    self.lines_vert = []
    lines_vert = detect_trace(img_binary_lines, threshold=VERTICAL_THRESHOLD, minLineLength=VERTICAL_MIN_LINE_LENGTH, maxLineGap=VERTICAL_MAX_LINE_GAP)
    if lines_vert is not None:
      lines_vert = filter_lines_by_distance(lines_vert, min_distance=VERTICAL_MIN_DISTANCE)
      lines_vert = filter_lines_by_angle(lines_vert, angle, tolerance=20)
      self.add_layer("lines_vertical", "lines", lines_vert)
      self.lines_vert = lines_vert 
      for line in lines_vert:
        x1, y1, x2, y2 = line[0]
        # drawAxis(img_out, (x1, y1), (x2, y2), (255, 255, 0), 5)
        cv2.line(img_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # horizontal line
    self.lines_hori = []
    self.lines_half_hori = []
    lines_hori = detect_trace(img_binary_lines, threshold=HORIZONTAL_THRESHOLD, minLineLength=HORIZONTAL_MIN_LINE_LENGTH, maxLineGap=HORIZONTAL_MAX_LINE_GAP)
    if lines_hori is not None:
      lines_hori = filter_lines_by_distance(lines_hori, min_distance=HORIZONTAL_MIN_DISTANCE)
      lines_hori = filter_lines_by_angle(lines_hori, angle-90, tolerance=30)
      lines_hori = filter_lines_by_distance_to_contour(lines_hori, outer_contour, min_distance=MIN_DISTANCE_TO_CONTOUR)
      self.add_layer("lines_horizontal", "lines", lines_hori)
      # further devided horizontal lines to full and half
      if len(lines_hori) >=3:
        lines_hori, lines_half_hori = filter_lines_by_length(lines_hori, self.width/2)
        self.lines_half_hori = lines_half_hori

      self.lines_hori = lines_hori
      
      # for line in lines_hori:
      #   x1, y1, x2, y2 = line[0]
        # drawAxis(img_out, (x1, y1), (x2, y2), (0, 255, 255), 5)
        # cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    # ---------------------------------------------- #
    # lines intersections
    # ---------------------------------------------- #
    
    self.intersection_points = []
    intersection_points = None
    if lines_hori is not None and lines_vert is not None:
      intersection_points = find_intersection_points(lines_vert, lines_hori)
      if intersection_points is not None:
        # draw intersection points as circles
        self.add_layer("intersections", "points", intersection_points)
        self.intersection_points = intersection_points
        for point in intersection_points:
          cv2.circle(img_out, (int(point[0]), int(point[1])), 10, (125, 255, 255), 1)

    # ---------------------------------------------- #
    # horizontal lines intersections
    # ---------------------------------------------- #
    
    self.horizontal_intersection_points = []
    horizontal_intersection_points = None
    if lines_hori is not None and lines_vert is not None:
      horizontal_intersection_points = find_intersection_points(lines_hori, lines_hori)
      if horizontal_intersection_points is not None:
        # draw intersection points as circles
        self.add_layer("horizontal_intersections", "points", horizontal_intersection_points)
        self.horizontal_intersection_points = horizontal_intersection_points
        for point in horizontal_intersection_points:
          cv2.circle(img_out, (int(point[0]), int(point[1])), 10, (125, 255, 255), 1)
    
    # ---------------------------------------------- #
    # circles lines intersections
    # ---------------------------------------------- #
    
    intersecting_circles, non_intersecting_circles = find_circle_line_intersections(lines_hori, island_circles)
    
    self.intersecting_circles = []
    if intersecting_circles is not None:
      self.add_layer("intersecting_islands", "circles", intersecting_circles)
      self.intersecting_circles = intersecting_circles
      for circle in intersecting_circles:
        center, radius = circle
        # cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 3)
    
    self.non_intersecting_circles = []
    if non_intersecting_circles is not None:
      self.add_layer("non_intersecting_islands", "circles", non_intersecting_circles)
      self.non_intersecting_circles = non_intersecting_circles
      for circle in non_intersecting_circles:
        center, radius = circle
        # cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)    
    
    # ---------------------------------------------- #
    # embrio & faults
    # ---------------------------------------------- #
    
    self.embrio_circle = []
    img_out, embrio_circle, triangle_faults, faults_mask = embrio_check(img_out, outer_contour)
    if embrio_circle is not None:
      (center, radius) = embrio_circle
      # cv2.circle(img_out, center, radius, (0, 0, 255), 1)
      self.add_layer("embrio_circle", "circles", [embrio_circle])
      self.embrio_circle = [embrio_circle]
    self.triangle_faults = []
    if triangle_faults is not None:
      self.triangle_faults = triangle_faults
      if len(triangle_faults) > 0: 
        self.add_layer("triangle_faults", "triangles", triangle_faults)
      # img_circles = img_raw.copy()
      #   for triangle in triangle_faults:
      #     (center, radius) = circle
      #     # img_circles = cv2.circle(img_circles, (int(center[0]), int(center[1])), radius, (0, 255, 0), -1)
      #     self.add_layer("circle_faults", "circles", [circle])
      #   # debug.push_image(img_circles, "circle_faults")
      # if faults_mask is not None:
      #  debug.push_image(faults_mask, "faults_mask")
    return True

  def draw_elements(self, img_out):
    
    for outer_contour in self.outer_contour:
      cv2.drawContours(img_out, [outer_contour], 0, (255,0,0), 2)
    
    for line in self.lines_hori:
      x1, y1, x2, y2 = line[0]
      cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255),  2)
    
    for line in self.lines_vert:
      x1, y1, x2, y2 = line[0]
      cv2.line(img_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
  
    for point in self.intersection_points:
      cv2.circle(img_out, (int(point[0]), int(point[1])), 10, (125, 255, 255),  2)
    
    for point in self.horizontal_intersection_points:
      cv2.circle(img_out, (int(point[0]), int(point[1])), 10, (125, 0, 255),  2)
        
    for circle in self.intersecting_circles:
      center, radius = circle
      cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 0, 255),  2)
    
    for circle in self.non_intersecting_circles:
      center, radius = circle
      cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 255, 0),  2)

    for circle in self.embrio_circle:
      center, radius = circle
      cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0),  2)
    
    for triangle in self.triangle_faults:
      cv2.polylines(img_out, [triangle], True, (0, 255, 0), 2)
    
    # for circle in self.circle_faults:
    #   center, radius = circle
    #   cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (0, 255, 0),  -1)

    # print("draw elements")
    # for layer in self.layers:
    #   t = layer["type"]
    #   data = json.loads(layer["data"])
    #   print("data", data)
    #   if t == "points": # draw intersection poitns
    #     for p in data:
    #       cv2.circle(img_out, (int(p[0]), int(p[1])), 10, (125, 255, 255), 1)
    #   elif t == "circles": # draw isand circles and embrios
    #     for c in data:
    #       print(layer["name"], c)
    #       center = c[0]
    #       # radius = c[1]
    #       # cv2.circle(img_out, center, radius, (0, 255, 0), 1)
    #   elif t == "lines": # draw vertical and horizontal lines
    #     for line in data:
    #       x1, y1, x2, y2 = line[0]
    #       cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #   elif t == "contours": # draw riice contour and island contour
    #     for c in data:
    #       print(layer["name"], c)
    #       cv2.drawContours(img_out, [c], 0, (255,0,0), 2)
    #   elif t == "contours": # draw riice contour and island contour
    #     for c in data:
    #       print(layer["name"], c)
    #       cv2.drawContours(img_out, c, 0, (255,0,0), 2)
    # self.clear_layers()
    return img_out


  def add_layer(self, name, _type, data):
    self.layers.append({
      "name": name,
      "type": _type,
      "data": data
    })
  
  # opencv write text "test" on image
  def test(self, img_input):
    self.results = { 'test': 'test'}
    self.img_out = img_input.copy()
    cv2.putText(self.img_out, "test", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 2)
    return self.img_out
  
  def get_angle(self):
    return self.rotation_angle

  def clear_layers(self):
    self.layers.clear()
    print("clear_layers", len(self.layers))

  def clear_vars(self):
    self.outer_contour = []
    self.island_circles = []
    self.inner_contours = []
    self.lines_hori = []
    self.lines_vert = []
    self.intersecting_circles = []
    self.non_intersecting_circles = []
    self.embrio_circle = []
    self.intersection_points = []
    self.horizontal_intersection_points = []
    self.triangle_faults = []