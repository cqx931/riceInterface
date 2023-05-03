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
    return self.layers
  
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
    img_binary = threshold_and_mask(img_masked)
  
    debug.push_image(img_binary, "binary image")

    # img_out = img_masked.copy()
    if (len(img_out.shape) == 1):
      img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    
    # darker image for island detection
    # img_darker = img_out # equalizeLight(img_masked, -10)

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
      cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 10)

    #if lines is not None:
    if lines is not None:
      lines_a = filter_lines(lines, angle_range=[angle-20, angle+20], min_distance=100)
      if lines_a is not None:
        for line in lines_a:
          x1, y1, x2, y2 = line[0]
          cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    # ---------------------------------------------- #
    # outer faults
    # ---------------------------------------------- #
    
    img_out = fault_check(img_out, outer_contour)
    
    # ---------------------------------------------- #
    # draw things 
    # ---------------------------------------------- #

    # draw island detections
    circles = []
    for c in inner_contours:
      center, radius = cv2.minEnclosingCircle(c)
      circles.append([center, radius])
      #cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 3)
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

def filter_lines_by_angle(lines, angle, tolerance=10):
  """
  Filters lines based on their angle compared to a reference angle, allowing for opposite directions.

  Args:
      lines: A list of OpenCV line representations (vx, vy, x, y).
      angle: The reference angle (in degrees).
      tolerance: The tolerance (in degrees) for the angle difference. Default is 10.

  Returns:
      A list of lines whose angle is within the tolerance of the reference angle or its opposite.
  """
  filtered_lines = []
  ref_angle_rad = np.radians(angle)

  for line in lines:
    x1, y1, x2, y2 = line[0]
    line_angle_rad = np.arctan2(vy, vx)
    line_angle_deg = np.degrees(line_angle_rad)
    print("line_angle_deg", line_angle_deg)
    angle_diff = abs(line_angle_deg - angle)

    if angle_diff <= tolerance or abs(angle_diff - 180) <= tolerance:
      filtered_lines.append(line)

  return filtered_lines

def fault_check(img, outer_contour):
  DIFF_AREA_THRESHOLD = 200000
  
  output = img.copy()
  # output = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)

  eps = 0.001

  # approximate the contour
  peri = cv2.arcLength(outer_contour, True)
  approx = cv2.approxPolyDP(outer_contour, eps * peri, True)
  # draw the approximated contour on the image

  cv2.drawContours(output, [outer_contour], -1, (0, 255, 0), 10)
  cv2.drawContours(output, [approx], -1, (255, 0, 0), 10)

  hull = cv2.convexHull(outer_contour)
  cv2.drawContours(output, [hull], 0, (0, 255, 0), 10)

  minEllipse = cv2.fitEllipse(hull)
  cv2.ellipse(output, minEllipse, (255, 0, 0), 3)

  rice_area = cv2.contourArea(outer_contour)

  diff_area = cv2.contourArea(hull) - rice_area
  print("diff area", rice_area, diff_area, diff_area / rice_area)

  if diff_area > DIFF_AREA_THRESHOLD:
    print("big gap!")
  
  return output
