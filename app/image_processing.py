# colect here the pre-processing functions
import cv2
import numpy as np
from utils import ccw
from math import atan2, cos, sin, sqrt, pi

# import dip.image as im

ISLAND_SIZE_TRESHOLD = 2000

# otsu thresholding
def otsu_thresholding(img):
  if (len(img.shape) == 3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(img,(5,5),0)
  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return th3

# find countours, expects otus thresholded image
def findMaxContour(img_otsu):
  # Find all the contours in the thresholded image
  contours, _ = cv2.findContours(img_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  max_contour = max(contours, key = cv2.contourArea)
  return max_contour

def getContourRect (contour):
  # get rectangle bounding contour
  rect = cv2.minAreaRect(contour)
  return rect

def getMaskedImage (img_raw, contour):
  img_out = img_raw.copy() 
  # img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
  # Plot the image to test
  mask = np.zeros(img_raw.shape, np.uint8)
  cv2.drawContours(mask, [contour], 0, color=(255, 255, 255), thickness=cv2.FILLED)
  img_masked = cv2.bitwise_and(img_out, mask)
  return img_masked

# return array of contours of the inner islands of the rice
def getInnerIslands (img_binary, contour):
  
  if len(img_binary.shape) > 2:
    bw = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
  else:
    bw = img_binary
  
  # make aproximation of outer contours so it doesnt demand too much processing power when checking intersections
  peri = cv2.arcLength(contour, True)
  aprox_outer = cv2.approxPolyDP(contour, 0.005 * peri, True)

  contours_inner, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  # filter out contours that are too small or too big
  # contours_inner = [c for c in contours_inner if cv2.contourArea(c) > 100 and cv2.contourArea(c) < 10000]
  
  contours_inner = filterIslandContours(contours_inner, aprox_outer)
  
  return contours_inner

def filterIslandContours(contours, aprox_outer):
  filtered = []
  
  for c in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(c)

    # make aproximation of contours so it doesnt demand too much processing power
    peri = cv2.arcLength(c, True)
    aprox_inner = cv2.approxPolyDP(c, 0.005 * peri, True)

    # Ignore contours that are too small
    if area < ISLAND_SIZE_TRESHOLD:
      continue
    # check if contours intersect with outer rice contour 
    if contour_intersect(aprox_outer, aprox_inner):
      continue
    
    filtered.append(c)
    
  return filtered
    
def drawIslandDetections(img_out, contours):
  for c in contours:
    # draw island detections
    center, radius = cv2.minEnclosingCircle(c)
    cv2.circle(img_out, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 3)
    cv2.drawContours(img_out, [c], 0, (255,0,0), 2)
  return img_out

# this is needed to avoid getting inner countours that are not on the center of the rice
def contour_intersect(cnt_ref,cnt_query):
    ## Contour is a list of points
    ## Connect each point to the following point to get a line
    ## If any of the lines intersect, then break

    for ref_idx in range(len(cnt_ref)-1):
    ## Create reference line_ref with point AB
      A = cnt_ref[ref_idx][0]
      B = cnt_ref[ref_idx+1][0] 
  
      for query_idx in range(len(cnt_query)-1):
        ## Create query line_query with point CD
        C = cnt_query[query_idx][0]
        D = cnt_query[query_idx+1][0]
    
        ## Check if line intersect
        if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
          ## If true, break loop earlier
          return True
    return False
  
def equalize_image(img):
    # convert image to grayscale
    if (len(img.shape) == 3):
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clip_limit=2.0
    tile_grid_size=(8, 8)

    # create CLAHE object with desired parameters
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # apply CLAHE to grayscale image
    equalized = clahe.apply(img)
    
    # perform histogram equalization on grayscale image
    # equalized = cv2.equalizeHist(img)

    # convert back to color image
    equalized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return equalized_image
  
def threshold_image(image, block_size=50, constant=2):
    # convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply adaptive thresholding to grayscale image
    thresh = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

    # invert the threshold image so that the most relatively bright elements are white
    thresh = cv2.bitwise_not(thresh)


    # apply the threshold to the original image to extract the bright elements
    # result = cv2.bitwise_and(image, image, mask=thresh)

    return thresh

def threshold_and_mask(image, exclude_percent=8):
  # convert image to grayscale
  gray = image.copy()# cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  gray = cv2.GaussianBlur(gray,(9,9),0)

  
  # apply Otsu's thresholding method
  _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # calculate the histogram of the thresholded image
  hist, _ = np.histogram(gray[thresh == 255], bins=256, range=[0, 256])

  # calculate the cumulative sum of the histogram
  cumsum = np.cumsum(hist)

  # calculate the cumulative percentage of the histogram
  cumsum_percent = (cumsum / cumsum[-1]) * 100

  # find the threshold value that excludes the top exclude_percent% of the brightest pixels
  threshold_value = np.argmax(cumsum_percent > (100 - exclude_percent))
  
  # create a binary mask that excludes the brightest pixels
  _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

  # invert the value of the mask so that the brightest pixels are white
  inverted = cv2.bitwise_not(mask)

  return inverted

def detect_trace(image):
  # apply Hough Line Transform to detect the trace
  lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=50, minLineLength=500, maxLineGap=100)

  # extract the longest line detected
  longest_line = None
  longest_length = 0
    
  if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > longest_length:
            longest_line = line
            longest_length = length
  if longest_line is None:
    return None, lines
  # extract the vector of the longest line
  x1, y1, x2, y2 = longest_line[0]
  vector = np.array([x2-x1, y2-y1])
  return longest_line, lines
  
def filter_lines(lines, angle_range=[-180, 180], min_distance=10):
  filtered_lines = []
  for i, line1 in enumerate(lines):
    x1, y1, x2, y2 = line1[0]
    is_valid = True
    for line2 in filtered_lines:
      x3, y3, x4, y4 = line2[0]
      d1 = np.linalg.norm(np.array([x1, y1]) - np.array([x3, y3]))
      d2 = np.linalg.norm(np.array([x2, y2]) - np.array([x4, y4]))
      d3 = np.linalg.norm(np.array([x1, y1]) - np.array([x4, y4]))
      d4 = np.linalg.norm(np.array([x2, y2]) - np.array([x3, y3]))
      if d1 < min_distance or d2 < min_distance or d3 < min_distance or d4 < min_distance:
        is_valid = False
        break
    # Add line to filtered list if it passes all checks
    if is_valid:
      filtered_lines.append(line1)
  return filtered_lines
  
def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
  textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
 
  return angle