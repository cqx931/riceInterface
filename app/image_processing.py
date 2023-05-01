# colect here the pre-processing functions
import cv2
import numpy as np
from utils import ccw
# import dip.image as im

ISLAND_SIZE_TRESHOLD = 20000

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
def getInnerIslands (img_masked, contour):
  
  # preprocess image
  _, img_binary = cv2.threshold(img_masked, 125, 255, cv2.THRESH_BINARY)
  
  if len(img_binary.shape) > 2:
    bw = cv2.cvtColor(img_binary, cv2.COLOR_RGB2GRAY)
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
  
# def equalizeLight(img_raw, bright_change=-10, contrast_change=20):
#   brightness = im.brightness(img_raw)
# 
#   # adjust the brightness for images that are too bright
#   if brightness > 10:
#       img_out = im.light(img_raw, bright=bright_change, contrast=0)
# 
#   # ref_image = im.light(image, bright=20, contrast=0)
# 
#   img_out = im.equalize_light(img_out, limit=1, grid=(2,2),gray=True)
#   #image, alpha, beta = im.automatic_brightness_and_contrast(image,clip_hist_percent=10)
# 
#   # black_level = im.back_in_black(image)
#   # if not black_level:
#   #     image = cv2.bitwise_not(image)
# 
#   img_out = im.gauss_filter(img_out, (3,3))
#   img_out = im.light(img_out, bright=0, contrast=20)
#   
#   return img_out