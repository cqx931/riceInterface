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
    # blur = cv2.GaussianBlur(img,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, img_binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    return img_binary

# find countours, expects otus thresholded image


def findMaxContour(img_otsu):
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(
        img_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour


def getContourCenterPoint(contour):
    # get rectangle bounding contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def getContourRect(contour):
    # get rectangle bounding contour
    rect = cv2.minAreaRect(contour)
    return rect


def getMaskedImage(img_raw, contour):
    img_out = img_raw.copy()
    # img_out = cv2.cvtColor(img_out,cv2.COLOR_GRAY2BGR)
    # Plot the image to test
    mask = np.zeros(img_raw.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, color=(
        255, 255, 255), thickness=cv2.FILLED)
    img_masked = cv2.bitwise_and(img_out, mask)
    return img_masked

# return array of contours of the inner islands of the rice


def getInnerIslands(img_binary, contour):

    if len(img_binary.shape) > 2:
        bw = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
    else:
        bw = img_binary

    # make aproximation of outer contours so it doesnt demand too much processing power when checking intersections
    peri = cv2.arcLength(contour, True)
    aprox_outer = cv2.approxPolyDP(contour, 0.005 * peri, True)

    contours_inner, _ = cv2.findContours(
        bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
        cv2.circle(img_out, (int(center[0]), int(
            center[1])), int(radius), (255, 0, 0), 3)
        cv2.drawContours(img_out, [c], 0, (255, 0, 0), 2)
    return img_out

# this is needed to avoid getting inner countours that are not on the center of the rice


def contour_intersect(cnt_ref, cnt_query):
    # Contour is a list of points
    # Connect each point to the following point to get a line
    # If any of the lines intersect, then break

    for ref_idx in range(len(cnt_ref)-1):
        # Create reference line_ref with point AB
        A = cnt_ref[ref_idx][0]
        B = cnt_ref[ref_idx+1][0]

        for query_idx in range(len(cnt_query)-1):
            # Create query line_query with point CD
            C = cnt_query[query_idx][0]
            D = cnt_query[query_idx+1][0]

            # Check if line intersect
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                # If true, break loop earlier
                return True
    return False


def equalize_image(img):
    # convert image to grayscale
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clip_limit = 2.0
    tile_grid_size = (8, 8)

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
    thresh = cv2.adaptiveThreshold(
        image, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

    # invert the threshold image so that the most relatively bright elements are white
    thresh = cv2.bitwise_not(thresh)

    # apply the threshold to the original image to extract the bright elements
    # result = cv2.bitwise_and(image, image, mask=thresh)

    return thresh


def threshold_and_mask(image, exclude_percent=8):
    # convert image to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      gray = image.copy()  # cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # apply Otsu's thresholding method
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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


def detect_trace(image, threshold=50, minLineLength=500, maxLineGap=500):
    # apply Hough Line Transform to detect the trace
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180,
                            threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

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
    return lines


def filter_lines_by_distance(lines, min_distance=10):
    filtered_lines = []
    for i, line1 in enumerate(lines):
        # print("line1", line1)
        if line1 is None:
            continue
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


def filter_lines_by_angle(lines, ref_angle, tolerance=50):
    # print("ref_angle", ref_angle)
    # ref_angle = np.rad2deg(ref_angle)
    filtered_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        is_valid = True
        angle = np.rad2deg(get_angle([x1, y1], [x2, y2]))
        #print("angles: ", angle, ref_angle)
        angle_diff = abs(angle - ref_angle)
        if angle_diff <= tolerance or abs(angle_diff - 180) <= tolerance:
            filtered_lines.append(line)
    return filtered_lines
    # if angle is within ref_angle tolerance


def get_angle(p_, q_):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    return angle


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    # [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) +
                      (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])),
             (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])),
             (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])),
             (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    # [visualization1]


def getOrientation(pts, img):
    # [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    # [pca]

    # [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
          cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
          cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    # drawAxis(img, cntr, p1, (255, 255, 0), 1)
    # drawAxis(img, cntr, p2, (0, 0, 255), 5)

    # orientation in radians
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    # [visualization]

    return angle


def embrio_check(img, outer_contour):
    DIFF_AREA_THRESHOLD = 200000

    output = img.copy()
    # output = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)

    eps = 0.001

    # approximate the contour
    peri = cv2.arcLength(outer_contour, True)
    approx = cv2.approxPolyDP(outer_contour, eps * peri, True)
    # draw the approximated contour on the image

    hull = cv2.convexHull(outer_contour)
    #cv2.drawContours(output, [hull], 0, (0, 255, 0), 10)

    (centerCoordinates, axesLength, angle) = cv2.fitEllipse(hull)
    axesLength = tuple([0.95*x for x in axesLength])
    # minEllipse[1] = float(minEllipse[1]) * 0.9

    # new_radius =  tuple([0.9*x for x in minEllipse[1]])
    minEllipse = (centerCoordinates, axesLength, angle)
    
    # Create a blank mask image and draw the contours
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [outer_contour], 0, (255, 255, 255), -1)

    # diff between hull area and rice area
    rice_area = cv2.contourArea(outer_contour)
    hull_area = cv2.contourArea(hull)
    area_diff = (hull_area - rice_area)/rice_area
    # print("area_diff", area_diff)
    # cv2.putText(output, str(area_diff), (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
    circle_faults = []
    if area_diff > 0.05:
      # get contours from neg mask.
      print(" ")
      # Create a blank image for the ellipse mask
      hull_mask = np.zeros(img.shape[:2], dtype=np.uint8)
      cv2.drawContours(hull_mask, [hull], 0, (255, 255, 255), -1)
      # Invert the ellipse mask & Get the intersection between the contour mask and the inverted ellipse mask
      inverted_hull_mask = cv2.bitwise_not(hull_mask)
      faults_mask = cv2.bitwise_not(cv2.bitwise_or(mask, inverted_hull_mask))
      cv2.imshow('window', faults_mask)
      faults_contours, _ = cv2.findContours(faults_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for fault_contour in faults_contours:
        fault_area = cv2.contourArea(fault_contour)
        centerPoint, radius = cv2.minEnclosingCircle(fault_contour)
        radius = int(cv2.pointPolygonTest(fault_contour, centerPoint, True))
        if radius > 0:
          circle_faults.append((centerPoint, radius))
    
    # Create a blank image for the ellipse mask
    ellipse_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(ellipse_mask, minEllipse, (255, 255, 255), -1)
    
    # Invert the ellipse mask & Get the intersection between the contour mask and the inverted ellipse mask
    inverted_ellipse_mask = cv2.bitwise_not(ellipse_mask)
    embrio_mask = cv2.bitwise_not(cv2.bitwise_or(mask, inverted_ellipse_mask))
    embrio_contours, _ = cv2.findContours(embrio_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    embrio_contour = max(embrio_contours, key=cv2.contourArea)

    embrio_circle = None
    centerPoint = getContourCenterPoint(embrio_contour)
    radius = abs(int(cv2.pointPolygonTest(embrio_contour, centerPoint, True)))
    # print("embrio radius", radius)
    if radius > 10 and radius < 100:
      embrio_circle = (centerPoint, radius)
    
    return output, embrio_circle, circle_faults


def find_largest_circle(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the largest fitting circle
    max_radius = 0
    max_center = None

    # Iterate through the contours
    for contour in contours:
        # Compute the minimum enclosing circle for the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)

        # Check if the circle can fit within the mask
        if x - radius >= 0 and x + radius <= mask.shape[1] and \
                y - radius >= 0 and y + radius <= mask.shape[0]:

            # Check if the circle is the largest so far
            if radius > max_radius:
                max_radius = radius
                max_center = (int(x), int(y))

    # Return the position and radius of the largest fitting circle
    return max_center, max_radius


def find_intersection_points(lines1, lines2):
    intersection_points = []
    for line1 in lines1:
        for line2 in lines2:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            denominator = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1))
            if denominator == 0:
                # lines are parallel, skip
                continue
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
            if ua < 0 or ua > 1 or ub < 0 or ub > 1:
                # intersection point is not within both line segments, skip
                continue
            intersection_x = int(x1 + ua * (x2 - x1))
            intersection_y = int(y1 + ua * (y2 - y1))
            intersection_points.append((intersection_x, intersection_y))
    return intersection_points

def find_circle_line_intersections(lines, circles):
  intersecting_circles = []
  non_intersecting_circles = []
  
  for circle in circles:
    # Check if the circle intersects with any of the lines
    intersects_line = False
    for line in lines:
      intersects_line = line_circle_intersection(line[0], circle)
      if intersects_line:
        break
    # If the circle intersects with at least one line, add it to intersecting_circles
    if intersects_line:
      intersecting_circles.append(circle)
    else:
      non_intersecting_circles.append(circle)
  
  return intersecting_circles, non_intersecting_circles

def line_circle_intersection(line, circle):
    x1, y1, x2, y2 = line
    (cx, cy), r = circle

    dx = x2 - x1
    dy = y2 - y1
    a = dx * dx + dy * dy
    b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
    c = (x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy) - r * r
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        # No intersection
        return False
    else:
        # Compute the intersection(s)
        t1 = (-b + sqrt(discriminant)) / (2 * a)
        t2 = (-b - sqrt(discriminant)) / (2 * a)

        # Check if the intersection points are within the line segment
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
        else:
            return False
