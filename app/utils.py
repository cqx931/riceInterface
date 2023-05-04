import glob
import random
from json import JSONEncoder
import numpy as np
import cv2

RESOLUTION=1080

def getRandomFile(path):
  # Get List of all images
  files = glob.glob(path + '/**/*.jpg', recursive=True)
  return random.choice(files)

def getAllImages(path):
  # Get List of all images
  files = glob.glob(path + '/**/*.jpg', recursive=True)
  return files

def readImage(path):
  img = cv2.imread(path)
  # resize image
  img = cv2.resize(img, (RESOLUTION, RESOLUTION))
  return img

def findFileInFolder(path, name):
  files = glob.glob(path + '/**/'+name+'.*', recursive=True)
  for file in files:
    if (name in file):
      return file
  return False

def ccw(A,B,C):
  return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
      