import glob
import random

def getRandomFile(path):
  # Get List of all images
  files = glob.glob(path + '/**/*.jpg', recursive=True)
  return random.choice(files)

def ccw(A,B,C):
  return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])