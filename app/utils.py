import matplotlib.pyplot as plt
import glob
import os
import random

def getRandomFile(path):
  # Get List of all images
  files = glob.glob(path + '/**/*.jpg', recursive=True)
  return random.choice(files)