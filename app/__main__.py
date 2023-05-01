"""
An example of detecting ArUco markers with OpenCV.
"""
import argparse
import os
from dotenv import load_dotenv
from utils import *
import threading
from socket_connection import connectSocket
from Classifier import Classifier
import cv2
from Debug import debug

# load .env file
load_dotenv()
DATASET_PATH = os.environ.get("DATASET_PATH")
TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH")

# full server url for connection to the socket
# server_url = "http://{}:{}/".format(SOCKET_SERVER_IP, SOCKET_SERVER_PORT)

# default values
# foo = 0

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug_mode', default=True, action='store_true')
parser.add_argument('-t', '--test', default=False, action='store_true')
parser.add_argument('-r', '--random', default=False, action='store_true')
args = parser.parse_args()

# arg variables
debug_mode = args.debug_mode
print("debug_mode", debug_mode)
mode = 'random'

# setup classfier mode
classifier_mode = "dataset"
if args.test:
  classifier_mode = "test"
if args.random:
  classifier_mode = "random"

debug_images = {}

classifier = Classifier(classifier_mode)

def init(): 
  # start opencv
  if classifier_mode == "test":
    print("Running test")
    run_test_image()
  if classifier_mode == "random":
    print("Running random")
    run_random_image()
  

def run_test_image():
  # load test image
  img_raw = cv2.imread(TEST_IMAGE_PATH, 0)
  img_out = classifier.run(img_raw)
  debug.push_image(img_raw, "raw")
  debug.display()
  # cv2.imshow("out", img_out)
  # cv2.waitKey(0)

def run_random_image():
  for i in range(0, 20):
    # load test image
    path = getRandomFile(DATASET_PATH)
    img_raw = cv2.imread(path, 0)
    img_out = classifier.run(img_raw)

    # images for debug
    # debug.push_image(img_raw, "raw")
    debug.push_image(img_out, "out")
  debug.display()
  
def sendResults():
  # send results to server
  # print("sending results to server")
  # print(classifier.get_results())
  # connectSocket(server_url, classifier.get_results())
  pass  


# run!
init()