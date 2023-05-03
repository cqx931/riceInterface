"""
An example of detecting ArUco markers with OpenCV.
"""
import argparse
import os
from dotenv import load_dotenv
from utils import *
# from socket_connection import connectSocket
from Classifier import Classifier
import cv2
# import json
from Debug import debug
from socket_connection import SocketClient
from Interpreter import Interpreter
# standard Python

socketio_client = SocketClient()
socketio_client.connect("http://localhost:3000")
# sendSocketMessage("hello")
import imutils
import threading

# load .env file
load_dotenv()
DATASET_PATH = os.environ.get("DATASET_PATH")
TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH")
STREAM_SNAPSHOT = "http://192.168.1.22:8080/?action=snapshot"
DATASET_EXPORT_PATH = "dataset_export/"
# full server url for connection to the socket
# server_url = "http://{}:{}/".format(SOCKET_SERVER_IP, SOCKET_SERVER_PORT)

# default values
# foo = 0

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug_mode', default=True, action='store_true')
parser.add_argument('-t', '--test', default=False, action='store_true')
parser.add_argument('-r', '--random', default=False, action='store_true')
parser.add_argument('-s', '--stream', default=False, action='store_true')
parser.add_argument('-c', '--categories', default=False, action='store_true')
parser.add_argument('-a', '--all_dataset', default=False, action='store_true')
parser.add_argument('-sc', '--specific_category',  default=None, type=int)
parser.add_argument('-S', '--save_file', default=False, action='store_true')
args = parser.parse_args()

# arg variables
debug_mode = args.debug_mode
print("debug_mode", debug_mode)
mode = 'random'

specific_category = args.specific_category

# setup classfier mode
classifier_mode = "dataset"
if args.test:
  classifier_mode = "test"
else:
  classifier_mode = "default"

debug_images = {}



classifier = Classifier(classifier_mode)
interpreter = Interpreter(classifier)

def init(): 
  # start opencv
  if args.test:
    print("Running test")
    run_test_image()
  elif args.random:
    print("Running random")
    run_random_image()
  elif args.categories:
    print("Running random")
    run_categories_images(specific_category)
  elif args.stream:
    print("Running stream")
    stream()
  elif args.all_dataset:
    print("Running stream")
    run_all_dataset()
  

def printTimer():
  print(".")

def stream():
  def stream_still():
    image = imutils.url_to_image(STREAM_SNAPSHOT)
    img_raw = image
    img_out = classifier.run(img_raw)
    cv2.imshow("out", img_out)
  # TODO: setInterval
  # t = threading.Timer(2, printTimer)
  # t.start()
  stream_still()
  cv2.waitKey(0)

def run_test_image():
  # load test image
  img_raw = cv2.imread(TEST_IMAGE_PATH, 0)
  img_out = classifier.run(img_raw)
  debug.push_image(img_raw, "raw")
  debug.display()
  # cv2.imshow("out", img_out)
  # cv2.waitKey(0)

def run_random_image():
  for i in range(0, 16):
    # load test image
    path = getRandomFile(DATASET_PATH)
    print("random image file", path)
    img_raw = cv2.imread(path, 0)
    img_out = classifier.run(img_raw)
    # debug.push_image(img_raw, "raw")
    debug.push_image(img_out, "out")
    sendResults()
  # analyzeResults()
  # debug.display()
  debug.display_images()
  
def run_categories_images(specific_category = None):
  for c in interpreter.categories:
    if specific_category != None and c["index"] != specific_category:
      continue
    path = findFileInFolder(DATASET_PATH, c["example"][0])
    for name in c["example"]:
      path = findFileInFolder(DATASET_PATH, name)
      # print("name", name, path)
      if path == None:
        break
      img_raw = cv2.imread(path, 0)
      img_out = classifier.run(img_raw)
      text = c["example"][0] + " " + str(c["index"]) + " " + c["title"] + " " + c["symbol"]
      debug.push_image(img_out, text)
  sendResults()
  if len(debug.images) % 8 == 0:
    debug.display_images()
  else:
    debug.display()  
  
def run_all_dataset():
  files = getAllImages(DATASET_PATH)
  for i, f in enumerate(files):
    img_raw = cv2.imread(f, 0)
    img_out = classifier.run(img_raw)
    filename = os.path.split(f)[1].split('.')[0] + "_opencv.jpg"
    print("filename", filename)
    # save image
    if args.save_file:
      cv2.imwrite(DATASET_EXPORT_PATH + filename, img_out)
  
  
def analyzeResults():
  results = interpreter.analyse(classifier.get_layers())
  
def sendResults():
  # TO DO, fix numpy array to json
  # results = classifier.get_results()
  results = interpreter.analyse()
  print("results", results)
  layers = classifier.get_json_layers()
  classifier.clear_layers()
  socketio_client.sendMessage('layers', layers)
  # sendSocketMessage('results', results)

  # print("layers", layers)
  # send results to server
  # print("sending results to server")
  # print(classifier.get_results())
  # connectSocket(server_url, classifier.get_results())
  pass  


# run!
init()