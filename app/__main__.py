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
from image_processing import image_diff, RESOLUTION
import time
from categories import detect_category
import sys
import requests

IMAGE_DIFF_THRESHOLD = 0.019

SECONDS_PER_ANGLE = 30 / 180

socketio_client = SocketClient()
socketio_client.connect("http://localhost:3000")
# sendSocketMessage("hello")
import imutils
import threading
import numpy as np
# load .env file
load_dotenv()
DATASET_PATH = os.environ.get("DATASET_PATH")
TEST_IMAGE_PATH = os.environ.get("TEST_IMAGE_PATH")
DATASET_EXPORT_PATH = "dataset_export/"
RASPBERRY_PI = "192.168.1.22"
STREAM_SNAPSHOT = "http://" + RASPBERRY_PI + ":8080/snapshot"
# ustreamer: /snapshot
# old mjg-streamer: ?action=snapshot
INTERVAL_SECONDS = 2
RESTART_SECONDS = 60*5 # 60*5
# full server url for connection to the socket
# server_url = "http://{}:{}/".format(SOCKET_SERVER_IP, SOCKET_SERVER_PORT)

conter = 0
# default values
# foo = 0

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug_mode', default=True, action='store_true')
parser.add_argument('-dm', '--demo', default=False, action='store_true')
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

specific_category = args.specific_category

# setup classfier mode
classifier_mode = "dataset"
if args.test:
  classifier_mode = "test"
elif args.stream:
  classifier_mode = "stream"
else:
  classifier_mode = "default"

debug_images = {}



classifier = Classifier(classifier_mode)
interpreter = Interpreter(classifier)
stream_on = True

def init():
  # start opencv
  if args.test == True:
    print("Running test")
    run_test_image()
  if args.demo == True:
    print("Demo")
    demo()
  elif args.random == True:
    print("Running random", args.random)
    run_random_image()
  elif args.categories == True:
    print("Running categories")
    run_categories_images(specific_category)
  elif args.stream == True:
    print("Running stream")
    stream()
  elif args.all_dataset == True:
    print("Running stream")
    run_all_dataset()


def stream():
  # TODO: try if there is a connection
  is_new_frame = False
  lastFrame = None
  sent_results = False
  has_rice = False
  has_rotated = False
  session_start_time = time.time()
  start_time = 0
  last_rotation_time = 0 # after rotation is done
  current_rotation_waiting_time = 0

  while stream_on:
    try:
      #print("img request", time.time() - session_start_time)
      raw = imutils.url_to_image(STREAM_SNAPSHOT)
      image = cv2.resize(raw, (RESOLUTION, RESOLUTION))
    except Exception as e:
      print("exception", e)
      continue
    if time.time() - last_rotation_time < 2:
      print("waiting for rotation!")
      img_out = image
      cv2.imshow("out", img_out)
      continue
    
    if time.time() - session_start_time >= RESTART_SECONDS and time.time() - start_time >= 90:
        print("RESTART BACKEND")
        # TODO: this idle restart time doesnt seem correct
        sendMessage(180) # rotate the rice for 180
        sendClear()
        os.execv(sys.executable, ['python'] + sys.argv)
    if np.array_equal(image, lastFrame) :
      img_out = image

    else: # if anything changes
      # avoid first frame exception
      found = False
      # if its first frame
      if lastFrame is None:
        lastFrame = image
        sessin_start_time = time.time()
        start_time = sessin_start_time
        classifier.clear_layers() # need to clear the data everytime so it doesnt accumulate
        print("resolution:", image.shape[0],image.shape[1])
        found = classifier.process(image)
        if found:
          print("found rice")
          sendLayers()
        else:
          print("no rice")
        has_rice = found
      # end of first frame
      # first check if the frames are diff enough, only compute when its stable
      if image_diff(lastFrame, image) > IMAGE_DIFF_THRESHOLD:
        print("diff", image_diff(lastFrame, image))
        classifier.clear_layers() # need to clear the data everytime so it doesnt accumulate
        found = classifier.process(image)
        has_rice = found
        sent_results = False
        if found:
          print("found rice")
          sendLayers()
          start_time = time.time() # everytime the frame is diff and there is a rice, restart timer
        else:
          print("no rice")
          #if there is no rice in a new frame !
          sendClear() # clear front end
          # sent_results = False
      
      # if it's stable
      lastFrame = image
      # draw layers
      img_out = classifier.draw_elements(image)
    # print(has_rice, sent_results , time.time() - start_time)
    if has_rice:
      # if the image is stabe for INTERVAL_SECONDS time, then send rersults 
      # print("Timer:", time.time() - start_time > INTERVAL_SECONDS, sent_results)
      if time.time() - start_time > INTERVAL_SECONDS:
        # ok is stable
        rotation_angle = classifier.get_angle()
        if (rotation_angle > 15 or rotation_angle < -15) and has_rotated == False:
          print("rotate", rotation_angle, has_rotated)
          sendMessage(rotation_angle)
          has_rotated = True
          start_time = time.time()
          last_rotation_time = time.time()
          current_rotation_waiting_time = abs(rotation_angle) * SECONDS_PER_ANGLE + 1
        
        if sent_results == False:
          print("send result")
          sent_results = sendResults()
          has_rotated = False

    cv2.imshow("out", img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


def run_test_image():
  # load test image
  img_raw = cv2.imread(TEST_IMAGE_PATH, 0)
  img_out = classifier.run(img_raw)
  debug.push_image(img_raw, "raw")
  debug.display()
  # cv2.imshow("out", img_out)
  # cv2.waitKey(0)

def demo():
  # load test image
  print("Test image:")
  print(TEST_IMAGE_PATH)
  img_raw = cv2.imread(TEST_IMAGE_PATH, 0)
  classifier.process(img_raw)
  img_out = classifier.draw_elements(img_raw)
  cv2.imshow("out", img_out)
  sendLayers()
  sendResults()
  cv2.waitKey(0)

def run_random_image():
  for i in range(0, 16):
    # load test image
    path = getRandomFile(DATASET_PATH)
    # print("random image file", path)
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
      img_raw = readImage(path)
      classifier.run(img_raw)
      img_out = classifier.draw_elements(img_raw)
      text = c["example"][0] + " " + str(c["index"]) + " " + c["title"] + " " + c["symbol"]
      debug.push_image(img_out, text)
  sendResults()
  print("len(debug.images)", len(debug.images))
  if len(debug.images) % 8 == 0:
    debug.display_images()
  else:
    debug.display()

def run_all_dataset():
  files = getAllImages(DATASET_PATH)
  with open('results.txt', 'a') as fd:
    for i, f in enumerate(files):
      img_raw = readImage(f)
      img_out = classifier.run(img_raw)
      results = interpreter.analyse()
      category = detect_category(results)
      cv2.putText(img_out, results, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (120, 222, 0), 5)
      filename = category + " " + results + " " + os.path.split(f)[1].split('.')[0] + ".jpg"
      print("filename", filename + " " + results)
      fd.write(f'\n{filename + "    " + results}')
      img_out = classifier.draw_elements(img_raw)
      classifier.clear_layers()
      # save image
      if args.save_file:
        cv2.imwrite(DATASET_EXPORT_PATH + filename, img_out)


def analyzeResults():
  results = interpreter.analyse(classifier.get_layers())

def sendResults():
  results = interpreter.analyse()
  # print("send results", results)
  if len(results) > 0:
    #print("send results", results)
    socketio_client.sendMessage('results', results)
    classifier.clear_layers()
    return True
  else:
    return False

def sendLayers():
  layers = classifier.get_json_layers()
  socketio_client.sendMessage('layers', layers)
  # print("send layers", len(layers))
  # classifier.clear_layers()


def sendClear():
  socketio_client.sendMessage('clear', "")
  print("send clear!")

def sendMessage(text):
  url = "http://192.168.1.22:5000/rotate?angle=" + str(text)
  requests.get(url)
  
  return

# run!
init()
