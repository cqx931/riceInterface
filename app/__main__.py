"""
An example of detecting ArUco markers with OpenCV.
"""
import argparse
import os
from dotenv import load_dotenv
from utils import *
import threading
from socket_connection import connectSocket
from test import test

# load .env file
load_dotenv()

DATASET_PATH = os.environ.get("WEBCAM")

# full server url for connection to the socket
# server_url = "http://{}:{}/".format(FLASK_SERVER_IP, FLASK_SERVER_PORT)

# default values
# foo = 0

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--debug', default=False, action='store_true')
parser.add_argument('-t', '--test', default=False, action='store_true')
args = parser.parse_args()

# arg variables
debug = args.debug
test = args.test

def init(): 
    # start opencv
    if test:
        print("Running test")
        test(DATASET_PATH)
        return

# run!
init()