import socketio

global socketClient
is_connected = False

class SocketClient:
  
  def __init__(self):
    pass
      
  def on_connect():
      print('connect')

  def on_disconnect():
      print('disconnect')

  def on_reconnect():
      print('reconnect')

  def connect(self, server_path):
    try:
      self.socketClient = socketio.Client()
      self.socketClient.connect(server_path)
    except socketio.exceptions.ConnectionError as err:
      self.is_connected = False
      print("Error on socket connection")
    else:
      self.is_connected = True

  def sendMessage(self, message, data):
    if (self.is_connected == False):
      print("socket not connected")
      return False
    try:
      self.socketClient.emit(message, data)
    except socketio.exceptions.BadNamespaceError as err:
      print("Error sending message")