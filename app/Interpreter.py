import json

class Interpreter:
  def __init__(self):
    self.categories = json.load(open('categories.json'))
    pass
  
  def analyse(self, layers):
    print("interpreter analyse")
    # load json categories.json file and parse as dict