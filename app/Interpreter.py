import json
import random

class Interpreter:
  results = ""
  def __init__(self, classifier):
    self.categories = json.load(open('categories.json'))
    self.classifier = classifier
    pass
  
  def analyse(self):
    num_horizontal_lines = len(self.classifier.get_layer_data("lines_horizontal"))
    num_vertical_lines = len(self.classifier.get_layer_data("lines_vertical"))
    num_intersections = len(self.classifier.get_layer_data("intersections"))
    embrios = len(self.classifier.get_layer_data("embrio_circle"))
    num_islands= len(self.classifier.get_layer_data("non_intersecting_islands"))
    s = '_'*num_horizontal_lines + '_'*num_vertical_lines + '+'*num_intersections + 'º'*embrios + '*'*num_islands
    # scramble string
    s = ''.join(random.sample(s,len(s)))
    return s
  
  def get_json_results(self):
    return self.results