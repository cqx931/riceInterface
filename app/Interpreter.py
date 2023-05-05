import json
import random
import ast 

class Interpreter:
  results = ""
  def __init__(self, classifier):
    self.categories = json.load(open('categories.json'))
    self.classifier = classifier
    pass
  
  def analyse(self):
    
    # lines_horizontal
    if type(self.classifier.get_layer_data("lines_horizontal")) == str:
      num_horizontal_lines = len(ast.literal_eval(self.classifier.get_layer_data("lines_horizontal")))
    else:
      num_horizontal_lines = len(self.classifier.get_layer_data("lines_horizontal"))
    
    # lines_vertical
    if type(self.classifier.get_layer_data("lines_vertical")) == str:
      num_vertical_lines = len(ast.literal_eval(self.classifier.get_layer_data("lines_vertical")))
    else:
      num_vertical_lines = len(self.classifier.get_layer_data("lines_vertical"))
    
    # intersections
    if type(self.classifier.get_layer_data("intersections")) == str:
      num_intersections = len(ast.literal_eval(self.classifier.get_layer_data("intersections")))
    else:
      num_intersections = len(self.classifier.get_layer_data("intersections"))
    
    # horizontal intersections
    if type(self.classifier.get_layer_data("horizontal_intersection_points")) == str:
      num_horizontal_intersections = len(ast.literal_eval(self.classifier.get_layer_data("horizontal_intersection_points")))
    else:
      num_horizontal_intersections = len(self.classifier.get_layer_data("horizontal_intersection_points"))
    
    # embrio_circle
    if type(self.classifier.get_layer_data("embrio_circle")) == str:
      embrios = len(ast.literal_eval(self.classifier.get_layer_data("embrio_circle")))
    else:
      embrios = len(self.classifier.get_layer_data("embrio_circle"))
    
    # non_intersecting_islands
    if type(self.classifier.get_layer_data("non_intersecting_islands")) == str:
      num_islands= len(ast.literal_eval(self.classifier.get_layer_data("non_intersecting_islands")))
    else:
      num_islands= len(self.classifier.get_layer_data("non_intersecting_islands"))
    
    # intersecting_islands
    if type(self.classifier.get_layer_data("intersecting_islands")) == str:
      num_islands_cross= len(ast.literal_eval(self.classifier.get_layer_data("intersecting_islands")))
    else:
      num_islands_cross= len(self.classifier.get_layer_data("intersecting_islands"))
    
    # triangle_faults
    if type(self.classifier.get_layer_data("triangle_faults")) == str:
      num_triangle_faults = len(ast.literal_eval(self.classifier.get_layer_data("triangle_faults")))
    else:
      num_triangle_faults = len(self.classifier.get_layer_data("triangle_faults"))

    s = '_'*num_horizontal_lines + '|'*num_vertical_lines + '+'*num_intersections + 'x'*num_horizontal_intersections + '>'*num_triangle_faults + 'º'*embrios + '*'*num_islands
    # scramble string
    #s = ''.join(random.sample(s,len(s)))
    return s
  
  def get_json_results(self):
    return self.results