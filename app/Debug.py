
import matplotlib.pyplot as plt
import math

class Debug:

  images = []
  titles = []
  
  def __init__(self):
    print("debug")
    # pass

  def push_image(self, img, title):
    self.images.append(img)
    self.titles.append(title)

  # Function to plot multiple images
  def display(self):
    # if there are no images, return False
    if len(self.images) == 0:
      return False
    square_size = math.ceil(math.sqrt(len(self.images)))
    num_rows = int(square_size)
    num_cols = int(square_size)
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols, figsize = (10, 8))
    i = 0
    for row in axs:
      for col in row:
        # col.plot(x, y)
        if i < len(self.images):
          col.imshow(self.images[i], 'gray')
          col.set_title(self.titles[i])
        col.set_xticks([])
        col.set_yticks([])
        col.xaxis.set_tick_params(labelbottom=False)
        col.yaxis.set_tick_params(labelleft=False)
        col.spines['top'].set_visible(False)
        col.spines['right'].set_visible(False)
        col.spines['bottom'].set_visible(False)
        col.spines['left'].set_visible(False)
        i += 1
    plt.show()

debug = Debug()