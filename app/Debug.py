
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np

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
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols, figsize = (12, 9))
    i = 0
    for row in axs:
      for col in row:
        # only display images when they exist
        if i < len(self.images):
          type = 'gray'
          if (self.images[i].shape) == 3:
            type = 'brg'
          col.imshow(self.images[i], type)
          col.set_title(self.titles[i])
        # hide all matplotlib stuff
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

  def display_images(self):
    stacked_images = self.stack_images(0.5, self.images, columns=4)
    cv2.imshow("Display Images", stacked_images)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Destroy the window
    cv2.destroyAllWindows()

  def stack_images(self, scale, img_array, columns=4):
    formatted_arr = []
    max_w = sorted([x.shape[1] for x in img_array], reverse=True)[0]
    max_h = sorted([x.shape[0] for x in img_array], reverse=True)[0]
    aggregate_cols = lambda arr_len, cols: cols if arr_len % cols == 0 and cols / arr_len >= 0.255\
      else aggregate_cols(arr_len, cols + 1)
    padding_space = np.zeros((max_w, max_h), np.uint8)
    for i, img in enumerate(img_array):
      img = cv2.resize(img, (int(max_w * scale), int(max_h * scale)))  # resize all images
      if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # add color channel
      cv2.putText(img, self.titles[i], (int(img.shape[1] * 10 / 100), int(img.shape[0] * 10 / 100),), cv2.FONT_HERSHEY_SIMPLEX, 3, (120, 222, 0), 5)  # add label

      formatted_arr.append(img)

    if len(img_array) % 2 != 0:  # pad image array to make it even
      formatted_arr.insert(len(img_array), cv2.resize(cv2.cvtColor(padding_space, cv2.COLOR_GRAY2BGR),
                                                        (int(max_w * scale), int(max_h * scale))))
    denominator = aggregate_cols(len(formatted_arr), columns)
    print("denominator\t", denominator, "\narr_len\t", len(formatted_arr), "\n ratio\t", columns/len(formatted_arr))
    acc = np.split(np.stack(formatted_arr), denominator)

    return np.concatenate([np.vstack(a) for a in acc[:]], axis=1)

debug = Debug()