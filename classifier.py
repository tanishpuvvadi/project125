import numpy as numpy
import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.model_selecion import train_test_split
from sklearn.linear_model import LogisticRegression 
from PIL import image
import PIL.ImageOps 
X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_test = X_train/255
X_test_scaled = X_test/255
clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled,y_train)
get_prediction(image):
  im_pil = Image.open(image)
  im_bw = im_pil.convert("L")
  image_bw_resized = im_bw.resize((28,28), image_ANTIALIAS)
  pixel_filter = 20 
  min_pixel = np.percentile(image_bw_resized, pixel_filter)
  image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
  max_pixel = np.max(image_bw_resized)
  image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
  test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
  test_pred = clf.predict(test_sample)
  return test_pred[0]
  