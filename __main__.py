import os 
from PIL import Image
import numpy
import matplotlib.pyplot as pyplot
import plotly.graph_objs as graph_objs
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh

data_dir = '/Users/andranik/Downloads/numTest'
target_digits = ['3', '8', '9']


# X is img data, y is label corresponding to data 
def load_digit_images(data_dir, target_digits, max_images_per_digit=500):
    X = []
    y = []

    for digit in target_digits:
        folder = os.path.join(data_dir, str(digit))
        count = 0
        for file in os.listdir(folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).convert("L")         # Grayscale conversion
                img = img.resize((28, 28))                      # MNIST standard size
                img_array = numpy.array(img).flatten() / 255.0  #1D vector from [0.0, 1.0]
                X.append(img_array)
                y.append(digit)
                count += 1
                if count >= max_images_per_digit:
                    break

    return numpy.array(X), numpy.array(y)


X, y = load_digit_images(data_dir, target_digits) 
print("Shape of image data:", X.shape)
print("Shape of labels:", y.shape)
