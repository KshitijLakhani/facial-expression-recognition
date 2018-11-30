import os
from PIL import Image
import numpy as np

photos_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__) + '/photos/'))

for subdir, dirs, files in os.walk(photos_base_path):
    for file in files:
        # print(os.path.join(subdir, file))
        im = Image.open(os.path.join(subdir, file))
        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        print(pixels)

