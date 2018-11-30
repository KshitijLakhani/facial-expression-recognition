import os
from PIL import Image
import numpy as np

photos_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__) + '/photos/'))
data_kind = "PrivateTest"
emotion_mappings = {'angry': '0',
                    'disgust': '1',
                    'fear': '2',
                    'happy': '3',
                    'sad': '4',
                    'surprise': '5',
                    'neutral': '6'}


for subdir, dirs, files in os.walk(photos_base_path):
    for file in files:
        curr_emotion = file.split('-')[0]
        im = Image.open(os.path.join(subdir, file)).convert('L')
        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        pixels = np.asarray(pixels).flatten().tolist()
        img_str = ' '.join(pixels)
        np.savetxt("test_data.csv", pixels, fmt='%d', delimiter=",")
