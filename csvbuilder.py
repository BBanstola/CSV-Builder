from scipy.misc import imread
import numpy as np
from skimage.color import rgb2gray
import os
import pandas as pd

import cv2

for directory, subdirectories, files in os.walk("./data"):
    for file in files:

        # if file == ".DS_Store": continue
        img = imread(os.path.join(directory, file))
        img = cv2.resize(img, (30, 42), interpolation=cv2.INTER_AREA)      #Sizing Image as per the requirement which is 30*42 in this case
        #print(img)
        if(len(img.shape)==3):
            im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            im = img

        value = im.flatten()
        value=255-value
        value = np.hstack((directory[7:], value))       #the 7 here is tricky, ./data has 6 element so its 7; change it according to your requirement
        df = pd.DataFrame(value).T
        with open('train_initial.csv', 'a') as f:
            df.to_csv(f, header=False, index=False)

df = pd.read_csv('train_initial.csv')
df = df.sample(frac=1)
df.to_csv('train.csv', header=False, index=False)

        #You can create a testing dataset similarly