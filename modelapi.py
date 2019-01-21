
import numpy as np
import cv2
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras import losses
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.optimizers import Adam
import glob
import matplotlib.pyplot as plt

def predict(i):
    X_cross = list()
    # Y_cross = list()
    
    lsd = cv2.imread(i)
    lsd = cv2.cvtColor(lsd,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(lsd,7) 
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = cv2.resize(otsu, (112,112))
    #name = i[:-4].split('_')[3:]
    #le = len(name)
    #letter = [0 for _ in range(127)]
    #is_valid = True
    #for j in name:
    #    value = int(j) - 2304
    #    if(value < 0 and value > 126):
    #        is_valid = False
    #    letter[value] = 1

    #if(is_valid and le > 0):
    #    Y_cross.append(letter)
    #    #print(sum(letter))
    img = img.reshape(112,112,1)
    X_cross.append(img)

    

    #Y_cross = np.asarray(Y_cross)
    X_cross = np.asarray(X_cross)/255

    model = load_model("tp112")

    Y_p_cross = model.predict(X_cross)

    Y_p_cross = np.where(Y_p_cross > 0.5 , 1 , 0)

    y_p = np.asarray(Y_p_cross[0])
    #print(y_p)
    anl = []
    for vas in range(127):
        if y_p[vas]==1:
            es = vas+2304
            anl.append(es)

    return anl  

for k in glob.glob('./*.png'):

    
    print(predict(k))
    print(k)