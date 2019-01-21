
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.optimizers import Adam
from keras import losses

def load():
    return load_model("Atharva")



# data_all = [mx.nd.array(data)] + init_state_arrays
# label_all = [mx.nd.array(label)]
# data_names = ['data'] + init_state_names
# label_names = ['label']
# data_batch = SimpleBatch(data_names, data_all, label_names, label_all, bucket_key)




def predict(i):
    Xt = list()
    # Y_cross = list()
    
    anl = cv2.imread(i)
    anl = cv2.cvtColor(anl,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(anl,7) 
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = cv2.resize(otsu, (112,112))
    img = img.reshape(112,112,1)
    Xt.append(img)

    Xt = np.asarray(Xt)/255

    model = load()

    Yt = model.predict(Xt)[0]

    w1 = np.argmax(Yt)
    d1 = Yt[w1]
    Yt[w1] = 0

    w2 = np.argmax(Yt)
    d2 = Yt[w2]
    Yt[w2] = 0

    w3 = np.argmax(Yt)
    d3 = Yt[w3]
    Yt[w3] = 0

    w4 = np.argmax(Yt)
    d4 = Yt[w4]
    Yt[w4] = 0

    w5 = np.argmax(Yt)
    d5 = Yt[w5]

    d2 = d2/d1
    d3 = d3/d1
    d4 = d4/d1
    d5 = d5/d1
    d1 = 1

    w = [w1, w2 , w3 , w4 , w5]
    psd =  [d1,d2,d3,d4,d5]


    thelist = []
    for ts in range(5):
        if psd[ts] > 0.5:
            ep = w[ts] +2304
            thelist.append(ep)

    thelist = np.sort(thelist)
    print(thelist)    
    return thelist  










# if ratio > max_ratio:
#            ratio = max_ratio
#        if ratio < 1:
#            ratio = 1



# for i in glob.glob('./*.png'):

    
#     print(predict(i))
#     print(i)
