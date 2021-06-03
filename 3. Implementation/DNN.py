
from __future__ import print_function
from keras.models import load_model
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
import numpy as np
from keras.models import *
from keras.layers import Conv2D 
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LambdaCallback
import glob
 
def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    #print("First downsampling")
    #print("Input Size:" , keras.print_tensor(inputs))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1=BatchNormalization(momentum=0.95,epsilon=0.005)(conv1)
    #print("1st Convolution:",keras.print_tensor(conv1))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1=BatchNormalization(momentum=0.95,epsilon=0.005)(conv1)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1=BatchNormalization(momentum=0.95,epsilon=0.005)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #print("After 1st Maxpooling:",keras.print_tensor(pool1))
    
    #print("Second Downsampling")
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2=BatchNormalization(momentum=0.95,epsilon=0.005)(conv2)
    #print("1st Convolution:",keras.print_tensor(conv2)) 
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #print("Second Convolution:",keras.print_tensor(conv2))
    conv2=BatchNormalization(momentum=0.95,epsilon=0.005)(conv2)
    #print("After 2nd BN:",keras.print_tensor(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #print("After 2nd Maxpooling:",keras.print_tensor(pool2))
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3=BatchNormalization(momentum=0.95,epsilon=0.005)(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3=BatchNormalization(momentum=0.95,epsilon=0.005)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4=BatchNormalization(momentum=0.95,epsilon=0.005)(conv4)   
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4=BatchNormalization(momentum=0.95,epsilon=0.005)(conv4)
    conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
 
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5=BatchNormalization(momentum=0.95,epsilon=0.005)(conv5)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5=BatchNormalization(momentum=0.95,epsilon=0.005)(conv5)
    conv5 = Dropout(0.5)(conv5)
    #print("Before first upsampling:",keras.print_tensor(conv5))
    #print("First Upsampling layer")

    up6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6=BatchNormalization(momentum=0.95,epsilon=0.005)(up6)
    #print("1st UpConvolution:",keras.print_tensor(up6)) 
    merge6 = concatenate([conv4,up6], axis = 3)
    #print("1st Concatenation:",keras.print_tensor(merge6)) 
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6=BatchNormalization(momentum=0.95,epsilon=0.005)(conv6)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6=BatchNormalization(momentum=0.95,epsilon=0.005)(conv6)
 
    up7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7=BatchNormalization(momentum=0.95,epsilon=0.005)(up7)

    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv6=BatchNormalization(momentum=0.95,epsilon=0.005)(conv6)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7=BatchNormalization(momentum=0.95,epsilon=0.005)(conv7)
 
    up8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8=BatchNormalization(momentum=0.95,epsilon=0.005)(up8)

    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8=BatchNormalization(momentum=0.95,epsilon=0.005)(conv8)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8=BatchNormalization(momentum=0.95,epsilon=0.005)(conv8)
 
    
    up9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9=BatchNormalization(momentum=0.95,epsilon=0.005)(up9)

    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9=BatchNormalization(momentum=0.95,epsilon=0.005)(conv9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9=BatchNormalization(momentum=0.95,epsilon=0.005)(conv9)

    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
 
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    merge10 = Add()([inputs,conv10])
    print("Output:",keras.print_tensor(merge10))
 
    #model = Model(input = inputs, output = merge10)                                                                                  
    model = Model(inputs, merge10)                                                                                  
    
 
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['accuracy'])
    
    #model.summary()
 
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
 
    return model
