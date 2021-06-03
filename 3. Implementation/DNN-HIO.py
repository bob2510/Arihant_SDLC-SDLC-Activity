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
import imageio
import matplotlib.pyplot as plt
import sys
from PIL import Image

import HIO.py
import Image_generator.py
import rotation_check.py

model=load_model("unet_membrane.hdf5")
global initialization
for number in range(37,300):
    x=number+2701
    image = imageio.imread("/content/drive/My Drive/BTP/Image_Dataset/"+str(x)+".bmp", as_gray=True)
    image=np.double(image)
    #image = imageio.imread("/content/drive/My Drive/BTP/test/einstein.bmp", as_gray=True)
    image_padded = np.pad(image, 128, 'constant')
    magnitudes_oversampled = np.abs(np.fft.fft2(image_padded))
    mask = np.pad(np.ones((256,256)), 128, 'constant')
    image_HIO = imageio.imread("/content/drive/My Drive/BTP/test/predict/"+ str(number) + ".bmp", as_gray=True)
    image_HIO=np.double(image_HIO)
    #image_HIO = imageio.imread("/content/drive/My Drive/BTP/0.bmp", as_gray=True)
    write_location = "/content/drive/My Drive/BTP/test/DNN_input/0.bmp"
    
    for iterations in range(0,20):
          image_padded = np.pad(image, 128, 'constant')
          image_HIO_padded = np.pad(image_HIO, 128, 'constant') 
          angle=np.angle(np.fft.fft2(image_HIO_padded))
          result_oversampled = fienup_phase_retrieval(angle,magnitudes_oversampled,beta=0.9,steps=50,mask=mask,verbose=False,read=1)
          imageio.imwrite(write_location, result_oversampled)
          im = Image.open(write_location) 
          im1 = im.crop((128, 128, 384, 384))
          im1.save(write_location)
          im1.save('/content/drive/My Drive/BTP/test/DNN-HIO/'+str(iterations)+'.bmp')
          
          #To correct the orientation of the image
          #read_file="/content/drive/My Drive/BTP/test/einstein.bmp"
          read_file="/content/drive/My Drive/BTP/Image_Dataset/"+str(x)+".bmp"
          write_file=write_location
          error = np.array([])
          original = Image.open(read_file)
          HIO_image = Image.open(write_file)
          HIO_image_rotated = HIO_image.rotate(180, resample=0, expand=0)
          error=rmsdiff(original,HIO_image)
          error_rotated = rmsdiff(original,HIO_image_rotated)
          if error>error_rotated:
            HIO_image_rotated.save(write_file)
            HIO_image_rotated.save('/content/drive/My Drive/BTP/test/DNN-HIO/'+str(iterations)+'.bmp')
            

          #Now we will find the DNN
          testGene = testGenerator("/content/drive/My Drive/BTP/test/DNN_input",num_image=1)
          results = model.predict_generator(testGene,1,verbose=0)
          saveResult('/content/drive/My Drive/BTP/test/HIO_input',results)

          image_DNN = imageio.imread("/content/drive/My Drive/BTP/test/HIO_input/0.bmp", as_gray=True)
          imageio.imwrite('/content/drive/My Drive/BTP/test/DNN-HIO/'+str(iterations)+'DNN.bmp', image_DNN)

          image_HIO = imageio.imread("/content/drive/My Drive/BTP/test/HIO_input/0.bmp", as_gray=True)

    original = Image.open(read_file)
    output =  Image.open('/content/drive/My Drive/BTP/test/DNN-HIO/'+str(iterations)+'.bmp')
    error=rmsdiff(original,output)
    im1.save('/content/drive/My Drive/BTP/test/DNN-HIO_result_with_error/'+str(error)+'_'+str(number)+'.bmp')
    print(number)


