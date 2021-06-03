number_of_training_images = p

for number in range(0,p):
  x=2701+number
  np.random.seed(12345)
  
  #Taking the ground truth image as input to find its fourier amplitude and give it as an input to our function
  #image = imageio.imread("/content/drive/My Drive/BTP/train/label/"+str(number)+".bmp", as_gray=True)
  image = imageio.imread("/content/drive/My Drive/BTP_2/train/Image_Dataset/"+str(number)+".bmp", as_gray=True)
  write_location="/content/drive/My Drive/BTP_2/" + str(number) + ".tiff"

  #Setting noise coefficient as 3
  alpha=3
  #Padding the input image
  image_padded = np.pad(image, 128, 'constant')
  #Finding the fourier transform of the image
  magnitudes_oversampled = np.abs(np.fft.fft2(image_padded))
  #Adding the noise to the image
  square=np.square(magnitudes_oversampled)
  #noise = np.random.normal(np.zeros((512,512)),np.sqrt(np.diag(np.diag(np.square(alpha)*square))))
  noise = np.random.normal(0,1,size=(512,512))
  noise= alpha*magnitudes_oversampled*noise
  
  magnitudes_oversampled= np.sqrt(np.abs(np.add(square,noise)))
  #Start Modification
  magnitudes_oversampled = gdal.Open("/content/drive/My Drive/BTP_2/train/Dataset_16bit/"+str(number)+".tiff").ReadAsArray()
  #end Modification
  mask = np.pad(np.ones((256,256)), 128, 'constant')
  minimum=sys.maxsize

#Taking  different random initializations to choose the one with the lwast rror
  for i in range(0,50):
    angle=np.random.rand(*magnitudes_oversampled.shape)
    angle=angle*np.pi*2 
    result_oversampled_init = fienup_phase_retrieval(angle,magnitudes_oversampled,beta=0.9,steps=50,mask=mask,verbose=True)
    result_oversampled_init=np.abs(np.fft.fft2(result_oversampled_init))
    error= (np.square(result_oversampled_init-magnitudes_oversampled)).mean(axis=None)

    minimum=min(minimum,error)
    if minimum==error:
      initialization=angle
  
  result_oversampled = fienup_phase_retrieval(initialization,magnitudes_oversampled,mask=mask,beta=0.9,steps=1000,verbose=True)
  imageio.imwrite(write_location, result_oversampled)
  im = Image.open(write_location) 
  im1 = im.crop((128, 128, 384, 384))
  im1.save(write_location)
  print("Image ",number," of 50 done" )
  
