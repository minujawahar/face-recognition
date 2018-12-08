# Face Recognition using opencv
#### code by Minu Jawahar 2018
#### developed at Data Science Academy https://datascience.one

Face Recognition is an application of computer vision.The coding steps for face recognition are :

  1)Collecting Data :Different images of a person is captured and stored in a dataset.It is used for training image with its corresponding identity.Along with images,we take some details such as id,name etc.It can be used further to train the model to identity specific persons.Create folder (eg:dataSet) to store images.
  
  2)Training Data: All the images stored are converted to numpy array.For futher computing the datas should be in numpy array.
   All the images and corresponding details on a person is converted to numpy array and trained with recognizer used.Create folder to store yml file .Yml file stores numpy array of immages and Ids.
   
  3)Detecting Face:Last stage is to predict whether the captured image is same as that of the ones stored.
  
  ## OpenCv Face Recognizer
  OpenCV has three built in face recognizers 
  
  1)EigenFaces Face Recognizer Recognizer -  cv2.createEigenFaceRecognizer()
  
  2)FisherFaces Face Recognizer Recognizer - cv2.createFisherFaceRecognizer()
  
  3)Local Binary Patterns Histograms (LBPH) Face Recognizer - cv2.createLBPHFaceRecognizer()
  
 Eigenfaces and Fisherfaces are both affected by light .To overcome those issues we use LBPH Face Recognizer.LBPH alogrithm try to find the local structure of an image and it does that by comparing each pixel with its neighboring pixels.
 
 ## Required Modules
  
  1)cv2: is OpenCV module for Python which we will use for face detection and face recognition.
  
  2)os: It is used to get directories and file names.
  
  3)numpy:To convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays only.
