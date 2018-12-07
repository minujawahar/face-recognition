# code by Minu Jawahar 2018
# developed at Data Science Academy https://datascience.one
import cv2

face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
cam=cv2.VideoCapture(0)
rec=cv2.createLBPHFaceRecognizer();

rec.load("recognizer/trainingData.yml")

id2=""
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = img[y:y+h, x:x+w]
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         id,conf=rec.predict(roi_gray)
         if id==1:
            id2="Minu"
        
         
        
         cv2.putText(img, id2, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255)) 
         
    cv2.imshow('img',img)
    k=cv2.waitKey(30)&0xff
    if k==27:
        break
cam.release()
cv2.destroyAllWindows()

 
