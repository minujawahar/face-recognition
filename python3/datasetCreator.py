import cv2
face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
cam=cv2.VideoCapture(0)
sampleNum=0
id=input('enter user id')
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
         sampleNum=sampleNum+1
         roi_gray = gray[y:y+h, x:x+w]
         cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",roi_gray)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = img[y:y+h, x:x+w]
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',img)
    k=cv2.waitKey(30)&0xff
    if k==27:
        break
    if sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
