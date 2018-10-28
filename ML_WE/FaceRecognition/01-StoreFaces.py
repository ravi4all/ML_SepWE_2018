import cv2
import numpy as np

data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
facedata = []
while True:
    ret, img = capture.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = data.detectMultiScale(gray,1.3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),5)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))

            if len(facedata) < 100:
                facedata.append(face)
            print(len(facedata))

        cv2.imshow('result',img)
        if cv2.waitKey(10) & 0xff == 27 or len(facedata) >= 100:
            break
    else:
        print("Camera not working")

facedata = np.asarray(facedata)
np.save('user_1.npy', facedata)

capture.release()
cv2.destroyAllWindows()
