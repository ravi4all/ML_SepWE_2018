import cv2

data = cv2.CascadeClassifier('haarcascade_smile.xml')
capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = data.detectMultiScale(gray)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),5)
        cv2.imshow('result',img)
        if cv2.waitKey(10) & 0xff == 27:
            break
    else:
        print("Camera not working")

capture.release()
cv2.destroyAllWindows()