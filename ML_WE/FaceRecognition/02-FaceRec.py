import numpy as np
import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_1 = np.load("user_1.npy").reshape(100,50*50*3)
face_2 = np.load("user_1.npy").reshape(100,50*50*3)

users = {
    0 : "User_1",
    1 : "User_2"
}

font = cv2.FONT_HERSHEY_COMPLEX

labels = np.zeros((200,1))
labels[:100,:] = 0.0
labels[100:,:] = 1.0

data = np.concatenate([face_1, face_2])

def dist(x1,x2):
    return np.sqrt(((x2 - x1) ** 2).sum())

def knn(x, train):
    n = train.shape[0]
    distance = []
    for i in range(n):
        distance.append(dist(x,train[i]))

    distance = np.asarray(distance)
    index = np.argsort(distance)
    sortedLabels = labels[index][:5]
    count = np.unique(sortedLabels, return_counts=True)
    return count[0][np.argmax(count[1])]

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray,1.3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),5)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            label = knn(face.flatten(), data)
            text = users[int(label)]
            cv2.putText(img,text,(x,y),font,2,(0,0,255),2)

        cv2.imshow('result',img)
        if cv2.waitKey(10) & 0xff == 27:
            break
    else:
        print("Camera not working")


capture.release()
cv2.destroyAllWindows()