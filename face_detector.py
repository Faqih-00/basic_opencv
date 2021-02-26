import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.resize(cv2.imread('assets/eminem.jpg'), (0, 0), fx=.75, fy=.75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in face:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (250, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eye = eye_cascade.detectMultiScale(roi_gray)
    #for (ex, ey, ew, eh) in eye:
    #    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 250, 0), 2)

cv2.imshow("Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
