#Face detection using haarcascades classifiers

path = "/home/gb/CDSAML/Images/1.jpg"

# Import the Images module from pillow
from PIL import Image
# Open the image by specifying the image path.
image_file = Image.open(path)
# the default
image_file.save("/home/gb/CDSAML/Images/edited.jpg", quality=25)

import cv2
import numpy as np

path2 = "/home/gb/CDSAML/Images/edited.jpg"
face_classifier = cv2.CascadeClassifier('/home/gb/CDSAML/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
resized = cv2.imread(path2)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
''' Our classifier returns the ROI of the detected face as a tuple, 
It stores the top left coordinate and the bottom right coordiantes'''
faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)
'''When no faces detected, face_classifier returns and empty tuple'''
print("Number of faces detected:",len(faces))

if len(faces)==0:
    print("No faces found")
'''We iterate through our faces array and draw a rectangle over each face in faces'''

for (x,y,w,h) in faces:
    cv2.rectangle(resized, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()