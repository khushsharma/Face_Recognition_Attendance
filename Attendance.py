from xml.etree.ElementPath import find
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
className = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curr_image = cv2.imread(f'{path}/{cl}')      #read images from path
    images.append(curr_image)                    #insert images into image list
    className.append(os.path.splitext(cl)[0])    #split .jpg from names


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #image converted to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv' , 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')


encodeListKnown = findEncodings(images)
cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,.25) #scale down the images
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame , faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc # for rectangle on webcam
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2) ,(0,255,0),2)
            #cv2.rectangle(img,(x1,y2-35) ,(x2,y2),(0,255,0),cv2.FILLED)
            #for putting name below rectangle
            markAttendance(name)
            cv2.putText(img,name,(x1+6 , y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)