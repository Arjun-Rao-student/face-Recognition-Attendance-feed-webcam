#
# ------------------------------------------------my first project--------------------------------
# i think this the very good
# but you have any  the implementation please contact
# my github page
#created by Arjun
#import part
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
#known path
path = 'ImagesAttendence'
#unknownPath = 'unknown'
images =[]
classNames = []
myList = os.listdir(path)
print(myList)
# name generating using images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#---------------------------------attendance-------------------------------------------------------------------------
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:#give your  .csv file name
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  nameList:#my wondring loop ifyou have save the multiple 
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')
#face encoding 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #print(encodeList)
    return encodeList
encodeListKnown = findEncodings(images)

print('I am done bro proceed......')

cap = cv2.VideoCapture(0)
#-------------------------------------------------while loop start------------------------------------
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            markAttendance(name)
    # unknown faces saving
    #it needs more space so please disable it
    
   # else:
    #        cv2.imwrite(os.path.join(unknownPath, f"unknown_{len(os.listdir(unknownPath)) + 1}.jpg"), img)
#----------------------------------------------------------while loop ending--------------------------------------------

#------------------------------capturing cam------------------------------------------------------------------------------
    cv2.imshow('MyCam',img)
    cv2.waitKey(1)

#----------------------------------Thank You using----------------------------------------------------------------------







