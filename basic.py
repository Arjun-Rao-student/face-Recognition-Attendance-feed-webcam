#------------------------images checking---------------------------------
#this basic python file used for checking face encodings and face comparing
#you can check all the attendance images

#created by ***Arjun Rao***

#-------------@code explain------------------------
# 1==>take images 'folder/image1.jpg'
# ==>take another image what you want check 'folder/image2.jpg'
# 2==>mark face locations
# 3==>take face encodings
# 5==>then compare images base on the encodings
#----------------------------------------------------------------------------------------------------------------------
#imports
import face_recognition
import cv2
import numpy as np

#-------------------------------------------------------1------------------------------------
imgs = face_recognition.load_image_file('ImagesBasic/Arjun.jpg')#give the basic image

imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/jagadeesh.jpg')#checking image
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#------------------------------------------------------------------------------------------------

#face location
#----------------------------------------------2------------------------------------
faceLoc = face_recognition.face_locations(imgs)[0]
faceLocTest = face_recognition.face_locations(imgTest)[0]

#----------------------------------------------3------------------------------------
encodeElon = face_recognition.face_encodings(imgs)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
#---------------------------------------------------------------------------------------

#face rectangle location
cv2.rectangle(imgs,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # top, right, bottom, left
#rgb colors(r,g,b)
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)# top, right, bottom, left



#---------------------------------4@Result--------------------------
results = face_recognition.compare_faces([encodeElon], encodeTest)#comparisons
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
print(results,faceDis)
cv2.imshow('Arjun.jpg',imgs)
cv2.imshow('jagadeesh.jpg',imgTest)
cv2.waitKey(0)
