import cv2
import time
import os
import sys
import uuid
from poster.encode import multipart_encode
from poster.streaminghttp import register_openers
from poster.encode import MultipartParam
import urllib2
import json
import contextlib


def saveFrame(frame):
    id = uuid.uuid4()   
    path = "save/{0}.jpg".format(id)
    savePath = os.path.realpath(path)  
    print 'Saving to {0}'.format(savePath)  
    cv2.imwrite(savePath ,frame)      
    return savePath 

def uploadToFaceOff(path, faces):
    try:
        register_openers()
        datagen, headers = multipart_encode({"file": open(path, "rb")})
        request = urllib2.Request("http://dk-mud:5004", datagen, headers)
        with contextlib.closing(urllib2.urlopen(request)) as x:
            x.read()
            x.close()  
    except:
        print "Unexpected error:", sys.exc_info()[0]
     
cascPath = "./haarcascade_frontalface_default.xml"    
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    facesInFrame = len(faces)   
     
    if facesInFrame > 0:         
       video_capture.grab() 
       retval, image =  video_capture.retrieve()
       filePath = saveFrame(image)
       cv2.waitKey(5)
       uploadToFaceOff(filePath, faces)  
       time.sleep(5)     
       os.remove(filePath); 
         
    cv2.imshow('Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()