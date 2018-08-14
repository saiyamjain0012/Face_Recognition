#Importing the required packages
import numpy as np
import cv2
import pickle

#Capturing video from the webcam 
cap = cv2.VideoCapture(0)

#Creating the face_cascade to detect faces using haarcascade classifier
face_cascade=cv2.CascadeClassifier(r'C:\Users\MUJ\cascades\data\haarcascade_frontalface_alt2.xml')

#Creating the recognizer object and makeing it read from the training.yml file
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("my_trainer.yml")

labels={}           #to give names to the ids of people to name the detected faces

#Open the pickle file in read format 
with open("my_label.picle",'rb') as f:
    og_labels = pickle.load(f)
    labels={ v:k for k,v in og_labels.items()} #reversing the dictonary to get the format of id as key and image as value
while(True):
    # Capture frame-by-frame and converting it into grayscale form
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detect faces in the captured frame image
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #Extracting the region of interest for both gray and color images
        roi_gray= gray[y:y+h,x:x+w] #y start to y end, x start to x end
        roi_color= frame[y:y+h,x:x+w]
        
        #Predicting the name for the detected face
        id_, conf = recognizer.predict(roi_gray)
        #If the confidence of face recognized is between the range of 4 and 85 then show the prediction
        if conf>=4 and conf <= 85:
            #Declaring font type
            font = cv2.FONT_HERSHEY_SIMPLEX
            #Getting the name from the labels[id_]
            name = labels[id_]
            #Definig the color as white for the text
            color = (255, 255, 255)
            #Defining the thickness of the rectangular frame
            stroke = 1
            #Putting the predicted name above the rectangular box around the detected face
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        
        #Defining rectangle around the detected face
        color=(255,0,0) #in bgr
        stroke=2 #thickness
        end_cord_x=x+w #End coordinates for x
        end_cord_y=y+h #End coordinates for y
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    #Display the resulting frame
    cv2.imshow('frame',frame)
    #Closing the webcam when the user presses the button 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close the program
cap.release()
cv2.destroyAllWindows()