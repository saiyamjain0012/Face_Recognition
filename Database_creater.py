# Importing the required packages
import cv2
import numpy as np

#Using cv2.VideoCapture(0) to fetch the video from the webcam 
vidcap = cv2.VideoCapture(0)

#Declaring a variable count to count the number of images saved
count = 0
 
#Creating the face_cascade object to detect faces in the frames captured
face_cascade = cv2.CascadeClassifier(r'C:\Users\MUJ\cascades\data\haarcascade_frontalface_alt2.xml')  

#Getting into an infinte loop
while(True):
    # Capture frame-by-frame image from the webcam feed
    ret, frame = vidcap.read()
    
    #Converting the captured frame into grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detecting faces in the image frame
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    
    #Getting the roi i.e. region of interest for both gray and colored frame
    for (x,y,w,h) in faces:
        roi_gray= gray[y:y+h,x:x+w] #y start to y end, x start to x end
        roi_color= frame[y:y+h,x:x+w]
        #Printing image count
        print("image-",count)
        #Saving the image into the working directory by appending count number
        cv2.imwrite("frame_%d.jpg" % count,frame)
        
    #Increasing the count by 1      
    count=count+1
         
    # Drawing a rectangle around the region of interest
    color=(255,0,0) #in bgr
    stroke=2 #thickness
    end_cord_x=x+w #end coordinate for x
    end_cord_y=y+h #end coordinate for y
    cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    #Displaying the frame with the rectangle on it
    cv2.imshow('frame',frame)
    
    #Breaking out of the loop when the image count reaches 1000
    if count==1000:
        break
    #Breaking out of the loop if the user presses the key 'q' 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
  
 #Closing the webcam and ending the program
vidcap.release()
cv2.destroyAllWindows()