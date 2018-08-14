#Importing the required libraries
import os
from PIL import Image
import numpy as np
import cv2
import pickle

#Creating the face_cascade object with the haarcascade classifier
face_cascade=cv2.CascadeClassifier(r'C:\Users\MUJ\cascades\data\haarcascade_frontalface_alt2.xml')

#Assigning the image directory to a variable named image_dir
image_dir="F:\OpenCV\images"

current_id=0 #to give id to labels to train them accrodungly
label_ids={} #dictonary of label ids
y_labels=[] #Empty list for labels
x_train=[]  #Empty list for the image value in numpy array format

#Going through the directory to find images
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label=os.path.basename(root).replace(" ","-").lower() #to get label of the images
            #Creating labels corresponding to every person
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1    
            id_ = label_ids[label] 
            
            pil_image=Image.open(path).convert("L") #gets image from path and convert it into gray scale
            image_array=np.array(pil_image,"uint8") #convert the image to numpy array to compute it
            #Detecting faces in the images
            faces= face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            #Appending the region of interest and the label ids to x_train and y_labels respectively
            for(x,y,w,h) in faces:
                roi= image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
    
with open("my_label.picle",'wb') as f:  #as file format
    #Dumping the information in the file
    pickle.dump(label_ids,f)
    
    
#Creting the recognizer object to recognize faces
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Training the object on the x_train and the array type of y_labels
recognizer.train(x_train, np.array(y_labels))
#Saving the trained model in the "trainer.yml" file in the working directory
recognizer.save("my_trainer.yml")