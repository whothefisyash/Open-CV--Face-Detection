import cv2
import cv2.data
import numpy as np
import streamlit as st
from PIL import Image


# for camera
def detect_faces_live():
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Haar Cascade Classifier: Haar Cascade is a machine learning-based approach that is used to identify objects in images or video streams. It works by training a cascade function on positive and negative images. The cascade function contains multiple stages, each of which contains a set of classifiers.
    # haarcascade_frontalface_default.xml: This XML file contains the trained data for the Haar cascade classifier specifically designed for frontal face detection. It consists of a set of features and weights that the classifier uses to identify whether a particular region of an image contains a face or not.
    # cv2.CascadeClassifier: This is a function provided by the OpenCV library that loads a cascade classifier from a file. It takes the path to the XML file as input and returns a cascade classifier object.
    # cv2.data.haarcascades: This is a predefined path in OpenCV where the pre-trained Haar cascade classifiers are stored. It contains XML files for various objects such as faces, eyes, and smiles.

    # opening camera
    cap=cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return
    st.write('Camera initialised.Press q to exit')

    # infinte loop to capture frames from camera
    while True:
        ret,frame=cap.read()

        # Check if frame is read correctly
        if not ret:
            st.error("Error: Could not read frame.")
            break
        st.write('Frame captured,processing')

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30,30)
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # display resulting frame with face detection
        cv2.imshow('Face Detection',frame)

        # break loop if q is pressed
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
         
        cap.release() #release camera capture
        cv2.destroyAllWindows() #close all opencv windows


#for uploaded image
def detect_faces_in_image(uploaded_image):
    
    # converting image in numpy array
    img_array=np.array(Image.open(uploaded_image))

    # creating haarcascade using pre trained xml file
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # converting image into grayscale
    gray=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

    # detecting faces in grey scale
    faces=faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30)
    )
    
    # drawing rectangle
    for(x,y,w,h) in faces:
        cv2.rectangle(img_array,(x,y),(x+w,y+h),(0,255,0),2)
    
    # display image
    st.image(img_array,channels='BGR',use_column_width=True)






# streamlit
st.title('Face Detection App')
st.subheader('Open camera or Upload Image')

if st.button('Open Camera'):
    detect_faces_live()

uploaded_image=st.file_uploader('Upload an image',type=['jpeg','jpg','png'])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)