# This script will detect faces via your webcam.
# Tested with OpenCV3
# Code adapted from: https://github.com/shantnu/Webcam-Face-Detect

import cv2
import time

def run_detection(scale_factor = 1.1, min_neighbors = 5, min_size = 30):
    cap = cv2.VideoCapture(0)
    if(not cap.isOpened()):  
        print('Video capture not initialized!');
        return -1;
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    while(True):
    	# Capture frame-by-frame
    	ret, frame = cap.read(); start_time = time.time();
        
    	# Our operations on the frame come here
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	# Detect faces in the image
    	faces = faceCascade.detectMultiScale(
    		gray,
    		scaleFactor=scale_factor,
    		minNeighbors=min_neighbors,
    		minSize=(min_size, min_size)
    		#flags = cv2.CV_HAAR_SCALE_IMAGE
    	); end_time = time.time();
    
    	print("Found {0} faces, in {1} seconds.".format(len(faces), end_time - start_time), end='\r')
    
    	# Draw a rectangle around the faces
    	for (x, y, w, h) in faces:
    		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    
    	# Display the resulting frame
    	cv2.imshow('frame', frame)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
