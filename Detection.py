import threading
import cv2
from deepface import DeepFace
import os
import shutil

#Turning on a camera
camera_on = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Set the resolution of the captured video frames
camera_on.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
camera_on.set(cv2.CAP_PROP_FRAME_WIDTH, 700)


#counter to track the amount of frames processed 
frames_count = 0

#creating folder that will host the reference images
output_folder = 'Registered_face'
#deletes registered images on program start ups
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

face_match = False
Run_program = True
tracking = False
save_images = False
registering = False
no_face_registered = False
empty_df = False

#this function is actually doing all the checking for a match 
def check_face(frame):
    #we send it frame which is a single frame from the camera
    global face_match
    try:
        reference_img = DeepFace.find(img_path= frame, db_path= "./Registered_face")
        '''we comapre the picture of the person to the reference 
        image provided and if its a match the verified key result in True
        Deep face is using a pretrained model'''
        empty_dfs = [df for df in reference_img if df.empty]
        if empty_dfs:
            face_match = False
        else:
            face_match = True
    except ValueError:
        face_match = False

def button_callback(event, x, y, flags, param):
    global tracking  # Make sure to use the global variable
    global save_images
    global img_counter
    global registering
    global no_face_registered
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 435 and 490 <= y <= 530 and not registering:

            if not os.path.exists(output_folder):
                no_face_registered = True    
            else:
                tracking = True
                save_images = False
                registering = False 
        elif 20 <= x <= 280 and 550 <= y <= 590:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            tracking = False
            save_images = True
            registering = True
            no_face_registered = False
            if img_counter < 5:  # Check if img_counter is less than 5 before incrementing
                 img_counter += 1
        elif 700 <= x <= 799 and 1 <= y <= 45:  # Quit button in the top right corner
            # Exit the application
            cv2.destroyAllWindows()
            exit()

# Create named window and set mouse callback
cv2.namedWindow("video")
cv2.setMouseCallback("video", button_callback)

#cv2.data.haarcascades: pre-trained XML files (Haar cascade classifiers) for different object detection tasks.
#The .xml file: pre-trained model for frontal face detection
#CascadeClassifier: a class used for object detection. It initializes a Cascade Classifier object, 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_counter = 0

#we set up while loop to constantly take in frames from the camera
while Run_program:
    #ret a bool created to check if a captured a frame
    #frame: this is the actual frame we captured 
    ret, frame = camera_on.read()

    #this places a conditional check if a captured a frame 
    if ret:
        '''We place the object detection for the face we are trying to 
        identify here since as soon as the camera identifies a frame
        we want to detect a face in that frame'''
        '''The following code uses the haarcascade frontal face
        object detection for the frames being seen video capturing
        devices'''
        #this converts the frames being seen to grey since its easier to process 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (5, 5), (200, 40), (255, 255, 255), -1)
        if no_face_registered:
            cv2.rectangle(frame, (5, 5), (370, 50), (255, 255, 255), -1)
            cv2.putText(frame, "No Face Registered !",(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)
        elif registering:
            cv2.rectangle(frame, (5, 5), (410, 50), (255, 255, 255), -1)
            cv2.putText(frame, f"Registering Image {img_counter}/5 !",(20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)
            if img_counter==5:
                registering = False
            if save_images and img_counter<6:
                img_name = f"face_{img_counter}.jpg"
                img_path = os.path.join(output_folder, img_name)
                cv2.imwrite(img_path, frame)
                print(f"Face saved: {img_path}")
                save_images = False
        elif tracking:
            '''We have set up the bool tracking to make it so only when the user clicks
            the key w then face matching will occur'''
            '''we then place another check to ensure we that we are 
            wait 30 frames before the next verification'''
            if frames_count % 30 == 0:
                try:
                    '''Using threading we pass a copy of the frame to the check_face
                    function and we pass a copy so we can continue to process more frames
                    without having to stop during the verfiication process'''
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()
                except ValueError:
                    pass
            #Here we just increment the counter so we can use the condition for 30 frames
            frames_count += 1

            #this just sets the location of the matched text
            if face_match:
                cv2.rectangle(frame, (5, 5), (200, 40), (255, 255, 255), -1)
                cv2.putText(frame, "Match!",(50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)
            else:
                cv2.rectangle(frame, (5, 5), (200, 40), (255, 255, 255), -1)
                cv2.putText(frame, "No Match!",(23, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
        else:
            cv2.rectangle(frame, (1, 1), (300, 50), (255, 255, 255), -1)
            cv2.putText(frame, "Select an Option",(17, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)       
        cv2.rectangle(frame, (20, 490), (435, 530), (255, 255, 255), -1)
        cv2.putText(frame, "Check Face Recognition", (40, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 550), (280, 590), (255, 255, 255), -1)
        cv2.putText(frame, "Register Face", (40, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (700, 1), (799, 45), (255, 255, 255), -1)
        cv2.putText(frame, "Quit", (710, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        #this allows the text to be over the video 
        cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
#this closes all the windows and frees up any associated resources 
cv2.destroyAllWindows()

