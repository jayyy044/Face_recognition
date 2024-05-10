<h2>Face Recognition and Detection</h2>
<h3>Using Haar Cascades Algorithm and DeepFace Library</h3>
<h5>This is a Face detection and recognition script that I wrote. I use the Haar Cascades algorithm, which is commonly used in machine learning object detection. The .xml file used is a pretrained model that works efficiently and quickly. It processes each frame and classifies each window as positive or negative where positive represents something belonging to what we are looking for while negative implies the opposite. This model's accuracy depends on the angle of the face compared to the camera. Since we are using the frontal face version minimal angles result in the highest accuracy. Following this, I use the DeepFace library to compare reference images taken of the user with what the camera is seeing. This library allows access to another pretrained model boasting a strong accuracy. 
</h5>
<hr>
<h2>How to use </h2>
<h4>Step 1: Installing Libraries</h4>
<hr>

```shell
$ pip install deepface opencv-python tf-keras
```

<h4>Step 2: Registering</h4>
<hr>
<h5>When You first run the program there are no reference faces thus you cannot use the face detection. Thus you need to click register and register 5 images of your self. Take one looking at the camera then one with your hit slightly tilted up, down, left and then right</h5>
<h4>Step 3: Recognition</h4>
<hr>
<h5>Now look at the camera and ensure that there is lots of light and enjoy your face recognition! If the recognition takes longer then 30 seconds try restarting the program and ensurign you have lots of light on your face when registering your pictures</h5>

<h4>Demo</h4>
<h5>Click this link to see a live demo on how this script works</h5>
<img src="https://github.com/jayyy044/Face_recognition/blob/main/Demo/FaceRecognition_Demo.mp4" alt="Demo">
<img src="https://github.com/jayyy044/Face_recognition/blob/main/Demo/FaceRecognition_Demo.mp4" alt="Demo">
<img src="https://github.com/jayyy044/Face_recognition/blob/main/Demo/FaceRecognition_Demo.mp4" alt="Demo">
<img src="https://github.com/jayyy044/Face_recognition/blob/main/Demo/FaceRecognition_Demo.mp4" alt="Demo">
<img src="https://github.com/jayyy044/Face_recognition/blob/main/Demo/FaceRecognition_Demo.mp4" alt="Demo">
