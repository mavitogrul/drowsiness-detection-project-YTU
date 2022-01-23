# Driver Drowsiness Detection System, Python-Dlib-OpenCV

This is our thesis project for graduation Yildiz Technical University Electronical-Communication Engineering

Students:
- Tolga Öcal tolgaocal80@gmail.com
- Ahmet Faruk Sırma ahmetfaruk2557@gmail.com
- Mavi Toğrul mavitogrul@gmail.com

The number of automobiles in traffic is increasing day by day as a result of improving 
technologies in our country and around the world. As a result of this growth, there are 
more traffic accidents. When the causes of accidents are investigated, it is discovered 
that the majority of them are caused by the driver. It is critical to keep drivers awake 
while driving in order to prevent or reduce the number of accidents. A smart design is 
to be created in this design project to reduce road accidents that occur when a driver is 
weary or sleepy. Using Python software development language and OpenCV library, 
images are taken from the video camera. And then to detect if the eyes are "Open" or 
"Closed", the images are processed by a Deep Learning Algorithm. We've aimed to 
design a visual computer system which can detect driver drowsiness in real time video 
feed, and which also rings an alarm if the driver is sleepy.

Used packages and libraries:

- OpenCV
- Dlib
- iBUG 300-W Database
- imutils
- Python 3.8

How to run this project?
- Install Python 3.8 (recommend version due to some library issues with later versions)
- Download this project directly
- Download dataset file from here => [click here](https://drive.google.com/file/d/1u6jzhaZ8tefzwYe6P4y1RyH2s7wkFUHd/view?usp=sharing)
- Move dataset file to a directory named "model"
- Open main.py file from a IDE, install required libraries
- Run file

> You can see your drowny face output on directory named "dataset" after running code.

## Project Structure:

```
│   main.py
└───model
│   │   shape_predictor_68_face_landmarks.dat
│   alarm.mp3

```
