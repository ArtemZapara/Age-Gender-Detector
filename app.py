import streamlit as st
import cv2
import os
from utils import age_gender_detector

weigthsDir = "./weights"

faceProto = os.path.join(weigthsDir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(weigthsDir, "opencv_face_detector_uint8.pb")

ageProto = os.path.join(weigthsDir, "age_deploy.prototxt")
ageModel = os.path.join(weigthsDir, "age_net.caffemodel")

genderProto = os.path.join(weigthsDir, "gender_deploy.prototxt")
genderModel = os.path.join(weigthsDir, "gender_net.caffemodel")


faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

st.title("Webcam Live Feed")
run = st.checkbox("Run")
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

while run:
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameCopy = age_gender_detector(faceNet, ageNet, genderNet, frame)
    FRAME_WINDOW.image(frameCopy)
else:
    st.write("")