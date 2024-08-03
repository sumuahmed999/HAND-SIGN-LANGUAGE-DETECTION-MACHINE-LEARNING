import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

# Initialize the hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

# Streamlit app
st.title("Hand Sign Language Detection")

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 30px; 
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 200px;
        margin-left: -300px;
    }
    .copyright-text {
        padding-top: 250px;
        left: 10px;
        font-size: 18px;
        color: #fff;
    }
    .title-text{
        padding-top: 0px;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown('<h1 class="title-text">HAND SIGN LANGUAGE DETECTION</h1>', unsafe_allow_html=True)

st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
            background: rgb(218,12,131);
            background: linear-gradient(32deg, rgba(218,12,131,1) 0%, rgba(99,51,242,1) 46%, rgba(20,0,71,1) 100%);
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
    unsafe_allow_html=True,
)


# st.markdown(
#         """
#         <style>
#         .video-container {
#             position: relative;
#             padding-bottom: 56.25%; /* 16:9 aspect ratio */
#             height: 0;
#             overflow: hidden;
#             max-width: 100%;
#             background: #fff;
#             margin: 10px 0;
#         }
#         .video-container iframe,
#         .video-container object,
#         .video-container embed {
#             position: absolute;
#             top: 0;
#             left: 0;
#             width: 80%;
#             height: 80%;
#         }
#         </style>
#         <div class="video-container">
#             <iframe src="https://www.youtube.com/embed/eLfqvVbvyzA" frameborder="1" allowfullscreen></iframe>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

def process_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hands, img = detector.findHands(frame)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((150, 150, 3), np.uint8) * 255
            imgCrop = frame[y - 20:y + h + 20, x - 20:x + w + 20]

            aspectRatio = h / w
            if aspectRatio > 1:
                k = 150 / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, 150))
                wGap = math.ceil((150 - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = 150 / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (150, hCal))
                hGap = math.ceil((150 - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            imgWhite = cv2.resize(imgWhite, (150, 150))
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(frame, (x - 30, y - 20 - 50),(x - 20 + 90, y - 20 - 50 + 50), (255, 170, 2), cv2.FILLED)
            cv2.putText(frame, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 170, 2), 4)

        stframe.image(frame, channels="BGR")

    cap.release()


if st.sidebar.button('Use Webcam'):
    process_webcam()


    def process_video():
        cap = cv2.VideoCapture(0)
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hands, img = detector.findHands(frame)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((150, 150, 3), np.uint8) * 255
                imgCrop = frame[y - 20:y + h + 20, x - 20:x + w + 20]

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = 150 / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, 150))
                    wGap = math.ceil((150 - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = 150 / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (150, hCal))
                    hGap = math.ceil((150 - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                imgWhite = cv2.resize(imgWhite, (150, 150))
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                results.append(labels[index])

        cap.release()
        return results

if st.sidebar.button('Run on Video'):
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        if st.button('Process Uploaded Video'):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_filepath = tmp_file.name

            results = process_video(tmp_filepath)
            os.remove(tmp_filepath)

            st.write("Detection Results:")
            st.write(results)

st.markdown('''
                  # About The Project \n 
    A hand sign language detection system, which has the potential to significantly improve communication for individuals with hearing impairments. The goal
    of this project is to develop a robust system that can accurately detect and interpret hand sign languages
    from videos and generate corresponding hand sign videos from text input.\n

    ## Tools I used
        - OpenCV
        - Keras
        - Matplotlib
        - Google Colab
        - Tensorflow
        - Streamlit
''')

st.sidebar.markdown('<p class="copyright-text">DEVELOPED BY SUMU AHMED</p>', unsafe_allow_html=True)
