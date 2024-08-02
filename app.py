from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

video = cv2.VideoCapture(0)

# load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FaceEmotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 100, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
            cv2.putText(img, output, (x + 6, y - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        return img


def main():
    # Face Analysis Application #
    st.title("Emotion Detection Application provide by KAMAL SINGH CHARAN")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown('''
      # About Me \n 
        Hey this is **KAMAL SINGH CHARAN**. \n

        Also check us out on Social Media
        - [Portfolio](https://main--nikmal8.netlify.app/)
        - [LinkedIn](https://www.linkedin.com/in/kamal-charan21/)
        - [Github](https://github.com/Nikmal8)
        ''')

    if choice == "Home":
        html_temp_home1 = """<div style=padding:10px">
                                            <h4 style=text-align:center;">
                                            Emotion detection application using OpenCV, Custom CNN model and Streamlit.
                                            </h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Real time Emotions recognition.

                 """)

    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your Eomtions")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=FaceEmotion)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style=padding:10px">
                                    <h4 style=text-align:center;">
                                    Emotions detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """ 
                        <div style=padding:10px">
                        <h5 style=text-align:center;">This Application is developed by KAMAL SINGH CHARAN using Streamlit, Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or want to comment just write a mail at KC621843@GMAIL.COM. </h5>
                        <h6 style=text-align:center;">Thanks for Visiting</h6>
                        </div>
                        <br></br>
                        <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    else:
        pass


if __name__ == "__main__":
    main()
