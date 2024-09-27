import cv2
import numpy as np
import av
import PoseModule as pm
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = pm.poseDetector()
        self.count = 0
        self.direction = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        maxh, maxw, _ = img.shape

        # Process frame with pose estimation
        img = self.detector.findPose(img, False)
        lmList = self.detector.findPosition(img, False)

        # Check if landmarks are detected
        if len(lmList) != 0:
            # Calculate angle
            angle = self.detector.findAngle(img, 12, 14, 16)

            per = np.interp(angle, (55, 145), (100, 0))
            bar = np.interp(angle, (55, 145), (400, 50))

            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if self.direction == 0:
                    self.count += 0.5
                    self.direction = 1
            if per == 0:
                color = (0, 255, 0)
                if self.direction == 1:
                    self.count += 0.5
                    self.direction = 0

            # Draw Bar
            cv2.rectangle(img, (maxw-100, maxh-400), (maxw - 50, maxh - 50), color, 3)
            cv2.rectangle(img, (maxw - 100, maxh - int(bar)), (maxw - 50, maxh - 50), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (maxw - 105, maxh - 430), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Draw Curl Count
            cv2.rectangle(img, (0, 0), (150, 100), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(self.count)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Bicep Curl Counter", 
        layout="wide", 
        menu_items = {
            'About': 'Made with ❤️ by Dhruv Singh'
        }
    )

    # Main Title and Description
    st.title("🏋️ Bicep Curl Counter App")
    st.markdown("### Perform Bicep Curls and see the rep count in real-time!")

    # Sidebar for additional information
    with st.sidebar:
        st.header("Instructions")
        st.write("""
            1. Select Camera Device (if you have multiple cameras connected)
            2. Position yourself in front of the camera as shown in the image below.
        """)
        st.image("bicep_curl.jpg", caption="Correct posture for bicep curls", use_column_width=True)
        st.write("""
                 * Ensure you stand sideways to the camera with your right side visible.
        """)
        st.write("""
            3. Perform bicep curls.
            4. Your rep count will be displayed on the screen.
        """)

    # Video Stream and Feedback

    st.header("Live Feed")
    webrtc_streamer(
        key="exercise-counter",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

    # Footer with your name
    st.markdown(
        """
        ***
        Made with ❤️ by Dhruv Singh
        """
    )

if __name__ == "__main__":
    main()
