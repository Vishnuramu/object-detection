import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="YOLOv8 App", layout="centered")
st.title("🧠 YOLOv8 Object Detection")

mode = st.radio("Select Mode:", ["📷 Image", "🎞️ Video"])

# ──────────────── IMAGE DETECTION ──────────────── #
if mode == "📷 Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        t_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(t_path, "wb") as f:
            f.write(uploaded_file.read())
        st.image(t_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            results = model(t_path)
            results[0].save(filename="detected.jpg")
            st.image("detected.jpg", caption="Detection Result", use_column_width=True)

# ──────────────── VIDEO DETECTION ──────────────── #
elif mode == "🎞️ Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        temp_input_path = os.path.join(tempfile.gettempdir(), video_file.name)
        with open(temp_input_path, "wb") as f:
            f.write(video_file.read())

        st.video(temp_input_path)

        if st.button("Run Detection on Video"):
            st.info("Processing video... Please wait ⏳")
            cap = cv2.VideoCapture(temp_input_path)

            # Prepare output path
            output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                20.0,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, save=False, stream=False)
                annotated = results[0].plot()
                out.write(annotated)

            cap.release()
            out.release()

            st.success("✅ Detection complete!")

            # Offer download
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Processed Video", f, file_name="detected_output.mp4")
