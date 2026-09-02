import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="AI Detector", page_icon="🐾", layout="wide")

st.title("🐾 AI Object & Animal Detector")

@st.cache_resource
def load_detection_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    return YOLO("yolov8x.pt")

model = load_detection_model()

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file bytes to a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Force BGR to RGB color conversion
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run YOLO prediction on the corrected image
    results = model.predict(pil_image, conf=confidence_threshold)
    
    res_plotted = results[0].plot()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detections Output")
        st.image(res_plotted, use_container_width=True)
        
    with col2:
        st.subheader("Results & Scores")
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({
                "Object": label.capitalize(),
                "Confidence": f"{confidence * 100:.2f}%"
            })
            
        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.info("No objects detected above the selected confidence threshold.")
