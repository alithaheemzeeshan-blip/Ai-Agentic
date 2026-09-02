import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="AI Object & Animal Detector",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 AI Object & Animal Detector")
st.write("Upload an image to perform object detection with accurate bounding boxes and confidence scores.")

# 2. Dynamic Model Loading
@st.cache_resource
def load_detection_model():
    # Priority: Load custom-trained best.pt if present in repo
    if os.path.exists("best.pt"):
        st.sidebar.success("Loaded Model: Custom Fine-Tuned Weights (best.pt)")
        return YOLO("best.pt")
    else:
        st.sidebar.info("Loaded Model: Pre-trained Checkpoint (yolov8x.pt)")
        return YOLO("yolov8x.pt")

model = load_detection_model()

# 3. Sidebar Configuration Controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.10, 
    max_value=1.00, 
    value=0.40, 
    step=0.05
)

# 4. File Upload Interface
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image natively with PIL (ensures RGB channel accuracy)
    image = Image.open(uploaded_file).convert("RGB")
    
    # Perform YOLO inference
    results = model.predict(image, conf=confidence_threshold)
    
    # Plot bounding box overlay
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
