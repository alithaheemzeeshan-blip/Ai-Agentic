import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI Object & Animal Detector", page_icon="🐾")

st.title("🐾 AI Detector (Humans, Animals & Objects)")
st.write("Upload an image to detect humans, animals, vehicles, and objects with confidence scores.")

# Load highest-accuracy standard YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8x.pt")

model = load_model()

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run prediction
    results = model.predict(image, conf=confidence_threshold)
    
    # Plot bounding boxes
    res_plotted = results[0].plot()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detections")
        st.image(res_plotted, use_container_width=True)
        
    with col2:
        st.subheader("Detected Objects")
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
            st.info("No objects detected above the chosen threshold.")
