import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image

st.set_page_config(page_title="Universal AI Detector", page_icon="🔍", layout="wide")

st.title("🔍 Universal AI Detector (Open Vocabulary)")
st.write("Detect animals, humans (man/woman), vehicles, objects, and more without custom training!")

# Load pre-trained YOLO-World model
@st.cache_resource
def load_yolo_world():
    # Options: yolov8s-world.pt, yolov8m-world.pt, yolov8l-world.pt
    return YOLOWorld("yolov8s-world.pt")

model = load_yolo_world()

# Sidebar: Define any objects/classes you want the AI to detect
st.sidebar.header("🎯 Target Categories")
default_classes = "man, woman, child, dog, cat, lion, elephant, car, chair, laptop, phone"
user_classes_str = st.sidebar.text_area(
    "Enter classes to detect (separated by commas):",
    value=default_classes,
    help="Type any object or animal name here."
)

# Parse user input into a clean list of text prompts
target_classes = [c.strip().lower() for c in user_classes_str.split(",") if c.strip()]

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.30, 0.05)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and target_classes:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 1. Dynamically set custom class vocabulary inside YOLO-World
    model.set_classes(target_classes)
    
    # 2. Perform open-vocabulary inference
    results = model.predict(image, conf=confidence_threshold)
    
    # 3. Plot bounding box predictions
    res_plotted = results[0].plot()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detection View")
        st.image(res_plotted, use_container_width=True)
        
    with col2:
        st.subheader("Results & Confidence Scores")
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            
            detections.append({
                "Detected Class": label.capitalize(),
                "Confidence": f"{confidence * 100:.2f}%"
            })
        
        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.info("No target objects found above the selected confidence threshold.")
