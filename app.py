import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page Config
st.set_page_config(page_title="AI Agent for Detection", page_icon="🚀", layout="wide")

st.title("🚀 AI Agent for Detection")
st.write("Created By Zeeshan Thaheem")

# Load model
@st.cache_resource
def load_animal_model():
    return YOLO("best.pt")

model = load_animal_model()

# Session State for history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls
st.sidebar.header("Settings & Actions")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.10, 
    max_value=1.00, 
    value=0.40, 
    step=0.05
)

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

# File Uploader
uploaded_files = st.file_uploader(
    "Upload images...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if not any(item["name"] == uploaded_file.name for item in st.session_state.history):
            image = Image.open(uploaded_file).convert("RGB")
            
            results = model.predict(image, conf=confidence_threshold)
            
            res_bgr = results[0].plot()
            res_rgb = res_bgr[:, :, ::-1]
            
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                
                detections.append({
                    "Animal": label.capitalize(),
                    "Confidence": f"{confidence * 100:.2f}%"
                })
            
            st.session_state.history.append({
                "name": uploaded_file.name,
                "image": res_rgb,
                "detections": detections
            })

# Display History
if st.session_state.history:
    st.write(f"### Total Processed Images: {len(st.session_state.history)}")
    
    for idx, item in enumerate(reversed(st.session_state.history), start=1):
        st.markdown("---")
        st.subheader(f"🖼️ {item['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(item["image"], use_container_width=True)
            
        with col2:
            st.write("**Detected Animals:**")
            if item["detections"]:
                st.dataframe(item["detections"], use_container_width=True)
            else:
                st.info("No animals detected above threshold.")
else:
    st.info("Upload an image above to start detecting!")
