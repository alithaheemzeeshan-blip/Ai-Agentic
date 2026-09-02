import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="AI Animal Detector", page_icon="🐾", layout="wide")

st.title("🐾 AI Animal Detector")
st.write("Upload an image to detect animal species from your custom trained model.")

# 2. Load your custom animal model (best.pt)
@st.cache_resource
def load_animal_model():
    return YOLO("best.pt")

model = load_animal_model()

# 3. Sidebar Confidence Threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.10, 
    max_value=1.00, 
    value=0.40, 
    step=0.05
)

# 4. Image Uploader
uploaded_file = st.file_uploader("Upload an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image natively in RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run YOLO inference
    results = model.predict(image, conf=confidence_threshold)
    
    # FIX: results[0].plot() outputs BGR numpy array.
    # Convert BGR -> RGB slice [..., ::-1] so colors render accurately in Streamlit.
    res_bgr = results[0].plot()
    res_rgb = res_bgr[:, :, ::-1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detections")
        st.image(res_rgb, use_container_width=True)
        
    with col2:
        st.subheader("Detected Animals")
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            
            detections.append({
                "Animal": label.capitalize(),
                "Confidence": f"{confidence * 100:.2f}%"
            })
            
        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.info("No animals detected above the selected confidence threshold.")
