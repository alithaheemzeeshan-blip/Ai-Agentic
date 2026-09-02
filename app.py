import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI Animal Detector", page_icon="🐾")

st.title("🐾 AI Animal Detector")
st.write("Upload an image to detect animals and view confidence scores.")

# Load your custom fine-tuned model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run YOLO inference
    results = model(image)
    
    # Display detection bounding boxes on image
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detections", use_container_width=True)
    
    # Extract and show confidence scores
    st.subheader("Results & Confidence Scores")
    detections = []
    
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detections.append({
            "Animal": label.capitalize(),
            "Confidence": f"{confidence * 100:.2f}%"
        })
    
    if detections:
        st.dataframe(detections)
    else:
        st.info("No animals detected.")