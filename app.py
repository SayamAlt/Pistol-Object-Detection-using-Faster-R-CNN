import os
os.environ["PYTORCH_JIT"] = "0"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import torch, time
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

st.set_page_config(layout="wide", page_title="🔫 Pistol Object Detection App")

try:
    font = ImageFont.truetype("arial.ttf", size=22)
except:
    font = ImageFont.load_default()
    
transform = transforms.ToTensor()

class_names = ['bg', 'Pistol']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@st.cache_resource
def load_detection_model(classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    
    model.load_state_dict(torch.load("pistol_object_detector.pth", map_location=torch.device('cpu'))) # Load the trained model weights
    model.to(device) # Move the model to the correct device
    model.eval() # Set the model to evaluation mode
    return model

model = load_detection_model(class_names)

st.title("🔫 Pistol Object Detection App")
st.write("📤 Upload an image, and the model will detect and highlight pistols in it.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
submit_button = st.button("🚀 Detect Pistols")
clear_button = st.button("🗑️ Clear")

if submit_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📤 Uploaded Image", use_container_width=True)

        with st.spinner("Detecting pistols... please wait ⏳"):
            progress_bar = st.progress(0)
            for percent in range(0, 100, 20):
                time.sleep(0.1)
                progress_bar.progress(percent)
        
            # Preprocess the image
            image_tensor = transform(image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                prediction = model(image_tensor)[0]

            # Draw bounding boxes
            detected_image = image.copy()
            draw = ImageDraw.Draw(detected_image)

            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            scores = prediction["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if score > 0.85 and class_names[label] == "Pistol":
                    box = box.tolist()
                    draw.rectangle(box, outline="red", width=4)
                    text = f"{class_names[label]} ({score*100:.1f}%)"
                    text_x, text_y = box[0], box[1] - 30  # slightly above box
                    text_bg_color = (255, 0, 0)  # red background
                    text_color = (255, 255, 255)  # white text

                    # Measure text size for background box
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Adjust Y position to avoid overflow
                    text_x = box[0]
                    text_y = max(0, box[1] - text_height - 8)  # ensures label stays inside image
        
                    # Draw background box
                    draw.rectangle(
                        [text_x, text_y, text_x + text_width + 4, text_y + text_height + 2],
                        fill=text_bg_color
                    )

                    draw.text((text_x + 2, text_y + 1), text, fill=text_color, font=font)

        with col2:
            st.image(detected_image, caption="🎯 Detected Pistols", use_container_width=True)
        
if clear_button:
    st.session_state.clear()
    st.rerun()

st.markdown("---")
st.markdown("**Model:** Faster R-CNN (ResNet50 backbone) | **Framework:** PyTorch | **Deployed with:** Streamlit")