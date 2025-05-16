import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import joblib
import cv2
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = joblib.load(f)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(le.classes_))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Define transform
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prediction function
def predict_image(image, model, transform, le):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return le.inverse_transform([pred.item()])[0]

# Streamlit UI
st.title("🧠 Real-Time Face Recognition with ResNet")

# Upload feature
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    predicted_label = predict_image(image, model, transform_val, le)
    st.success(f"🎯 Predicted Label: {predicted_label}")

# Webcam feature
if st.button("📷 Use Webcam"):
    st.warning("Press 'q' to quit webcam window.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        try:
            predicted_label = predict_image(image_pil, model, transform_val, le)
            label_text = f"Label: {predicted_label}"
        except Exception as e:
            label_text = "Face not recognized"

        # Put text on frame
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
