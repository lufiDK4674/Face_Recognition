import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define prediction function
def predict_image(image, model, transform, le):
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return le.inverse_transform([pred.item()])[0]

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = joblib.load('label_encoder.pkl')

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(le.classes_))  # Adjust output layer
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Define transforms
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example usage
if __name__ == "__main__":
    # Replace 'example.jpg' with the path to your image
    image_path = "Face Recognition Dataset\Original Images\Robert Downey Jr\Robert Downey Jr_3.jpg"
    image = Image.open(image_path).convert("RGB")
    predicted_label = predict_image(image, model, transform_val, le)
    print(f"Predicted Label: {predicted_label}")