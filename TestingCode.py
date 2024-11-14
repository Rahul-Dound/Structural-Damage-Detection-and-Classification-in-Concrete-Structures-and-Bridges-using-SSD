import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Define the label map
LABEL_MAP = {
    1: 'corrosion',
    2: 'crack',
    3: 'freelime',
    4: 'leakage',
    5: 'spalling',
    6: 'damage'
}

# Load your SSD model
model = ssdlite320_mobilenet_v3_large(weights=None)
checkpoint_path = '/content/drive/MyDrive/Edi 7th Sem/ssd_model_Efinal.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Ensure compatibility with CPU/GPU
checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Debugging: Check what keys are in the checkpoint and the model
print("Checkpoint keys:", checkpoint_state_dict.keys())
print("Model keys:", model.state_dict().keys())

# Filter and load the state dict
filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
model_state_dict = model.state_dict()
model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict)
model.eval()

# Image transformations (for model input)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to visualize predictions
def visualize_predictions(image, boxes, labels, scores, threshold=0.1):  # Lowered the threshold to 0.1
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for i in range(len(boxes)):
        box = boxes[i]
        label_idx = labels[i].item()
        label = LABEL_MAP.get(label_idx, 'Unknown')
        score = scores[i].item()
        if score > threshold:  # Filter boxes with score above threshold
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1], f'Label: {label}, Score: {score:.2f}',
                     bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 2})
    plt.axis('off')
    plt.show()

# Load the damage image
image_path = '/content/drive/MyDrive/Edi 7th Sem/spa(6).jpeg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')

# Apply transformations
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Make predictions
with torch.no_grad():
    predictions = model(image_tensor)

# Extract boxes, labels, and scores
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# Debugging: Print the predictions with more detail
print("Raw Predictions:")
for i in range(len(boxes)):
    print(f"Box: {boxes[i]}, Label: {LABEL_MAP.get(labels[i].item(), 'Unknown')}, Score: {scores[i]}")

# Visualize predictions
visualize_predictions(np.array(image), boxes, labels, scores)