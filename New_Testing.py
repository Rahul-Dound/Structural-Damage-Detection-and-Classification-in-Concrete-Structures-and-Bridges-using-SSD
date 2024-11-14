import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Define the label map, combining previous map and adding any new labels from the trained model
LABEL_MAP = {
    1: 'alligator crack',
    2: 'bearing',
    3: 'cavity',
    4: 'crack',
    5: 'drainage',
    6: 'efflorescence',
    7: 'expansion joint',
    8: 'exposed rebars',
    9: 'graffiti',
    10: 'hollowareas',
    11: 'joint tape',
    12: 'protective equipment',
    13: 'restformwork',
    14: 'rockpocket',
    15: 'rust',
    16: 'spalling',
    17: 'washouts/concrete corrosion',
    18: 'weathering',
    19: 'wetspot'
}


# Load your trained SSD model
model = ssd300_vgg16(weights=None)
checkpoint_path = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/ssd_model_trained_Concrete_FF_F_f_f_A.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Load the state dict into the model
filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
model_state_dict = model.state_dict()
model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to visualize predictions
def visualize_predictions(image, boxes, labels, scores, threshold=0.1):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    for i in range(len(boxes)):
        box = boxes[i]
        label_idx = labels[i].item()
        label = LABEL_MAP.get(label_idx, 'Unknown')
        score = scores[i].item()
        if score > threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1], f'Label: {label}, Score: {score:.2f}',
                     bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 2})
    plt.axis('off')
    plt.show()

# Load the image you want to test
image_path = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/dacl10k_v2_train_1133.jpg'  # Ensure this is a valid image
image = Image.open(image_path).convert('RGB')

# Apply transformations
image_tensor = transform(image).unsqueeze(0)

# Make predictions
with torch.no_grad():
    predictions = model(image_tensor)

# Extract boxes, labels, and scores
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# Print the predictions with more detail
print("Raw Predictions:")
for i in range(len(boxes)):
    print(f"Box: {boxes[i]}, Label: {LABEL_MAP.get(labels[i].item(), 'Unknown')}, Score: {scores[i]:.2f}")

# Visualize predictions
visualize_predictions(np.array(image), boxes, labels, scores)
