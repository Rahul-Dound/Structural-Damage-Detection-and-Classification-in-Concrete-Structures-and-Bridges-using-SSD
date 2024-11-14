import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set CUDA to run synchronously for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Define paths
base_image_dir = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/split_5/img'
base_annotation_dir = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/split_5/ann'
meta_file_path = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/meta.json'
model_save_path = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/ssd_model_trained_Concrete_FF_F_f_f_B.pth'  # Path to save the trained model
pretrained_model_path = '/content/drive/MyDrive/EDI 7TH SEM/Concrete St/ssd_model_trained_Concrete_FF_F_f_f_A.pth'  # Path to load your earlier trained model

# Load meta.json to create class mapping
def load_class_mapping(meta_file_path):
    with open(meta_file_path, 'r') as f:
        meta_data = json.load(f)
    class_map = {cls['title']: i + 1 for i, cls in enumerate(meta_data['classes'])}
    print(f"Class mapping loaded: {class_map}")
    return class_map

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, class_map):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_map = class_map
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.images)

    def clean_filename(self, filename):
        # Remove unwanted characters like '(1)' from filenames
        return filename.replace(' (1)', '')

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Clean the filename to match with annotations
        clean_img_name = self.clean_filename(img_name)

        annotation_file = os.path.join(self.annotation_dir, clean_img_name + '.json')

        # Check if annotation file exists, if not, skip this image
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}. Skipping.")
            return None, None

        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)

        boxes = []
        labels = []
        for obj in annotation_data['objects']:
            points = obj['points']['exterior']
            x_coordinates, y_coordinates = zip(*points)
            x_min, y_min = min(x_coordinates), min(y_coordinates)
            x_max, y_max = max(x_coordinates), max(y_coordinates)

            # Ensure the bounding box has a valid area
            if x_max - x_min > 0 and y_max - y_min > 0:
                boxes.append([x_min, y_min, x_max, y_max])
                class_id = self.class_map.get(obj['classTitle'], -1)
                labels.append(class_id)
            else:
                print(f"Invalid bounding box in {img_name}: {x_min, y_min, x_max, y_max}")

        if len(boxes) == 0:
            print(f"No valid bounding boxes found for image: {img_name}. Skipping.")
            return None, None

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])

        image = self.transform(image)
        return image, target

    def transform(self, image):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])
        return transform(image)

def custom_collate_fn(batch):
    # Filter out images with None targets (i.e., skipped images due to missing annotations)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    images, targets = zip(*batch)
    return list(images), list(targets)

# Load or initialize model
def load_combined_model(pretrained_model_path):
    model = ssd300_vgg16(weights='DEFAULT')  # Initialize SSD300 model with VGG16 backbone

    # Load your fine-tuned model on top of it
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading custom pre-trained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training function
def train_on_split(image_dir, annotation_dir, class_map, pretrained_model_path, model_save_path):
    model = load_combined_model(pretrained_model_path)
    model.train()
    model.to(device)

    train_dataset = CustomDataset(image_dir, annotation_dir, class_map)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # Dynamic LR scheduling

    num_epochs = 100
    patience = 10
    best_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for images, targets in train_loader:
            if len(images) == 0:
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            epoch_losses.append(losses.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Check for improvement
        if avg_loss < 1:
            # Save the model and stop training if the loss goes below 1
            torch.save(model.state_dict(), model_save_path)
            print(f"Training stopped. Model saved with loss {avg_loss:.4f}")
            break  # Stop training immediately if loss < 1

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs} epochs.")

        scheduler.step(avg_loss)

        # Early stopping
        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model if not already saved due to low loss
    if avg_loss >= 1:
        torch.save(model.state_dict(), model_save_path)
        print(f"Final model saved with loss {avg_loss:.4f}")

# Training on all splits
def train_on_all_splits():
    class_map = load_class_mapping(meta_file_path)
    img_dir = base_image_dir
    ann_dir = base_annotation_dir
    print(f"Training on split: split_5")
    train_on_split(img_dir, ann_dir, class_map, pretrained_model_path, model_save_path)

train_on_all_splits()
