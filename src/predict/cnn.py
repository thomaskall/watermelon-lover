import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Literal
from PIL import Image
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from collect import WatermelonData
from utils import get_device

# Define a custom model
#### Previously CombinedModel
class SpectrogramModel(nn.Module):
    def __init__(self, metadata_dim=1, output_dim=1):
        super(SpectrogramModel, self).__init__()
        self.cnn = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Freeze convolutional layers
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Remove the final classification layer
        cnn_output_dim = self.cnn.classifier[0].in_features
        self.cnn.classifier = nn.Identity()

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + metadata_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, image, metadata):
        cnn_features = self.cnn(image)
        combined_features = torch.cat((cnn_features, metadata), dim=1)
        output = self.fc(combined_features)
        return output
    
    def load_model(self, data_type: Literal["tap", "sweep"]):
        model_path = f'models/{data_type}/best.pth'
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def predict(self, image, weight: float):
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0)
            metadata = weight.unsqueeze(0)
            output = self(image, metadata)
            return output.item()

def predict_from_path(data: WatermelonData):
    """
    Make a prediction using a spectrogram image path and weight value.
    
    Args:
        model: Trained SpectrogramModel instance
        image_path: Path to the spectrogram image
        weight: Weight value of the watermelon
        device: Device to run the model on (defaults to available device)
    
    Returns:
        float: Predicted brix score
    """
    if device is None:
        device = get_device()

    model = SpectrogramModel(metadata_dim=1, output_dim=1)
    model.load_model(data.cycle_type)
    model.to(device)
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(data.spectrogram_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    weight_tensor = torch.tensor([[data.weight]], dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image, weight_tensor)
    
    return prediction.item()