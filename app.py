from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTConfig
import os
import gzip
import shutil

# Path to the compressed model file
compressed_model_path = 'model/vit_model.pth.gz'
model_path = 'model/vit_model.pth'

# Check if the decompressed model already exists
if not os.path.exists(model_path):
    # Decompress the model if it doesn't exist
    with gzip.open(compressed_model_path, 'rb') as f_in:
        with open(model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

app = Flask(__name__)

# Initialize the model with the same configuration as your saved model
num_classes = 2  # Replace with the number of classes used during training
model_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
model_config.num_labels = num_classes
model = ViTForImageClassification(config=model_config)

# Load the state dictionary from your checkpoint
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define class labels
class_labels = ['Gingivitis', 'Healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    image = Image.open(file.stream)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor).logits
        _, predicted_class = outputs.max(1)
    
    class_idx = predicted_class.item()
    predicted_label = class_labels[class_idx]
    
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
