import torch 
import torch
import torch.nn as nn 
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os, sys, random
import numpy as np

from torchmetrics.classification import BinaryRecall, BinaryF1Score
from fault_detection_model import FaultDetectionModel
from data_utils import iPhoneDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

model = FaultDetectionModel(2)

saved_model_path = 'saved_models/fault_prediction_model_ep_30.pt'

model.load_state_dict(torch.load(saved_model_path))
print('Model weights loaded from', saved_model_path)

model.eval()

dataroot = 'CV-image-assignment/images_set'  

test_preprocess = v2.Compose([
    v2.Resize(size=(224,224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_images_folder = 'CV-image-assignment/images_set/Defective/test'
test_images = os.listdir(test_images_folder)

class_map = {
    0 : 'Not Defective',
    1 : 'Defective'
}

for image_name in test_images:
    image_path = os.path.join(test_images_folder, image_name)
    image = test_preprocess(read_image(image_path))

    output = model(image.unsqueeze(0))
    preds = torch.argmax(output, dim=-1).numpy()[0]
    print(image_name, 'Label: Defective', 'Predicted:', class_map[preds])

test_images_folder = 'CV-image-assignment/images_set/Non_Defective/test'
test_images = os.listdir(test_images_folder)

for image_name in test_images:
    image_path = os.path.join(test_images_folder, image_name)
    image = test_preprocess(read_image(image_path))

    output = model(image.unsqueeze(0))
    preds = torch.argmax(output, dim=-1).numpy()[0]
    print(image_name, 'Label: Not Defective', 'Predicted:', class_map[preds])