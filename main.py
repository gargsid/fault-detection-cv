import torch
import torch.nn as nn 
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os, sys, random
import numpy as np

from torchmetrics.classification import BinaryRecall, BinaryF1Score
from focal_loss.focal_loss import FocalLoss

from tqdm import tqdm
import wandb 

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

class iPhoneDataset(Dataset):
    def __init__(self, dataroot, transforms, mode='train'):
        self.mode = mode # train, val, test
        self.transforms = transforms

        self.dataroot = dataroot

        if mode == 'train':
            defective_root = os.path.join(dataroot, 'Defective/train')
            non_defective_root = os.path.join(dataroot, 'Non_Defective/train')
        else:
            defective_root = os.path.join(dataroot, 'Defective/test')
            non_defective_root = os.path.join(dataroot, 'Non_Defective/test')

        self.defective_paths = [os.path.join(defective_root, image_name) for image_name in os.listdir(defective_root)]
        self.non_defective_paths = [os.path.join(non_defective_root, image_name) for image_name in os.listdir(non_defective_root)]

        self.files = self.defective_paths + self.non_defective_paths
        self.labels = [1 for _ in range(len(self.defective_paths))] + [0 for _ in range(len(self.non_defective_paths))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.transforms(read_image(self.files[idx]))
        label = self.labels[idx]
        return image, label
    
config = {
    'lrate' : 1e-5,
    'batch_size' : 32,
    'epochs' : 10,
    'weight_decay' : 1e-3,   
}

wandb.init(
    project='ML-Models',
    config=config
)

dataroot = 'CV-image-assignment/images_set' 

class FaultDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet_weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.resnet_weights)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.linear_1 = nn.Linear(in_features=1024, out_features=256, bias=True)
        self.bn_2 = nn.BatchNorm1d(256)
        self.linear_2 = nn.Linear(in_features=256, out_features=32, bias=True)
        self.bn_3 = nn.BatchNorm1d(32)
        self.logits = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.bn_1(self.resnet(x)))
        x = self.gelu(self.bn_2(self.linear_1(x)))
        x = self.gelu(self.bn_3(self.linear_2(x)))
        x = self.logits(x)
        return x

# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

model = FaultDetectionModel(num_classes=2).to(device)

train_preprocess = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_preprocess = v2.Compose([
    v2.Resize(size=(224,224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# preprocess = weights.transforms()

train_dataset = iPhoneDataset(dataroot, train_preprocess, 'train')
test_dataset = iPhoneDataset(dataroot, test_preprocess, 'test')

class_counts = np.array([len(train_dataset.non_defective_paths), len(train_dataset.defective_paths)])
weights_per_class = 1 / class_counts
weights_per_class[0] += 0.02 
weights_per_class[1] -= 0.03
weights_per_image = [weights_per_class[label] for label in train_dataset.labels]

sampler = WeightedRandomSampler(weights=weights_per_image, num_samples=len(train_dataset), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
# train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

criterion = FocalLoss(gamma=0.7)
sigmoid = torch.nn.Sigmoid()

lrate = config['lrate']
weight_decay = config['weight_decay']
epochs = config['epochs']

optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

recall_fn = BinaryRecall()
F1Score_fn = BinaryF1Score()

best_recall = 0. 
best_f1 = 0. 

for epoch in range(epochs):
    
    epoch_loss = 0.
    train_preds = []
    train_labels = []

    optimizer.param_groups[0]['lr'] = lrate* (1 - epoch/epochs) 

    wandb.log({'lrate' : lrate * (1 - epoch/epochs)})
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(sigmoid(outputs), labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1)
        train_preds += preds 
        train_labels += labels

    epoch_loss /= len(train_loader)
    train_preds = torch.tensor(train_preds)
    train_labels = torch.tensor(train_labels)
    train_recall = recall_fn(train_preds, train_labels)
    train_f1 = F1Score_fn(train_preds, train_labels)

    wandb.log({
        'train_loss' : epoch_loss,
        'train_recall' : train_recall,
        'train_f1' : train_f1
    })

    print(f'epoch:{epoch}/{epochs} train_loss: {epoch_loss} train_recall: {train_recall} train_f1:{train_f1}')
    print('labels; num_okay:', (train_preds==0).sum(), 'num_def:', (train_preds==1).sum())
    print('preds; num_okay:', (train_labels==0).sum(), 'num_def:', (train_labels==1).sum())

    # if epoch % 10 == 0:
    model.eval()
    val_loss = 0. 
    val_preds = []
    val_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(sigmoid(outputs), labels)
        val_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1)
        val_preds += preds 
        val_labels += labels
    
    val_loss /= len(test_loader)
    val_preds = torch.tensor(val_preds)
    val_labels = torch.tensor(val_labels)
    val_recall = recall_fn(val_preds, val_labels)
    val_f1 = F1Score_fn(val_preds, val_labels)
    print(f'epoch:{epoch}/{epochs} val_loss: {val_loss} val_recall: {val_recall} val_f1:{val_f1}')

    if best_recall < val_recall or (best_recall == val_recall and best_f1 < val_f1):
        best_recall = val_recall
        best_f1 = val_f1
        torch.save(model.state_dict(), 'saved_models/fault_prediction_model.pt')

    wandb.log({
        'val_loss' : val_loss,
        'val_recall' : val_recall,
        'val_f1' : val_f1
    })

    model.train()

model.load_state_dict(torch.load('saved_models/fault_prediction_model.pt'))
model.eval()
for images, val_labels in test_loader:
    outputs = model(images)
    val_preds = torch.argmax(outputs, dim=-1)

val_recall = recall_fn(val_preds, val_labels)
val_f1 = F1Score_fn(val_preds, val_labels)
print('recall:', val_recall, 'F1:', val_f1)

wandb.finish()