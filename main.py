import torch
import torch.nn as nn 
from torchvision.transforms import v2

from torch.utils.data import DataLoader
import numpy as np

from torchmetrics.classification import BinaryRecall, BinaryF1Score

import wandb 

from fault_detection_model import FaultDetectionModel
from data_utils import iPhoneDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

config = {
    'lrate' : 1e-4,
    'batch_size' : 32,
    'epochs' : 30,
    'weight_decay' : 1e-2,   
}

wandb.init(
    project='ML-Models',
    config=config
)

dataroot = 'CV-image-assignment/images_set' 

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

train_dataset = iPhoneDataset(dataroot, train_preprocess, 'train')
test_dataset = iPhoneDataset(dataroot, test_preprocess, 'test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

criterion = nn.CrossEntropyLoss()

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
        loss = criterion(outputs, labels)
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

    model.eval()
    val_loss = 0. 
    val_preds = []
    val_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
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
        torch.save(model.state_dict(), f'saved_models/fault_prediction_model_ep_{epochs}.pt')

    wandb.log({
        'val_loss' : val_loss,
        'val_recall' : val_recall,
        'val_f1' : val_f1
    })

    model.train()

model.load_state_dict(torch.load(f'saved_models/fault_prediction_model_ep_{epochs}.pt'))
model.eval()
for images, val_labels in test_loader:
    outputs = model(images)
    val_preds = torch.argmax(outputs, dim=-1)

val_recall = recall_fn(val_preds, val_labels)
val_f1 = F1Score_fn(val_preds, val_labels)
print('recall:', val_recall, 'F1:', val_f1)

wandb.finish()