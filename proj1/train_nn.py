import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim

import os
from PIL import Image

class CaptchaDataset(Dataset):
    def __init__(self, root, label_file, transform=None):
        self.root = root
        self.transform = transform
        self.images = [f"{i}.png" for i in range(1, len(open(os.path.join(root, label_file), 'r').readlines()) + 1)]
        self.labels = []

        with open(os.path.join(root, label_file), 'r') as f:
            for line in f:
                label = line.strip()
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'captcha_img', self.images[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.zeros((4, 10))
        for i, char in enumerate(label):
            label_tensor[i, int(char)] = 1

        return image, label_tensor


def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs = outputs.view(-1, 4, 10)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def main():
    TRAIN_PATH = '.'
    LABEL_FILE = 'label_nn.txt'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CaptchaDataset(root=TRAIN_PATH, label_file=LABEL_FILE, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=40)
    model = model.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, device, train_loader, criterion, optimizer, NUM_EPOCHS)

    model_save_path = './models/model_{:03d}.pth'.format(NUM_EPOCHS)
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()
