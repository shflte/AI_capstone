import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim

class CaptchaDataset(Dataset):
    # Your CaptchaDataset class implementation...

def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def main():
    TRAIN_PATH = '.'  # Update with the path to your training data
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your dataset and data loader
    train_data = [['captcha1.png', '0123'], ['captcha2.png', '4567']]  # Update with your data
    train_dataset = CaptchaDataset(train_data, root=TRAIN_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Initialize the model
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=40)  # 40 outputs for 4 digits over 10 possibilities each
    model = model.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()  # Use appropriate loss for your problem
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, device, train_loader, criterion, optimizer, NUM_EPOCHS)

if __name__ == '__main__':
    main()
