import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x8
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 28x28x16
        self.pool = nn.MaxPool2d(2, 2)  # 14x14x16
        self.fc1 = nn.Linear(16 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = LightMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Accuracy: {100.*correct/total:.2f}%')
    
    final_accuracy = 100.*correct/total
    return model, final_accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model, accuracy = train_model()
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count}")
    print(f"Final accuracy: {accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "mnist_model.pth")