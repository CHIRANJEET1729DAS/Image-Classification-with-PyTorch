# Developed by Aayush Sharma :)
# repo: https://github.com/aayushsharma-io/Image-Classification-with-PyTorch/
# feel free to mess this up :)
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the neural network architecture
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Changed convolution layers to lazy convolution layers.
        # Perks: Lazy layers defer the initialization of parameters until the input is passed through. 
        # This avoids the need to explicitly define input sizes beforehand, making the model more flexible.
        # It also helps in memory efficiency as the actual memory is allocated only when the layers are used.
        # Added BatchNormalisation for faster convergence
        # Added Dropout for regularisation
        # Added adaptive_pool to avoid any tensor size mismatches
        
        self.conv1 = nn.LazyConv2d(6, 5)
        self.bn1 = nn.BatchNorm2d(6)  # Added Batch Normalization
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.LazyConv2d(16, 5)
        self.bn2 = nn.BatchNorm2d(16)  # Added Batch Normalization

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # Adaptive pooling instead of fixed size
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.5)  # Added Dropout for regularization
        
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)  # Added Dropout for regularization
        
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        
        x = torch.flatten(x, 1)  # Flatten feature maps
        
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Dropout after fully connected layer
        
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Dropout after fully connected layer
        
        x = self.fc3(x)
        return x


# Load and preprocess the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Initialize the model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(2):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # Print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
