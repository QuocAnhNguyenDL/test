import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
from torchinfo import summary

class FireDetec(nn.Module):
    def __init__(self):
        super(FireDetec, self).__init__()
        self.tanh = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 4 * 4, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # Quantization
        x = self.quant(x)
        # First Layer 128x128 -> 64x64
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool1(x)
        x = self.dropout(x)
        # Second Layer 64x64 -> 32x32
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool2(x)
        # x = self.dropout(x)
        # Third Layer 32x32 -> 16x16
        x = self.conv3(x)
        x = self.tanh(x)
        x = self.pool3(x)
        # x = self.dropout(x)
        # Fourth Layer 16x16 -> 8x8
        x = self.conv4(x)
        x = self.tanh(x)
        x = self.pool4(x)
        # x = self.dropout(x)
        # Fifth Layer 8x8 -> 4x4
        x = self.conv5(x)
        x = self.tanh(x)
        x = self.pool5(x)
        x = self.dropout(x)
        # Sixth Layer 4x4 -> Flatten() -> FC
        x = self.flatten(x)
        x = self.fc(x)
        # Dequantization
        x = self.dequant(x)
        return F.softmax(x, dim=1)

class Model (nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net= FireDetec()
    
    def forward(self,x):
        return self.net(x)

def info(model, verbose=False):
    n_p = sum(p.numel() for p in model.parameters())
    n_g = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f'{"Layer":<5} {"Name":<40} {"Gradient":<9} {"Parameters":<12} {"Shape":<20} {"Mean":<10} {"Std":<10}')
        for i, (name, p) in enumerate(model.named_parameters()):
            print(f'{i:<5} {name:<40} {str(p.requires_grad):<9} {p.numel():<12} {str(list(p.shape)):<20} {p.mean():.3g} {p.std():.3g}')
    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients")


if __name__ == '__main__':
    model = Model()
    model.eval()

    summary(model)
