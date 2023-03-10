import torch.nn as nn
#channels = [16, 32]
#kSizes = [5, 5]

class Network(nn.Module):
    def __init__(self, channels, kSizes, stride, padding, pool, output_flatten):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels[0],
                      kernel_size=kSizes[0], stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool, stride=stride)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                      kernel_size=kSizes[1], stride=stride, padding = padding),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool, stride=stride)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=output_flatten, out_features=256),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=256, out_features=10)

    def forward(self, t):
        t = self.layer1(t)
        t = self.layer2(t)
        #t = t.view(-1, t)
        t = t.view(t.size(0), -1)
        t = self.fc(t)
        t = self.out(t)

        return t
        
