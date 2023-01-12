import torch
import torch.nn as nn


class SymbolClassifier(nn.Module):
    def __init__(self):
        super(SymbolClassifier, self).__init__()

        self.ConvolutionalLayers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3,
                      stride=1, padding=1),  # output size = B × 12 × 48 x 48
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # output size = B × 12 × 24 x 24
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            # Second Convolutional layer
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3,
                      stride=1, padding=1),  # output size = B × 24 × 24 x 24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # output size = B × 24 × 12 x 12
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout(0.2),
        )

        self.MLP = nn.Sequential(
            nn.Linear(in_features=12*12*24, out_features=300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features=300, out_features=180),
            nn.BatchNorm1d(180),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features=180, out_features=5),
            nn.LogSoftmax(dim=1)  # probability distribution
        )

    def forward(self, x):
        # Input x has dimensions 64 x 1 x 48 x 48
        x = self.ConvolutionalLayers(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)  # 64 x 5
        return x


def classify(images, weight_location):
    device = torch.device("cuda" if images.is_cuda else "cpu")
    model = SymbolClassifier()
    model = model.to(device)

    # Load network weights
    model.load_state_dict(torch.load(
        weight_location, map_location=torch.device(device)))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(images)

    # Select class with highest probability for each input
    predicted_classes = torch.argmax(output, 1)

    # Return predicted classes
    return predicted_classes
