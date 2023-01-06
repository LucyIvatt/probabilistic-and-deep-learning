import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


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

            # initialize our softmax classifier
            nn.Linear(in_features=180, out_features=5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Input x has dimensions B x 1 x 48 x 48, B is batch size
        x = self.ConvolutionalLayers(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        # Output has dimensions B x 5
        return x


def classify(images):
    # Determine which device the input tensor is on
    device = torch.device("cuda" if images.is_cuda else "cpu")

    model = SymbolClassifier()
    # Move to same device as input images
    model = model.to(device)
    # Load network weights
    model.load_state_dict(torch.load(
        'weights.pkl', map_location=torch.device(device)))
    # Put model in evaluation mode
    model.eval()

    # Optional: do whatever preprocessing you do on the images
    # if not included as tranformations inside the model

    with torch.no_grad():
        # Pass images to model, get back logits or probabilities
        output = model(images)

    # Select class with highest probability for each input
    predicted_classes = torch.argmax(output, 1)

    # Return predicted classes
    return predicted_classes


image_tensors = []
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

for i in ["0000", "0001", "0002", "0003"]:
    img = Image.open("images/classify_testing/" + i + ".tif")
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    image_tensors.append(img)

cat = torch.cat(image_tensors)

output = classify(cat)
print(output)
