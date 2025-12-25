import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = mobilenet_v3_small(pretrained=pretrained)

        # hanya ambil feature extractor (tanpa avgpool & classifier)
        self.features = model.features

    def forward(self, x):
        # Indeks layer untuk mengambil feature map pada skala berbeda
        feature_indices = {3, 6, 12}
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in feature_indices:
                outputs.append(x)
        return outputs
