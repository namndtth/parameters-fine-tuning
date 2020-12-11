import torch
import timm

class Classifier(torch.nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, n_classes)
    def forward(self, x):
        x = self.model(x)
        return x