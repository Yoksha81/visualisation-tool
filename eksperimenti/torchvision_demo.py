from torchvision.models import resnet18, ResNet18_Weights

from model_visualizer import visualize


model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

visualize(model)
