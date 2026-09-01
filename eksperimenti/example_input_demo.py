import torch
import torch.nn as nn

from model_visualizer import visualize


class MaliModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        # Python uslov nad dimenzijom služi samo kao primer
        # konstrukcije zbog koje symbolic_trace može da ne uspe.
        if x.shape[0] > 0:
            return self.linear(x)
        return x


model = MaliModel()
x = torch.randn(1, 16)

visualize(
    model,
    example_inputs=x
)
